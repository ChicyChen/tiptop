"""TiPToP as a persistent service, mirroring the cap-x real-robot contract.

The operator sets the task AT THE ROBOT: franky_service sends `prompt` on every
frame and increments `episode_id`. Nothing here may take a task from a CLI flag,
and one process must serve many episodes (openpi_client connects once and never
reconnects, so exiting would permanently break the driver's socket).
"""

from __future__ import annotations

import json
import pathlib
import time

import numpy as np
import pytest

from tiptop.franky.franky_client import EpisodeSuperseded, FrankyClient
from tiptop.franky.serve_franky import (
    TiptopFrankyService,
    _episode_dir,
    _existing_attempts,
)


def _frame(ep, prompt, q=None):
    return {
        "episode_id": ep,
        "prompt": prompt,
        "observation/joint_position": np.zeros(7) if q is None else np.asarray(q),
    }


def _client():
    return FrankyClient(port=0, driver_hz=15.0)


class TestTaskComesFromTheWire:
    """The whole point: no --task-instruction anywhere."""

    def test_prompt_is_read_from_every_frame(self):
        c = _client()
        c._on_frame(_frame(7, "pick up the larger block"))
        assert c.current_prompt() == "pick up the larger block"
        assert c.current_episode() == 7

    def test_a_changed_prompt_is_adopted(self):
        c = _client()
        c._on_frame(_frame(7, "task A"))
        c._on_frame(_frame(7, "task B"))
        assert c.current_prompt() == "task B"

    def test_blank_prompt_does_not_erase_the_task(self):
        c = _client()
        c._on_frame(_frame(7, "task A"))
        c._on_frame(_frame(7, "   "))
        assert c.current_prompt() == "task A"

    def test_service_module_takes_no_task_argument(self):
        """A CLI task flag would mean the operator types it — the reported bug."""
        import inspect

        from tiptop.franky.serve_franky import main

        params = inspect.signature(main).parameters
        assert "task_instruction" not in params
        assert not any("task" in p for p in params), sorted(params)


class TestEpisodeBoundaries:
    def test_new_episode_id_drops_queued_motion(self):
        c = _client()
        c._on_frame(_frame(7, "t"))
        c._enqueue([np.zeros(8)] * 50)
        assert c._cursor.pending() == 50
        c._on_frame(_frame(8, "t"))
        assert c._cursor.pending() == 0, "stale plan must not drive the arm"

    def test_new_episode_aborts_the_running_plan(self):
        c = _client()
        c._on_frame(_frame(7, "t"))
        c.begin_plan()
        c._abort_if_superseded()             # still valid
        c._on_frame(_frame(8, "t"))
        with pytest.raises(EpisodeSuperseded):
            c._abort_if_superseded()

    def test_motion_after_a_boundary_is_refused(self):
        c = _client()
        c._on_frame(_frame(7, "t"))
        c.begin_plan()
        c._on_frame(_frame(8, "t"))
        with pytest.raises(EpisodeSuperseded):
            c.execute_joint_impedance_path(
                np.linspace(0, 0.2, 10)[:, None] * np.ones((1, 7)),
                None,
                [0.02] * 10,
            )

    def test_first_episode_is_not_treated_as_a_boundary(self):
        """episode_id 0 is VALID, and a first frame must not drop the queue.

        _on_frame legitimately CONSUMES up to action_horizon rows, so compare
        against that rather than the full queue.
        """
        c = _client()
        c._enqueue([np.zeros(8)] * 40)
        c._on_frame(_frame(0, "t"))
        assert c.current_episode() == 0
        assert c._cursor.pending() == 40 - c.action_horizon, (
            "a first frame consumes a chunk but must not DROP the plan"
        )


class TestEndOfEpisodeStopsWork:
    """Frame silence is the only end signal: the driver's policy.reset() clears
    local state only and leaves the socket open."""

    def test_end_episode_invalidates_the_plan(self):
        c = _client()
        c._on_frame(_frame(7, "t"))
        c.begin_plan()
        c.end_episode("no frames for 12s")
        with pytest.raises(EpisodeSuperseded):
            c._abort_if_superseded()

    def test_begin_plan_refuses_a_dead_episode(self):
        """Otherwise each new plan re-arms itself against a finished episode."""
        c = _client()
        c._on_frame(_frame(7, "t"))
        c.end_episode("driver disconnected")
        with pytest.raises(EpisodeSuperseded):
            c.begin_plan()

    def test_a_new_frame_revives_the_service(self):
        c = _client()
        c._on_frame(_frame(7, "t"))
        c.end_episode("no frames for 12s")
        c._on_frame(_frame(8, "t"))         # operator starts a new episode
        c.begin_plan()                      # must not raise

    def test_frames_are_live_rejects_a_stale_prompt(self):
        c = _client()
        c._on_frame(_frame(7, "t"))
        assert c.frames_are_live() is True
        c._cursor.last_frame_t = time.time() - 60
        assert c.frames_are_live() is False


class TestOutputLayout:
    """Mirrors the cap-x layout: episode_NNN/attempt_N."""

    def test_episode_dir_naming(self, tmp_path):
        assert _episode_dir(tmp_path, 7).name == "episode_007"
        assert _episode_dir(tmp_path, 0).name == "episode_000"
        assert _episode_dir(tmp_path, None).name == "no_episode_trials"

    def test_attempts_resume_from_disk(self, tmp_path):
        (tmp_path / "episode_007/attempt_1").mkdir(parents=True)
        (tmp_path / "episode_007/attempt_2").mkdir(parents=True)
        assert _existing_attempts(tmp_path) == {7: 2}

    def test_service_seeds_attempts_so_a_restart_does_not_collide(self, tmp_path):
        (tmp_path / "episode_007/attempt_1").mkdir(parents=True)
        svc = TiptopFrankyService(output_dir=str(tmp_path), port=0)
        assert svc.attempts == {7: 1}


class TestServiceLoop:
    def test_waits_for_a_live_frame_not_a_remembered_one(self, tmp_path):
        """A stale prompt must not trigger a re-plan of a finished episode."""
        svc = TiptopFrankyService(output_dir=str(tmp_path), port=0)
        svc.client._on_frame(_frame(7, "t"))
        svc.client._cursor.last_frame_t = time.time() - 60   # gone quiet

        done: list = []

        import threading

        th = threading.Thread(
            target=lambda: done.append(svc._wait_for_work(None)), daemon=True
        )
        th.start()
        th.join(timeout=1.0)
        assert not done, "must keep waiting while the driver is silent"

        svc.client._on_frame(_frame(7, "t"))                 # streaming again
        th.join(timeout=3.0)
        assert done and done[0][0] == 7

    def test_a_new_episode_counts_as_new_work(self, tmp_path):
        svc = TiptopFrankyService(output_dir=str(tmp_path), port=0)
        svc.client._on_frame(_frame(8, "next task"))
        ep, task = svc._wait_for_work(already_handled=7)
        assert (ep, task) == (8, "next task")

    def test_same_episode_already_handled_is_not_rerun(self, tmp_path):
        import threading

        svc = TiptopFrankyService(output_dir=str(tmp_path), port=0)
        svc.client._on_frame(_frame(7, "t"))
        svc.tasks[7] = "t"
        done: list = []
        th = threading.Thread(
            target=lambda: done.append(svc._wait_for_work(7)), daemon=True
        )
        th.start()
        th.join(timeout=1.0)
        assert not done, "an already-handled episode must not be re-planned"


class TestRetryUntilTheDriverEndsTheEpisode:
    """Operator decision: keep re-planning the SAME episode until the driver
    stops it.

    A retry is meaningful because perception is stochastic -- the VLM's box
    placement and M2T2's point subsampling both vary between identical calls, so
    a fresh attempt on an unchanged scene is a genuine new roll of the dice.
    cap-x already retries this way by generating new code blocks.
    """

    def _svc(self, tmp_path, statuses, **kw):
        """Drive serve_forever with scripted per-attempt outcomes."""
        import threading

        # monitor_port=0 -> let the OS pick, so parallel tests never collide
        svc = TiptopFrankyService(
            output_dir=str(tmp_path), port=0, monitor_port=0, **kw
        )
        svc.client._on_frame(_frame(7, "pick up the block"))
        calls: list = []
        seq = list(statuses)

        def keep_streaming():
            # The retry loop requires live frames; emulate the driver.
            import time as _t

            while True:
                svc.client._on_frame(_frame(7, "pick up the block"))
                _t.sleep(0.02)

        threading.Thread(target=keep_streaming, daemon=True).start()

        def fake_episode(episode, task):
            calls.append((episode, task))
            status = seq[min(len(calls) - 1, len(seq) - 1)]
            svc.attempts[episode] = svc.attempts.get(episode, 0) + 1
            if status == "ends_episode":
                svc.client.end_episode("no frames for 12s")
                svc._last_status = "no_plan_found"
                return False
            # Mirror the real _run_episode contract exactly.
            svc._last_status = status
            return status in ("executed", "plan_only")

        svc._run_episode = fake_episode
        svc.client.serve = lambda **k: svc.client
        svc._start_idle_watchdog = lambda: None
        th = threading.Thread(target=svc.serve_forever, daemon=True)
        th.start()
        th.join(timeout=4.0)
        return svc, calls

    def test_no_plan_triggers_another_attempt(self, tmp_path):
        svc, calls = self._svc(
            tmp_path, ["no_plan_found", "no_plan_found", "executed"]
        )
        assert len(calls) >= 3, f"expected retries, got {len(calls)}"
        assert all(c[0] == 7 for c in calls), "retries must stay on episode 7"

    def test_a_successful_plan_stops_retrying(self, tmp_path):
        svc, calls = self._svc(tmp_path, ["executed"])
        assert len(calls) == 1, "an executed plan must not be redone"

    def test_driver_ending_the_episode_stops_retrying(self, tmp_path):
        """The operator's stop is authoritative."""
        svc, calls = self._svc(tmp_path, ["ends_episode", "no_plan_found"])
        assert len(calls) == 1, (
            f"must stop once the driver closed the episode, got {len(calls)}"
        )

    def test_max_attempts_cap_is_honoured(self, tmp_path):
        svc, calls = self._svc(
            tmp_path, ["no_plan_found"] * 10, max_attempts_per_episode=3
        )
        assert len(calls) == 3, f"cap should stop at 3, got {len(calls)}"

    def test_unlimited_is_the_default(self, tmp_path):
        svc = TiptopFrankyService(output_dir=str(tmp_path), port=0)
        assert svc.max_attempts_per_episode == 0, "0 = retry until the driver stops"

    def test_attempt_numbers_increment_across_retries(self, tmp_path):
        svc, calls = self._svc(
            tmp_path, ["no_plan_found", "no_plan_found", "executed"]
        )
        assert svc.attempts[7] >= 3


class TestRetryClassification:
    """Retry whenever NO plan reached the robot.

    A "planning_error" is usually data-dependent, not deterministic: seen live,
    cuTAMP raised ``Shrunk OBB for clear_bin has half extents <= 0`` because the
    stochastic VLM labelled a small object as a placement SURFACE. The next
    attempt may not label it that way, so stopping there was wrong.
    """

    def test_only_a_delivered_plan_stops_retrying(self, tmp_path):
        import inspect

        src = inspect.getsource(TiptopFrankyService._run_episode)
        assert 'return status in ("executed", "plan_only")' in src

    def test_planning_error_is_retried(self, tmp_path):
        svc, calls = TestRetryUntilTheDriverEndsTheEpisode()._svc(
            tmp_path,
            ["planning_error: ValueError: Shrunk OBB", "no_plan_found", "executed"],
        )
        assert len(calls) >= 3, (
            f"a planning_error must be retried (it is data-dependent), got "
            f"{len(calls)}"
        )


class TestExecutionFailuresAreBounded:
    """A plan that reaches the arm and fails there must not loop forever.

    Seen live: franky_service can latch a stop on a libfranka I/O fault. Retrying
    then re-plans and re-commands a faulted arm, which hides the real fault and
    burns VLM calls.
    """

    def test_execution_failure_retries_once_then_stops(self, tmp_path):
        svc, calls = TestRetryUntilTheDriverEndsTheEpisode()._svc(
            tmp_path, ["execution_stopped: RuntimeError: boom"] * 10
        )
        assert len(calls) == 2, (
            f"expected one retry then stop, got {len(calls)} attempts"
        )

    def test_no_plan_still_retries_freely(self, tmp_path):
        """The execution cap must not restrict planning retries."""
        svc, calls = TestRetryUntilTheDriverEndsTheEpisode()._svc(
            tmp_path, ["no_plan_found"] * 10, max_attempts_per_episode=5
        )
        assert len(calls) == 5

    def test_counter_resets_on_a_new_episode(self, tmp_path):
        svc = TiptopFrankyService(output_dir=str(tmp_path), port=0, monitor_port=0)
        svc._exec_failures = 2
        # serve_forever resets it when it picks up new work
        import inspect

        assert "self._exec_failures = 0" in inspect.getsource(
            TiptopFrankyService.serve_forever
        )


class TestDriversThatOmitEpisodeId:
    """A second robot connected sending NO episode_id.

    Its observation carried only prompt / joint_position / gripper_position /
    two 224x224 images -- no episode_id, and no depth / camera_K /
    camera_extrinsic / native RGB either.

    Consequence on the dashboard: with no id there is no boundary, so the
    PREVIOUS driver's episode_seen (202) persisted and the new robot's
    "Put the banana in the bowl" was filed as attempt 5 of an 11-hour-old
    episode, mixing two tasks into one row.
    """

    def _frame(self, prompt, ep=None):
        f = {"prompt": prompt, "observation/joint_position": np.zeros(7)}
        if ep is not None:
            f["episode_id"] = ep
        return f

    def _client(self):
        c = _client()
        c._cursor.session_tag = "185034"
        return c

    def test_a_synthetic_id_is_used(self):
        c = self._client()
        c._on_frame(self._frame("Put the banana in the bowl"))
        ep = c.current_episode()
        assert isinstance(ep, str) and ep.startswith("auto"), ep

    def test_same_task_stays_one_episode(self):
        c = self._client()
        c._on_frame(self._frame("Put the banana in the bowl"))
        first = c.current_episode()
        c._on_frame(self._frame("Put the banana in the bowl"))
        assert c.current_episode() == first

    def test_a_changed_task_starts_a_new_episode(self):
        """The prompt is the only episode signal such a driver gives us."""
        c = self._client()
        c._on_frame(self._frame("Put the banana in the bowl"))
        first = c.current_episode()
        c._on_frame(self._frame("Sort the fruits"))
        assert c.current_episode() != first

    def test_synthetic_ids_cannot_collide_with_numeric_ones(self):
        """A string id can never be confused with another driver's int 202."""
        c = self._client()
        c._on_frame(self._frame("task"))
        assert not isinstance(c.current_episode(), int)

    def test_a_driver_that_DOES_send_the_id_is_unaffected(self):
        c = self._client()
        c._on_frame(self._frame("task", ep=202))
        assert c.current_episode() == 202

    def test_reconnect_forgets_the_previous_episode(self):
        """Otherwise a stale episode_seen suppresses the next boundary."""
        c = self._client()
        c._on_frame(self._frame("task", ep=202))
        # emulate the disconnect cleanup
        with c._cursor.lock:
            c._cursor.episode_seen = None
            c._cursor.prompt = None
            c._cursor.synth_seq = 0
            c._cursor.synth_prompt = None
        c._on_frame(self._frame("a different robot's task"))
        assert c.current_episode() != 202


class TestDashboardSeparatesReusedIds:
    """Drivers restart their episode_id counter, so ids ARE re-used."""

    def test_the_same_id_in_two_sessions_is_two_rows(self):
        from tiptop.monitor import hooks as h
        from tiptop.monitor.events import BUS

        BUS._history.clear()
        BUS.episodes.clear()
        BUS._session = 0

        BUS.new_session("robot A")
        h.episode_start(202, task="Sort the three fruits by size")
        BUS.new_session("robot B")
        h.episode_start(202, task="Put the banana in the bowl")

        assert len(BUS.episodes) == 2, "a re-used id must not append to the old row"
        tasks = {v["task"] for v in BUS.episodes.values()}
        assert tasks == {
            "Sort the three fruits by size",
            "Put the banana in the bowl",
        }

    def test_episode_lookup_defaults_to_the_current_session(self):
        from tiptop.monitor import hooks as h
        from tiptop.monitor.events import BUS

        BUS._history.clear()
        BUS.episodes.clear()
        BUS._session = 0
        BUS.new_session("A")
        h.episode_start(9, task="old")
        BUS.new_session("B")
        h.episode_start(9, task="new")
        assert BUS.episode(9)["task"] == "new"
