"""``FrankyClient``: TiPToP's RobotClient over the franky_service policy wire.

Inversion of control is the crux. ``BambooFrankaClient``/``UR5Client`` push
commands to a controller; ``franky_service`` instead POLLS us for action chunks
at a fixed rate. So the handler must answer instantly, and TiPToP's blocking
semantics come from waiting for a trajectory cursor to drain.
"""

from __future__ import annotations

import threading
import time

import numpy as np
import pytest

from tiptop.franky.franky_client import (
    ACTION_HORIZON,
    WIRE_GRIPPER_CLOSED,
    WIRE_GRIPPER_OPEN,
    FrankyClient,
)


def _client(hz=15.0):
    c = FrankyClient(port=0, driver_hz=hz)
    return c


def _frame(q=None):
    return {
        "observation/joint_position": np.zeros(7) if q is None else np.asarray(q),
    }


class TestNeverBlocksTheDriver:
    """A slow reply stalls the arm: the driver has a fixed control period."""

    def test_frame_is_answered_before_any_trajectory_exists(self):
        c = _client()
        actions = c._on_frame(_frame())
        assert actions.shape == (ACTION_HORIZON, 8)

    def test_hold_repeats_the_last_command_not_zeros(self):
        """Commanding zeros would drive the arm to the zero configuration."""
        c = _client()
        q = np.array([0.1, -0.6, 0.2, -2.5, 0.0, 1.9, 0.3])
        c._on_frame(_frame(q))
        actions = c._on_frame(_frame(q))
        assert np.allclose(actions[:, :7], q), "must hold the measured pose"
        assert not np.allclose(actions[0, :7], 0.0)

    def test_trajectory_execution_does_not_deadlock(self):
        """execute_joint_impedance_path blocks, so the driver must be pumped by
        another thread -- exactly as it is in reality."""
        c = _client()
        c._on_frame(_frame())
        stop = threading.Event()

        def pump():
            while not stop.is_set():
                c._on_frame(_frame())
                time.sleep(0.002)

        t = threading.Thread(target=pump, daemon=True)
        t.start()
        try:
            confs = np.linspace(0, 0.3, 20)[:, None] * np.ones((1, 7))
            res = c.execute_joint_impedance_path(confs, confs * 0, [0.02] * 20)
        finally:
            stop.set()
        assert res["success"] is True

    def test_a_stalled_driver_is_reported_not_hidden(self):
        """No driver pumping -> the queue never drains -> loud failure."""
        c = _client()
        c._on_frame(_frame())
        # A slow plan so resampling yields small steps (the jump guard is
        # tested separately); nothing pumps the driver, so it must time out.
        confs = np.linspace(0, 0.2, 10)[:, None] * np.ones((1, 7))
        durations = [0.5] * 10
        real_drain = c._drain
        c._drain = lambda budget_s, label: real_drain(0.2, label)
        res = c.execute_joint_impedance_path(confs, confs * 0, durations)
        assert res["success"] is False
        assert "still queued" in res["error"], res["error"]


class TestTimingIsPreserved:
    """durations is cuRobo's interpolation_dt. Forwarding rows verbatim would
    run the arm at driver_hz * dt times the intended speed.
    """

    def test_resample_matches_the_planned_duration(self):
        c = _client(hz=15.0)
        n, dt = 50, 0.02                      # 1.0 s of motion
        confs = np.linspace(0, 1.0, n)[:, None] * np.ones((1, 7))
        rows = c._resample(confs, [dt] * n)
        expected = (n - 1) * dt * 15.0        # 0.98s * 15Hz
        assert abs(len(rows) - expected) <= 1, (
            f"{len(rows)} rows for {(n - 1) * dt:.2f}s at 15Hz; expected "
            f"~{expected:.0f}"
        )

    def test_slow_plan_yields_more_rows_than_waypoints(self):
        """dt=0.1s at 15Hz must be interpolated UP, not truncated."""
        c = _client(hz=15.0)
        confs = np.linspace(0, 1.0, 10)[:, None] * np.ones((1, 7))
        rows = c._resample(confs, [0.1] * 10)
        assert len(rows) > 10

    def test_fast_plan_is_downsampled(self):
        """dt=0.005s at 15Hz means fewer rows than waypoints."""
        c = _client(hz=15.0)
        confs = np.linspace(0, 1.0, 100)[:, None] * np.ones((1, 7))
        rows = c._resample(confs, [0.005] * 100)
        assert len(rows) < 100

    def test_endpoints_are_exact(self):
        """The arm must arrive at the planned final configuration."""
        c = _client()
        confs = np.linspace(0, 0.7, 30)[:, None] * np.ones((1, 7))
        rows = c._resample(confs, [0.02] * 30)
        assert np.allclose(rows[0], confs[0], atol=1e-6)
        assert np.allclose(rows[-1], confs[-1], atol=1e-6)

    def test_monotonic_plan_stays_monotonic(self):
        c = _client()
        confs = np.linspace(0, 0.5, 25)[:, None] * np.ones((1, 7))
        rows = c._resample(confs, [0.02] * 25)
        assert np.all(np.diff(rows[:, 0]) >= -1e-9)


class TestSafety:
    def test_malformed_plan_with_huge_jump_is_refused(self):
        c = _client()
        c._on_frame(_frame())
        confs = np.array([[0.0] * 7, [3.0] * 7])     # 3 rad in one step
        res = c.execute_joint_impedance_path(confs, confs * 0, [1 / 15.0] * 2)
        assert res["success"] is False
        assert "rad step" in res["error"]

    def test_empty_trajectory_succeeds(self):
        c = _client()
        assert c.execute_joint_impedance_path(np.zeros((0, 7)), None, [])["success"]

    def test_wrong_shape_is_reported(self):
        c = _client()
        res = c.execute_joint_impedance_path(np.zeros((5, 3)), None, [0.02] * 5)
        assert res["success"] is False
        assert "must be (N, >=7)" in res["error"]

    def test_set_joint_positions_is_not_a_robot_command(self):
        """It is a Rerun visualisation hook in TiPToP; must never move the arm."""
        c = _client()
        c._on_frame(_frame())
        c.set_joint_positions(np.ones(7))
        assert c._cursor.pending() == 0, "must not enqueue any motion"

    def test_joint_positions_before_any_frame_raises(self):
        with pytest.raises(RuntimeError, match="not sent a frame"):
            _client().get_joint_positions()


class TestGripper:
    def test_open_and_close_use_the_wire_polarity(self):
        """franky_service: 0 = open, 1 = closed."""
        c = _client()
        c._on_frame(_frame())
        stop = threading.Event()

        def pump():
            while not stop.is_set():
                c._on_frame(_frame())
                time.sleep(0.002)

        threading.Thread(target=pump, daemon=True).start()
        try:
            assert c.close_gripper()["success"] is True
            assert c._cursor.gripper_wire == WIRE_GRIPPER_CLOSED
            assert c.open_gripper()["success"] is True
            assert c._cursor.gripper_wire == WIRE_GRIPPER_OPEN
        finally:
            stop.set()

    def test_gripper_state_persists_into_trajectories(self):
        """A closed gripper must stay closed while the arm moves the object."""
        c = _client()
        c._on_frame(_frame())
        c._cursor.gripper_wire = WIRE_GRIPPER_CLOSED
        confs = np.linspace(0, 0.2, 10)[:, None] * np.ones((1, 7))
        c._enqueue(
            [np.concatenate([r, [c._cursor.gripper_wire]]) for r in c._resample(confs, [0.02] * 10)]
        )
        actions = c._on_frame(_frame())
        assert np.allclose(actions[:, 7], WIRE_GRIPPER_CLOSED)


class TestSnapshotAndRobotShareOneEndpoint:
    """The driver connects to exactly one port, so one object must serve both
    perception and control."""

    def test_latest_wire_is_exposed(self):
        c = _client()
        w = _frame([0.1] * 7)
        w["observation/depth_external"] = np.full((4, 4), 0.9, np.float32)
        c._on_frame(w)
        assert "observation/depth_external" in c.latest_wire()

    def test_latest_wire_is_a_copy(self):
        c = _client()
        c._on_frame(_frame())
        c.latest_wire()["injected"] = 1
        assert "injected" not in c.latest_wire()

    def test_latest_wire_before_any_frame_raises(self):
        with pytest.raises(RuntimeError, match="not sent a frame"):
            _client().latest_wire()


class TestAgainstRealSavedPlans:
    """Resampling must preserve the planned duration of REAL cuTAMP output.

    Synthetic trajectories can hide off-by-one and unit errors; these are actual
    plans from the sim eval (dt = 0.02s, 155-274 waypoints per segment).
    """

    @staticmethod
    def _a_real_plan():
        import pathlib

        hits = sorted(
            pathlib.Path("results-memory").rglob("tiptop_plan.json")
        ) + sorted(pathlib.Path("results").rglob("tiptop_plan.json"))
        return hits[0] if hits else None

    def test_duration_is_preserved_on_every_segment(self):
        plan_path = self._a_real_plan()
        if plan_path is None:
            pytest.skip("no saved plan available in this checkout")
        from tiptop.franky.run_franky import _load_plan_for_execution

        c = _client(hz=15.0)
        checked = 0
        for step in _load_plan_for_execution(plan_path):
            if step["type"] != "trajectory":
                continue
            pos = step["plan"].position.cpu().numpy()[:, :7]
            dt = step["dt"]
            rows = c._resample(pos, [dt] * len(pos))
            planned = (len(pos) - 1) * dt
            driver = len(rows) / c.driver_hz
            assert abs(planned - driver) < 0.1, (
                f"{step['label']}: planned {planned:.2f}s but the driver would "
                f"take {driver:.2f}s"
            )
            checked += 1
        assert checked > 0, "expected at least one trajectory segment"

    def test_real_plans_pass_the_jump_guard(self):
        plan_path = self._a_real_plan()
        if plan_path is None:
            pytest.skip("no saved plan available in this checkout")
        from tiptop.franky.run_franky import _load_plan_for_execution
        from tiptop.franky.franky_client import MAX_JOINT_STEP_RAD

        c = _client(hz=15.0)
        for step in _load_plan_for_execution(plan_path):
            if step["type"] != "trajectory":
                continue
            pos = step["plan"].position.cpu().numpy()[:, :7]
            rows = c._resample(pos, [step["dt"]] * len(pos))
            worst = np.abs(np.diff(rows, axis=0)).max()
            assert worst <= MAX_JOINT_STEP_RAD, (
                f"{step['label']}: {worst:.4f} rad step would be refused; the "
                f"guard is too tight for real cuRobo output"
            )

    def test_gripper_steps_are_preserved(self):
        plan_path = self._a_real_plan()
        if plan_path is None:
            pytest.skip("no saved plan available in this checkout")
        from tiptop.franky.run_franky import _load_plan_for_execution

        steps = _load_plan_for_execution(plan_path)
        grips = [s for s in steps if s["type"] == "gripper"]
        assert grips, "a pick-and-place plan must contain gripper actions"
        assert all(s["action"] in {"open", "close"} for s in grips)


class TestDriverConsumptionModel:
    """The driver consumes ONE chunk row per control step and re-requests only
    when the chunk is exhausted (``episode_runner.py:175``, rate_hz=15).

    Getting this wrong is a silent speed error: an early mock that took the last
    row of every chunk drained the queue 8x too fast, executing 4.06s of planned
    motion in 0.63s. The arm still reached the target, so only the WALL CLOCK
    revealed it.
    """

    def test_one_row_per_step_matches_the_resampled_length(self):
        c = _client(hz=15.0)
        c._on_frame(_frame())
        confs = np.linspace(0, 0.3, 60)[:, None] * np.ones((1, 7))
        rows = c._resample(confs, [0.02] * 60)
        c._enqueue([np.concatenate([r, [0.0]]) for r in rows])

        # Emulate the driver: pull a chunk, consume it row by row.
        consumed = 0
        while c._cursor.pending() > 0:
            chunk = c._on_frame(_frame())
            consumed += len(chunk)
        expected_s = (60 - 1) * 0.02
        assert abs(consumed / c.driver_hz - expected_s) < 0.6, (
            f"{consumed} rows = {consumed / c.driver_hz:.2f}s of control time, "
            f"but the plan is {expected_s:.2f}s"
        )

    def test_a_chunk_is_exactly_action_horizon_rows(self):
        c = _client()
        c._on_frame(_frame())
        assert c._on_frame(_frame()).shape == (ACTION_HORIZON, 8)

    def test_rows_are_handed_out_in_order_without_gaps(self):
        """A skipped or repeated row is a discontinuity in commanded motion."""
        c = _client()
        c._on_frame(_frame())
        rows = [np.concatenate([np.full(7, i / 100.0), [0.0]]) for i in range(24)]
        c._enqueue(rows)
        seen = []
        while c._cursor.pending() > 0:
            seen.extend(c._on_frame(_frame()))
        seen = np.asarray(seen)[: len(rows)]
        assert np.allclose(seen[:, 0], [r[0] for r in rows]), "order must be exact"


class TestStalePlanIsRefused:
    """A plan is computed from a SNAPSHOT and executed later.

    If the arm moved in between (a retry took ~80s, or the operator jogged it),
    commanding the plan's first waypoint would be a sudden jump. The
    between-rows jump guard cannot see this gap, so it is checked separately.
    """

    def _env(self, q):
        c = _client()
        c._on_frame(_frame(q))
        return c

    def test_matching_start_is_accepted(self):
        q = np.array([0.0, -0.399, 0.0, -1.901, 0.0, 1.5, 0.0])
        c = self._env(q)
        confs = q + np.linspace(0, 0.2, 30)[:, None] * np.ones((1, 7))
        import threading, time as _t

        stop = threading.Event()
        threading.Thread(
            target=lambda: [
                (c._on_frame(_frame(q)), _t.sleep(0.002))
                for _ in iter(lambda: not stop.is_set(), False)
            ],
            daemon=True,
        ).start()
        try:
            res = c.execute_joint_impedance_path(confs, None, [0.02] * 30)
        finally:
            stop.set()
        assert res["success"] is True

    def test_far_start_is_refused_loudly(self):
        """The real hazard: snapshot went stale."""
        c = self._env(np.zeros(7))
        confs = np.full((30, 7), 1.5)          # 1.5 rad away from the arm
        res = c.execute_joint_impedance_path(confs, None, [0.02] * 30)
        assert res["success"] is False
        assert "stale" in res["error"]
        assert c._cursor.pending() == 0, "must not enqueue any motion"

    def test_the_limit_is_configurable(self):
        c = _client()
        c.max_start_gap_rad = 5.0
        c._on_frame(_frame(np.zeros(7)))
        confs = np.full((5, 7), 1.0)
        # accepted by the gap check, then caught by the per-row jump guard
        res = c.execute_joint_impedance_path(confs, None, [0.02] * 5)
        assert "stale" not in (res.get("error") or "")
