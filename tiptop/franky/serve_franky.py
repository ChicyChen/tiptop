"""TiPToP as a persistent service for the ``franky_service`` Franka cell.

This is the TiPToP counterpart of the cap-x real-robot service, and it obeys the
same contract:

* **The task comes from the wire.** ``franky_service`` sends ``prompt`` on EVERY
  frame; the operator sets the task at the robot, never on our command line.
* **Episodes come from the wire.** The driver increments ``episode_id`` when a
  new episode starts. One episode = one snapshot + plan + execution.
Port 8042 by default: 8041 is the cap-x real-robot service in the driver's
``custom_policies.json``, so TiPToP takes its own to avoid any confusion.

* **One process serves many episodes.** ``openpi_client`` connects once and never
  reconnects, so exiting after an episode would permanently break the driver's
  socket. The endpoint therefore outlives every episode.
* **A boundary voids the running work.** Queued waypoints are dropped and the
  in-flight plan aborts, rather than driving the arm through a plan describing a
  scene that no longer exists.
* **No ground-truth success on hardware.** The operator scores each episode;
  nothing here invents a success signal.

Per-episode artifacts mirror the cap-x layout::

    <root>/episode_007/attempt_1/{observation.h5, plan/, execution.json}
                       attempt_2/
                       episode.json
           episodes.jsonl

Run it with ``tiptop/franky/serve_franky.sh`` (which handles the cuTAMP cwd
shadowing), then point the driver at ``ws://<host>:8042/``.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import tyro

_log = logging.getLogger(__name__)

#: How long frame silence means "the operator ended this episode".
IDLE_EPISODE_END_S = 12.0


def apply_llm_backend_patch() -> str:
    """Point TiPToP's detect-and-translate at the configured VLM backend.

    ``perception_wrapper`` imports ``tiptop.perception.gemini.
    detect_and_translate_async`` directly, so switching backends means
    monkeypatching that module -- the same thing the osmo sim runner does via
    ``robolab/scripts/tiptop_{nvidia,gemini}_model_patch.py``. Without this the
    service would call Google GenAI and fail with no API key.

    ``TIPTOP_LLM_BACKEND=gemini`` (plus ``GOOGLE_API_KEY``) selects the
    paper-faithful path; the default ``nvidia`` reuses ``NVIDIA_API_KEY``.
    """
    import os

    backend = os.environ.get("TIPTOP_LLM_BACKEND", "nvidia").lower()
    import tiptop.perception.gemini as g

    if backend == "gemini":
        if not os.environ.get("GOOGLE_API_KEY"):
            raise RuntimeError(
                "TIPTOP_LLM_BACKEND=gemini needs GOOGLE_API_KEY; unset it to "
                "use the NVIDIA inference backend instead"
            )
        model = os.environ.get("TIPTOP_GEMINI_MODEL", "gemini-robotics-er-1.5-preview")
        print(f"[tiptop-service] LLM backend = gemini (model={model})", flush=True)
        return f"gemini:{model}"

    from tiptop.perception import nvidia_vlm

    g.detect_and_translate = nvidia_vlm.detect_and_translate
    g.detect_and_translate_async = nvidia_vlm.detect_and_translate_async
    model = os.environ.get("TIPTOP_NVIDIA_MODEL", "gcp/google/gemini-2.5-flash")
    print(f"[tiptop-service] LLM backend = nvidia (model={model})", flush=True)
    return f"nvidia:{model}"


def _episode_dir(root: Path, episode: Any) -> Path:
    if episode is None:
        # A trial that never saw a frame is not an episode; keep it out of the
        # episode_* namespace. NOTE: episode_id 0 IS valid, so this tests
        # `is None`, not falsiness.
        return root / "no_episode_trials"
    try:
        return root / f"episode_{int(episode):03d}"
    except (TypeError, ValueError):
        safe = "".join(c if c.isalnum() else "_" for c in str(episode))
        return root / f"episode_{safe or 'unknown'}"


def _existing_attempts(root: Path) -> dict[Any, int]:
    """episode -> highest attempt already on disk, so a restart continues."""
    out: dict[Any, int] = {}
    try:
        for ep_dir in root.glob("episode_*"):
            if not ep_dir.is_dir():
                continue
            label = ep_dir.name[len("episode_") :]
            try:
                key: Any = int(label)
            except ValueError:
                key = label
            best = 0
            for att in ep_dir.glob("attempt_*"):
                try:
                    best = max(best, int(att.name.split("_")[1]))
                except (IndexError, ValueError):
                    continue
            if best:
                out[key] = best
    except Exception:
        pass
    return out


def _append(path: Path, row: dict) -> None:
    try:
        with open(path, "a") as fh:
            fh.write(json.dumps(row, default=str) + "\n")
    except Exception:
        pass


class TiptopFrankyService:
    """Waits for episodes on the wire and runs TiPToP for each one."""

    def __init__(
        self,
        output_dir: str,
        host: str = "0.0.0.0",
        port: int = 8042,
        driver_hz: float = 15.0,
        max_planning_time: float = 60.0,
        plan_only: bool = False,
        idle_episode_end_s: float = IDLE_EPISODE_END_S,
        rr_spawn: bool = False,
        monitor_port: int = 8301,
        llm_backend: str = "",
        max_attempts_per_episode: int = 0,
    ) -> None:
        #: 0 = unlimited: retry until the driver ends the episode.
        self.max_attempts_per_episode = max_attempts_per_episode
        self._last_status = ""
        self._exec_failures = 0
        self.monitor_port = monitor_port
        self.llm_backend = llm_backend
        self.root = Path(output_dir)
        self.root.mkdir(parents=True, exist_ok=True)
        self.ledger = self.root / "episodes.jsonl"
        self.plan_only = plan_only
        self.max_planning_time = max_planning_time
        self.idle_episode_end_s = idle_episode_end_s
        self.rr_spawn = rr_spawn

        from tiptop.franky.franky_client import FrankyClient

        self.client = FrankyClient(host=host, port=port, driver_hz=driver_hz)
        self.attempts = _existing_attempts(self.root)
        self.started: dict[Any, float] = {}
        self.tasks: dict[Any, str] = {}

    # ── lifecycle ──────────────────────────────────────────────────────
    def serve_forever(self, first_frame_timeout_s: float = 600.0) -> None:
        try:
            from tiptop.monitor.server import start_monitor

            start_monitor(
                port=self.monitor_port,
                endpoint=f"ws://{self.client.host}:{self.client.port}/",
                model=self.llm_backend,
            )
        except Exception as exc:
            # Monitoring is optional, but a silent failure would leave the
            # operator staring at an unreachable dashboard.
            print(f"[monitor] NOT started: {type(exc).__name__}: {exc}", flush=True)

        try:
            from tiptop.monitor.instrument import instrument_perception

            instrument_perception()
        except Exception:
            pass

        self.client.serve(wait_for_driver_s=first_frame_timeout_s)
        self._start_idle_watchdog()
        print(
            f"[tiptop-service] ready. Artifacts -> {self.root}\n"
            "[tiptop-service] the task comes from the driver's `prompt`; "
            "episodes from `episode_id`.",
            flush=True,
        )
        if self.attempts:
            print(
                "[tiptop-service] resuming: "
                + ", ".join(f"ep{k}={v}" for k, v in sorted(self.attempts.items(), key=str)),
                flush=True,
            )

        handled: Optional[Any] = None
        while True:
            episode, task = self._wait_for_work(handled)
            if episode is None:
                continue
            handled = episode
            self._exec_failures = 0
            # Retry the SAME episode until the operator ends it. Perception is
            # stochastic (the VLM's box placement and M2T2's point sampling both
            # vary run to run), so a fresh attempt on an unchanged scene is a
            # genuine new roll of the dice rather than a repeat of the same
            # computation. This mirrors cap-x, which re-plans within an episode
            # by generating new code blocks.
            while True:
                try:
                    executed = self._run_episode(episode, task)
                except BaseException as exc:  # includes EpisodeSuperseded
                    print(
                        f"[tiptop-service] episode {episode} attempt ended early: "
                        f"{type(exc).__name__}: {exc}",
                        flush=True,
                    )
                    break
                if executed:
                    # A plan ran on the robot; do not immediately redo the task.
                    break
                if self._last_status.startswith("execution_stopped"):
                    # A plan DID reach the arm and failed there. Retry once (the
                    # scene may have changed, e.g. the object was moved), then
                    # stop: repeatedly commanding a faulted arm hides the real
                    # fault and burns VLM calls.
                    self._exec_failures += 1
                    if self._exec_failures >= 2:
                        print(
                            f"[tiptop-service] episode {episode}: execution "
                            "failed twice; stopping retries — check the robot "
                            "(franky_service /health, stop_latched)",
                            flush=True,
                        )
                        break
                if self.client.episode_ended() or not self.client.frames_are_live():
                    print(
                        f"[tiptop-service] episode {episode} closed by the "
                        "driver; stopping retries",
                        flush=True,
                    )
                    break
                if (
                    self.max_attempts_per_episode
                    and self.attempts.get(episode, 0) >= self.max_attempts_per_episode
                ):
                    print(
                        f"[tiptop-service] episode {episode}: reached "
                        f"--max-attempts-per-episode="
                        f"{self.max_attempts_per_episode}; stopping retries",
                        flush=True,
                    )
                    break
                if self.client.current_episode() != episode:
                    break            # operator moved on mid-attempt
                print(
                    f"[tiptop-service] episode {episode}: no plan — retrying "
                    "(perception is stochastic)",
                    flush=True,
                )

    def _wait_for_work(self, already_handled: Any) -> tuple[Any, str]:
        """Block until the driver presents an episode we have not run yet.

        A NEW episode_id, or a changed prompt on the same episode, both count as
        new work. The frame must be recent: the last prompt lingers forever, so
        testing it alone would re-plan a finished episode immediately.
        """
        while True:
            if not self.client.frames_are_live():
                time.sleep(0.25)
                continue
            episode = self.client.current_episode()
            task = self.client.current_prompt()
            if task is None:
                time.sleep(0.25)
                continue
            if episode != already_handled:
                return episode, task
            if self.tasks.get(episode) not in (None, task):
                # Operator retyped the instruction without advancing the id.
                print(
                    f"[tiptop-service] task changed on episode {episode}: "
                    f"{task!r}",
                    flush=True,
                )
                return episode, task
            time.sleep(0.25)

    def _start_idle_watchdog(self) -> None:
        """Frame silence is the only end-of-episode signal the driver gives.

        ``policy.reset()`` on the driver clears local chunk state only and leaves
        the socket open, so a stopped episode is indistinguishable from a slow
        one apart from frames ceasing.
        """

        def _loop() -> None:
            while True:
                time.sleep(1.0)
                try:
                    if not self.client.frames_are_live(self.idle_episode_end_s):
                        self.client.end_episode(
                            f"no frames for {self.idle_episode_end_s:.0f}s"
                        )
                except Exception:
                    pass

        threading.Thread(target=_loop, daemon=True).start()

    # ── one episode ────────────────────────────────────────────────────
    def _run_episode(self, episode: Any, task: str) -> bool:
        """Run one attempt. Returns True if a plan was executed (or plan_only
        produced a plan), i.e. there is nothing to retry."""
        attempt = self.attempts.get(episode, 0) + 1
        self.attempts[episode] = attempt
        self.started.setdefault(episode, time.time())
        self.tasks[episode] = task

        ep_dir = _episode_dir(self.root, episode)
        out = ep_dir / f"attempt_{attempt}"
        out.mkdir(parents=True, exist_ok=True)
        print(
            f"\n[tiptop-service] ===== episode {episode} attempt {attempt} =====\n"
            f"[tiptop-service] task: {task!r}",
            flush=True,
        )
        from tiptop.monitor import hooks as mon

        if attempt == 1:
            mon.episode_start(episode, task=task, attempt=attempt)
        else:
            mon.episode_start(episode, task=task, attempt=attempt)
            mon.attempt(episode, attempt)

        # Bind this attempt to the current episode: any robot call after a
        # boundary aborts instead of racing the next episode.
        self.client.begin_plan()

        # 1. snapshot
        h5 = out / "observation.h5"
        self.client.write_h5(h5)
        q_capture = self.client.get_joint_positions()
        try:
            # Give the stage renderers the image + camera for THIS attempt, so
            # every visualisation is drawn on the frame the planner actually saw.
            import numpy as _np

            from tiptop.franky.frame_to_h5 import wire_to_observation_arrays
            from tiptop.monitor.instrument import set_frame_context

            arrs = wire_to_observation_arrays(self.client.latest_wire())
            from scipy.spatial.transform import Rotation as _R

            q = arrs["quat_w_ros"]
            W = _np.eye(4)
            W[:3, :3] = _R.from_quat([q[1], q[2], q[3], q[0]]).as_matrix()
            W[:3, 3] = arrs["pos_w"]
            set_frame_context(
                rgb=arrs["rgb"],
                K=arrs["intrinsic_matrix"],
                cam_from_world=_np.linalg.inv(W),
            )
        except Exception as exc:
            print(f"[monitor] no frame context: {exc}", flush=True)

        try:
            wire = self.client.latest_wire()
            from tiptop.franky.frame_to_h5 import DRIVER_DEPTH_KEYS, DRIVER_RGB_KEYS

            mon.snapshot(
                rgb=next((wire[k] for k in DRIVER_RGB_KEYS if k in wire), None),
                depth=next((wire[k] for k in DRIVER_DEPTH_KEYS if k in wire), None),
                joints=q_capture,
                h5_path=str(h5),
            )
        except Exception:
            pass

        # 2. plan
        t0 = time.perf_counter()
        status = "planned"
        plan_path = None
        try:
            from tiptop.tiptop_h5 import run_tiptop_h5

            run_tiptop_h5(
                h5_path=str(h5),
                task_instruction=task,
                output_dir=str(out / "plan"),
                max_planning_time=self.max_planning_time,
                rr_spawn=self.rr_spawn,
            )
            plan_path = self._find_plan(out / "plan")
        except Exception as exc:
            status = f"planning_error: {type(exc).__name__}: {exc}"
            print(f"[tiptop-service] planning FAILED: {exc}", flush=True)
        plan_s = time.perf_counter() - t0
        try:
            n_steps = None
            if plan_path is not None:
                n_steps = len(json.loads(plan_path.read_text()).get("steps", []))
            mon.planning(
                success=plan_path is not None,
                seconds=plan_s,
                steps=n_steps,
                reason=None if plan_path is not None else status,
            )
        except Exception:
            pass

        # 3. execute
        if plan_path is None and status == "planned":
            status = "no_plan_found"
            print("[tiptop-service] no plan produced — nothing to execute", flush=True)
        elif plan_path is not None and self.plan_only:
            status = "plan_only"
            print("[tiptop-service] --plan-only: not executing", flush=True)
        elif plan_path is not None:
            status = self._execute(plan_path)

        self._write_manifests(
            ep_dir=ep_dir,
            out=out,
            episode=episode,
            attempt=attempt,
            task=task,
            status=status,
            plan_path=plan_path,
            plan_s=plan_s,
            q_capture=q_capture,
        )
        print(
            f"[tiptop-service] episode {episode} attempt {attempt}: {status} "
            f"(planning {plan_s:.1f}s)",
            flush=True,
        )
        try:
            mon.note(f"attempt {attempt} finished: {status}", status=status)
        except Exception:
            pass
        # Retry whenever no plan reached the robot. Perception is stochastic —
        # the VLM's object/surface labelling and M2T2's point sampling both vary
        # per attempt — so a "planning_error" is usually data-dependent, not
        # deterministic. Example seen live: cuTAMP raised
        #   ValueError: Shrunk OBB for clear_bin has half extents <= 0
        # because the VLM labelled a small object as a placement SURFACE; the
        # next attempt may not label it that way at all.
        #
        # Only stop when a plan actually ran (or --plan-only produced one). The
        # operator's episode-end and --max-attempts-per-episode remain the real
        # bounds on retrying.
        self._last_status = status
        return status in ("executed", "plan_only")

    def _execute(self, plan_path: Path) -> str:
        from tiptop.execute_plan import execute_cutamp_plan
        from tiptop.franky.run_franky import _load_plan_for_execution

        print("[tiptop-service] executing on the real robot...", flush=True)
        try:
            plan = _load_plan_for_execution(plan_path)
            execute_cutamp_plan(plan, client=self.client)
            print("[tiptop-service] execution complete", flush=True)
            return "executed"
        except BaseException as exc:
            # EpisodeSuperseded lands here too: the operator moved on.
            print(
                f"[tiptop-service] execution stopped: {type(exc).__name__}: {exc}",
                flush=True,
            )
            return f"execution_stopped: {type(exc).__name__}: {exc}"

    def _write_manifests(self, **kw: Any) -> None:
        ep_dir: Path = kw["ep_dir"]
        out: Path = kw["out"]
        try:
            (out / "attempt.json").write_text(
                json.dumps(
                    {
                        "episode_id": kw["episode"],
                        "attempt": kw["attempt"],
                        "task": kw["task"],
                        "status": kw["status"],
                        "plan": str(kw["plan_path"]) if kw["plan_path"] else None,
                        "planning_seconds": round(kw["plan_s"], 2),
                        "q_at_capture": kw["q_capture"],
                        "note": (
                            "Success on hardware is scored MANUALLY by the "
                            "operator. 'executed' only means the trajectory ran "
                            "without raising."
                        ),
                    },
                    indent=2,
                    default=str,
                )
            )
            (ep_dir / "episode.json").write_text(
                json.dumps(
                    {
                        "episode_id": kw["episode"],
                        "task": kw["task"],
                        "attempts": kw["attempt"],
                        "started_iso": datetime.fromtimestamp(
                            self.started[kw["episode"]]
                        ).strftime("%Y-%m-%d %H:%M:%S"),
                        "updated_iso": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    },
                    indent=2,
                    default=str,
                )
            )
        except Exception:
            pass
        _append(
            self.ledger,
            {
                "iso": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "episode_id": kw["episode"],
                "attempt": kw["attempt"],
                "task": kw["task"],
                "status": kw["status"],
                "planning_seconds": round(kw["plan_s"], 2),
                "output_dir": str(out),
            },
        )

    @staticmethod
    def _find_plan(plan_root: Path) -> Path | None:
        hits = sorted(
            plan_root.rglob("tiptop_plan.json"), key=lambda p: p.stat().st_mtime
        )
        return hits[-1] if hits else None


def main(
    output_dir: str = "outputs/tiptop_franky",
    host: str = "0.0.0.0",
    port: int = 8042,
    driver_hz: float = 15.0,
    max_planning_time: float = 60.0,
    plan_only: bool = False,
    idle_episode_end_s: float = IDLE_EPISODE_END_S,
    first_frame_timeout_s: float = 600.0,
    rr_spawn: bool = False,
    monitor_port: int = 8301,
    max_attempts_per_episode: int = 0,
) -> None:
    """Serve TiPToP to the franky_service driver, one episode at a time.

    Args:
        output_dir: Root for per-episode artifacts.
        host: Bind address for the policy WebSocket.
        port: Port the driver connects to.
        driver_hz: Rate at which the driver consumes action-chunk rows.
        max_planning_time: cuTAMP budget per episode, seconds.
        plan_only: Plan but never move the robot (safe first run).
        idle_episode_end_s: Frame silence that counts as end-of-episode.
        monitor_port: Browser dashboard port.
        max_attempts_per_episode: Safety cap on retries (0 = unlimited, retry
            until the driver ends the episode).
        first_frame_timeout_s: How long to wait for the driver at startup.
        rr_spawn: Spawn a local Rerun viewer.
    """
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    backend = apply_llm_backend_patch()
    TiptopFrankyService(
        output_dir=output_dir,
        host=host,
        port=port,
        driver_hz=driver_hz,
        max_planning_time=max_planning_time,
        plan_only=plan_only,
        idle_episode_end_s=idle_episode_end_s,
        rr_spawn=rr_spawn,
        monitor_port=monitor_port,
        llm_backend=backend,
        max_attempts_per_episode=max_attempts_per_episode,
    ).serve_forever(first_frame_timeout_s=first_frame_timeout_s)


def entrypoint() -> None:
    try:
        tyro.cli(main)
    except KeyboardInterrupt:
        print("\n[tiptop-service] stopped", flush=True)
    finally:
        import os

        os._exit(0)


if __name__ == "__main__":
    entrypoint()
