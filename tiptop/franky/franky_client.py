"""A ``RobotClient`` backend for the ``franky_service``-driven Franka cell.

TiPToP's robot interface is small -- ``execute_plan.py`` and ``tiptop_run.py``
between them call only:

* ``get_joint_positions()``
* ``set_joint_positions(q)``          (visualisation only)
* ``open_gripper(speed=...)`` / ``close_gripper(speed=...)``
* ``execute_joint_impedance_path(joint_confs, joint_vels, durations)``
* ``close()``

``UR5Client`` is the precedent for adding a backend; this is the third one.

Inversion of control
--------------------
``BambooFrankaClient`` and ``UR5Client`` are *clients*: they connect out to a
controller and push commands. ``franky_service`` is the opposite -- it is a
policy CLIENT that connects to US, streams observations, and polls for action
chunks at a fixed rate. So this class hosts a WebSocket SERVER and answers with
chunks; TiPToP's calls become "enqueue and wait for the queue to drain".

The driver must never be blocked inside its request: it has a fixed control
period, so a slow reply stalls the arm. Every handler therefore returns
immediately from a trajectory cursor, and blocking happens only in TiPToP's own
thread while the cursor drains.

Timing
------
``durations`` is cuRobo's ``interpolation_dt`` repeated per waypoint. The driver
consumes chunk rows at its own fixed rate, so a trajectory planned at
``dt = 0.02 s`` would execute at the wrong speed if the rows were forwarded
verbatim. ``_resample`` maps the planned time base onto the driver's, preserving
the intended duration of each segment.
"""

from __future__ import annotations

import asyncio
import logging
import threading
import time
from typing import Any, Optional, Sequence

import numpy as np

_log = logging.getLogger(__name__)

#: Rate at which franky_service consumes action-chunk rows. 15 Hz matches the
#: DROID-style control loop the cap-x adapter measured on this cell.
DRIVER_HZ = float(15.0)

#: Rows per action chunk, as the driver's policy contract expects.
ACTION_HORIZON = 8

#: Wire gripper convention is 0 = open, 1 = closed (the inverse of cap-x's).
WIRE_GRIPPER_OPEN = 0.0
WIRE_GRIPPER_CLOSED = 1.0

#: Ceiling on the gap between the arm's current pose and a plan's first
#: waypoint. cuRobo plans from q_init, so a correct plan starts ~0 rad away;
#: anything larger means the snapshot no longer describes the robot.
MAX_START_GAP_RAD = 0.25

#: Safety ceiling on a single joint step between consecutive commanded rows.
#: cuRobo output is already smooth; this only catches a malformed plan.
MAX_JOINT_STEP_RAD = 0.08


class FrankyExecutionTimeout(RuntimeError):
    """A trajectory did not drain within its expected wall-clock budget."""


class EpisodeSuperseded(BaseException):
    """The operator moved on; the running plan is void.

    Derives from BaseException on purpose: a plan-execution loop wrapped in
    ``except Exception`` would swallow an abort and keep commanding the arm from
    a plan describing a scene that no longer exists.
    """


class _Cursor:
    """Queued joint targets plus the last row actually handed to the driver."""

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.rows: list[np.ndarray] = []
        self.index = 0
        self.last_cmd: Optional[np.ndarray] = None
        self.gripper_wire = WIRE_GRIPPER_OPEN
        self.joints: Optional[np.ndarray] = None
        self.frames = 0
        self.last_frame_t = 0.0
        self.wire: dict = {}
        #: The driver sends `prompt` on EVERY frame and increments `episode_id`
        #: when the operator starts a new episode. The task therefore comes off
        #: the wire, never from a CLI flag.
        self.prompt: Optional[str] = None
        #: State for drivers that omit episode_id (see _synthetic_episode_id).
        self.synth_prompt: Optional[str] = None
        self.synth_seq: int = 0
        self.session_tag: str = ""
        self.episode_seen: Any = None
        #: Bumped on every episode boundary / end. A plan bound to an older
        #: generation must abort rather than keep driving a dead episode.
        self.generation: int = 0
        self.ended_announced: bool = False

    def pending(self) -> int:
        with self.lock:
            return max(0, len(self.rows) - self.index)


class FrankyClient:
    """Serves franky_service's policy protocol; exposes TiPToP's robot API."""

    #: Generation the running plan is bound to; None = unbound, never aborts.
    #: Class-level so the attribute always exists.
    _plan_generation: Optional[int] = None

    def __init__(
        self,
        host: str = "0.0.0.0",
        port: int = 8042,
        action_horizon: int = ACTION_HORIZON,
        driver_hz: float = DRIVER_HZ,
        max_joint_step_rad: float = MAX_JOINT_STEP_RAD,
        max_start_gap_rad: float = MAX_START_GAP_RAD,
    ) -> None:
        self.max_start_gap_rad = max_start_gap_rad
        self.host = host
        self.port = port
        self.action_horizon = action_horizon
        self.driver_hz = driver_hz
        self.max_joint_step_rad = max_joint_step_rad
        self._cursor = _Cursor()
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        self._serving = threading.Event()

    # ── server ─────────────────────────────────────────────────────────
    def serve(self, wait_for_driver_s: float = 600.0) -> "FrankyClient":
        """Start the WebSocket server and block until the driver streams."""
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()
        if not self._serving.wait(timeout=30):
            raise RuntimeError("franky WebSocket server did not start")
        print(
            f"[franky] serving ws://{self.host}:{self.port}/ "
            f"(horizon={self.action_horizon}, {self.driver_hz:g} Hz)",
            flush=True,
        )
        self.wait_for_frames(wait_for_driver_s)
        return self

    def _run_loop(self) -> None:
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        self._loop.run_until_complete(self._serve_forever())

    async def _serve_forever(self) -> None:
        import websockets

        async with websockets.serve(
            self._handler, self.host, self.port, max_size=None, compression=None
        ):
            self._serving.set()
            await asyncio.Future()

    async def _handler(self, ws) -> None:
        from tiptop.franky.codec import packb, unpackb

        await ws.send(packb(self._metadata()))
        print("[franky] driver connected", flush=True)
        # Start a fresh dashboard namespace. Drivers restart their episode_id
        # counter, so ids are re-used across connections: a second robot reused
        # id 202 eleven hours later and its events were filed as "attempt 5" of
        # the first robot's episode 202, mixing two different tasks into one row.
        try:
            from tiptop.monitor.events import BUS

            BUS.new_session(f"{getattr(ws, 'remote_address', ('?',))[0]}")
        except Exception:
            pass
        # Per-connection tag so synthetic ids from different drivers/sessions
        # can never collide.
        with self._cursor.lock:
            self._cursor.session_tag = time.strftime("%H%M%S")
            self._cursor.synth_prompt = None
            self._cursor.synth_seq = 0
        n = 0
        try:
            async for raw in ws:
                wire = unpackb(raw)
                if wire.get("__finalize_only"):
                    continue
                if n == 0:
                    self._report_first_frame(wire)
                actions = self._on_frame(wire)
                await ws.send(packb({"actions": actions}))
                n += 1
        except Exception as exc:
            print(
                f"[franky] session ended after {n} frame(s): "
                f"{type(exc).__name__}: {exc}",
                flush=True,
            )
        finally:
            with self._cursor.lock:
                # A disconnected driver's last frame is stale, and its queued
                # waypoints describe a scene we can no longer verify.
                self._cursor.rows, self._cursor.index = [], 0
                self._cursor.last_cmd = None
                self._cursor.wire = {}
                # Forget the episode id too: the next driver may restart its
                # counter, and a stale value would suppress the boundary that
                # starts a fresh episode.
                self._cursor.episode_seen = None
                self._cursor.prompt = None
            print("[franky] driver disconnected", flush=True)

    @staticmethod
    def _metadata() -> dict:
        return {"policy": "tiptop", "action_horizon": ACTION_HORIZON}

    def _report_first_frame(self, wire: dict) -> None:
        print("[franky] === first observation ===", flush=True)
        for k in sorted(wire):
            v = wire[k]
            desc = (
                f"ndarray {np.asarray(v).shape} {np.asarray(v).dtype}"
                if isinstance(v, np.ndarray)
                else f"{type(v).__name__} {str(v)[:40]!r}"
            )
            print(f"[franky]   {k:<46} {desc}", flush=True)
        print("[franky] === end first observation ===", flush=True)

    # ── the cursor: answer instantly, never block the driver ───────────
    def _on_frame(self, wire: dict) -> np.ndarray:
        q = wire.get("observation/joint_position")
        ep_id = wire.get("episode_id")
        prompt = wire.get("prompt")
        if ep_id is None:
            # Some drivers do not send episode_id at all. Without it there is no
            # boundary to detect, so a previous robot's episode_seen would
            # persist and this driver's events would be filed under THAT episode
            # (observed: a second robot's "Put the banana in the bowl" appeared
            # as attempt 5 of an 11-hour-old episode 202). Synthesise a per-
            # connection id instead, and start a new one whenever the task
            # changes -- the prompt is the only episode signal such a driver
            # gives us.
            ep_id = self._synthetic_episode_id(prompt)
        c = self._cursor
        boundary_from = None
        with c.lock:
            if q is not None:
                c.joints = np.asarray(q, dtype=np.float64).reshape(-1)[:7]
            if isinstance(prompt, str) and prompt.strip():
                c.prompt = prompt
            prev = c.episode_seen
            if ep_id is not None and prev is not None and ep_id != prev:
                # New episode: the queued plan describes a scene that no longer
                # exists. Drop it rather than driving the arm through stale
                # waypoints, and invalidate the running plan.
                dropped = max(0, len(c.rows) - c.index)
                c.rows, c.index = [], 0
                c.last_cmd = None
                c.generation += 1
                boundary_from = (prev, ep_id, dropped)
            if ep_id is not None:
                c.episode_seen = ep_id
            c.frames += 1
            c.last_frame_t = time.time()
            c.ended_announced = False
            c.wire = wire
        if boundary_from is not None:
            prev, new, dropped = boundary_from
            print(
                f"[franky] episode {prev} -> {new}: dropped {dropped} queued "
                "waypoint(s); aborting the stale plan",
                flush=True,
            )
        with c.lock:
            if c.last_cmd is None and c.joints is not None:
                # Hold where the arm is; never command zeros, which would be a
                # violent move to the zero configuration.
                c.last_cmd = np.concatenate([c.joints, [c.gripper_wire]])

            out: list[np.ndarray] = []
            for _ in range(self.action_horizon):
                if c.index < len(c.rows):
                    row = c.rows[c.index]
                    c.index += 1
                    c.last_cmd = row
                else:
                    row = c.last_cmd
                out.append(
                    row
                    if row is not None
                    else np.zeros(8, dtype=np.float64)  # only before any frame
                )
            if c.index >= len(c.rows) and c.rows:
                c.rows, c.index = [], 0
            return np.asarray(out, dtype=np.float64)

    def _enqueue(self, rows: Sequence[np.ndarray]) -> None:
        c = self._cursor
        with c.lock:
            c.rows.extend(np.asarray(r, dtype=np.float64) for r in rows)

    def _drain(self, budget_s: float, label: str) -> None:
        """Wait for the queue to empty, with a wall-clock ceiling."""
        deadline = time.time() + budget_s
        while self._cursor.pending() > 0:
            # A boundary can land mid-motion; this loop is where a plan spends
            # most of its time.
            self._abort_if_superseded()
            if time.time() > deadline:
                remaining = self._cursor.pending()
                raise FrankyExecutionTimeout(
                    f"{label}: {remaining} waypoint(s) still queued after "
                    f"{budget_s:.1f}s — is the driver still streaming?"
                )
            time.sleep(0.01)

    def wait_for_frames(self, timeout_s: float) -> None:
        deadline = time.time() + timeout_s
        announced = False
        while time.time() < deadline:
            if self._cursor.joints is not None:
                return
            if not announced:
                print(
                    "[franky] waiting for the driver to stream observations "
                    f"(connect it to ws://{self.host}:{self.port}/)...",
                    flush=True,
                )
                announced = True
            time.sleep(0.2)
        raise TimeoutError(
            f"no observation within {timeout_s:.0f}s — is franky_service "
            f"pointed at ws://{self.host}:{self.port}/ ?"
        )

    # ── episode state, from the wire ───────────────────────────────────
    def current_episode(self) -> Any:
        with self._cursor.lock:
            return self._cursor.episode_seen

    def current_prompt(self) -> Optional[str]:
        """Task instruction as sent by the driver on every frame."""
        with self._cursor.lock:
            return self._cursor.prompt

    def _synthetic_episode_id(self, prompt: Optional[str]) -> str:
        """Stable id for drivers that omit ``episode_id``.

        Without one there is no boundary to detect, so a PREVIOUS driver's
        episode_seen persists and this driver's events get filed under it -- a
        second robot's "Put the banana in the bowl" showed up as attempt 5 of an
        11-hour-old episode 202, mixing two tasks into one dashboard row.

        Keyed on (connection, task text): a changed prompt means a new episode,
        which is the only episode signal such a driver gives us. Returned as a
        STRING so it can never collide with a numeric id from a driver that does
        send one.
        """
        c = self._cursor
        with c.lock:
            if isinstance(prompt, str) and prompt.strip():
                if prompt != c.synth_prompt:
                    c.synth_prompt = prompt
                    c.synth_seq += 1
            elif c.synth_seq == 0:
                c.synth_seq = 1
            return f"auto{c.session_tag}-{c.synth_seq}"

    def episode_ended(self) -> bool:
        """True once the idle watchdog / disconnect closed the live episode."""
        with self._cursor.lock:
            return self._cursor.ended_announced

    def frames_are_live(self, max_age_s: float = 3.0) -> bool:
        """True if a frame arrived recently.

        ``current_prompt()`` keeps returning the last task forever, so a service
        loop must test liveness too or it will re-plan a finished episode.
        """
        with self._cursor.lock:
            last = self._cursor.last_frame_t
        return bool(last) and (time.time() - last) <= max_age_s

    def begin_plan(self) -> None:
        """Bind subsequent robot calls to the CURRENT episode."""
        with self._cursor.lock:
            if self._cursor.ended_announced:
                raise EpisodeSuperseded(
                    f"episode {self._cursor.episode_seen} has ended; refusing to "
                    "start new work on it"
                )
            self._plan_generation = self._cursor.generation

    def end_episode(self, reason: str) -> None:
        """Invalidate the running plan (idle timeout / disconnect)."""
        with self._cursor.lock:
            if self._cursor.ended_announced or self._cursor.episode_seen is None:
                return
            self._cursor.ended_announced = True
            self._cursor.generation += 1
            self._cursor.rows, self._cursor.index = [], 0
            ep, gen = self._cursor.episode_seen, self._cursor.generation
        print(
            f"[franky] episode {ep} ended ({reason}); invalidating the running "
            f"plan (generation -> {gen})",
            flush=True,
        )

    def _abort_if_superseded(self) -> None:
        with self._cursor.lock:
            gen = self._cursor.generation
            planned = self._plan_generation
        if planned is not None and gen != planned:
            raise EpisodeSuperseded(
                f"episode advanced (plan generation {planned} -> {gen}); "
                "aborting the stale plan"
            )

    # ── TiPToP's RobotClient interface ─────────────────────────────────
    def get_joint_positions(self) -> list[float]:
        c = self._cursor
        with c.lock:
            if c.joints is None:
                raise RuntimeError(
                    "no joint positions yet — the driver has not sent a frame"
                )
            return [float(x) for x in c.joints]

    def set_joint_positions(self, q) -> None:
        """Visualisation hook in TiPToP; not a robot command. Intentionally a
        no-op so it can never be mistaken for one."""
        _log.debug("set_joint_positions ignored on hardware (viz-only API)")

    def open_gripper(self, speed: float = 1.0, force: float = 0.1) -> dict:
        return self._set_gripper(WIRE_GRIPPER_OPEN, "open")

    def close_gripper(self, speed: float = 1.0, force: float = 0.1) -> dict:
        return self._set_gripper(WIRE_GRIPPER_CLOSED, "close")

    def _set_gripper(self, wire_value: float, label: str) -> dict:
        self._abort_if_superseded()
        c = self._cursor
        with c.lock:
            c.gripper_wire = float(wire_value)
            base = (
                c.joints
                if c.joints is not None
                else (c.last_cmd[:7] if c.last_cmd is not None else None)
            )
        if base is None:
            return {"success": False, "error": "no joint state yet"}
        # Hold the pose while the gripper actuates, for long enough that the
        # driver sees the new command and the fingers finish moving.
        hold_rows = max(self.action_horizon, int(round(self.driver_hz * 1.0)))
        self._enqueue([np.concatenate([base, [wire_value]])] * hold_rows)
        try:
            self._drain(15.0, f"{label}_gripper")
        except FrankyExecutionTimeout as exc:
            return {"success": False, "error": str(exc)}
        _log.info("gripper %s", label)
        return {"success": True}

    def execute_joint_impedance_path(
        self,
        joint_confs,
        joint_vels,
        durations,
    ) -> dict:
        """Execute a timed joint trajectory by streaming it to the driver.

        Args:
            joint_confs: (N, 7) joint angles in radians.
            joint_vels: (N, 7) joint velocities (unused: the driver takes
                positions, and the timing is carried by ``durations``).
            durations: per-waypoint segment times in seconds (cuRobo's
                ``interpolation_dt`` repeated).
        """
        self._abort_if_superseded()
        confs = np.asarray(joint_confs, dtype=np.float64)
        if confs.size == 0:
            return {"success": True}
        if confs.ndim != 2 or confs.shape[1] < 7:
            return {
                "success": False,
                "error": f"joint_confs must be (N, >=7); got {confs.shape}",
            }
        confs = confs[:, :7]

        planned_total = float(np.sum(np.asarray(durations, dtype=np.float64)))
        rows = self._resample(confs, durations)

        # Guard the gap from the arm's CURRENT pose to the plan's FIRST row.
        # The plan was computed from a snapshot; if the arm moved since (a retry
        # took ~80s, or the operator jogged it), commanding the first waypoint
        # would be a sudden jump. The between-rows check below cannot see this.
        with self._cursor.lock:
            here = None if self._cursor.joints is None else self._cursor.joints.copy()
        if here is not None:
            gap = float(np.abs(rows[0] - here).max())
            if gap > self.max_start_gap_rad:
                return {
                    "success": False,
                    "error": (
                        f"plan starts {gap:.3f} rad from the arm's current pose "
                        f"(limit {self.max_start_gap_rad}); the scene snapshot is "
                        "stale — refusing to jump. Re-plan from the current state."
                    ),
                }

        jumps = (
            np.abs(np.diff(rows, axis=0)).max(axis=1) if len(rows) > 1 else np.zeros(1)
        )
        if jumps.size and jumps.max() > self.max_joint_step_rad:
            return {
                "success": False,
                "error": (
                    f"resampled trajectory has a {jumps.max():.3f} rad step "
                    f"(limit {self.max_joint_step_rad}); refusing to execute"
                ),
            }

        with self._cursor.lock:
            grip = self._cursor.gripper_wire
        self._enqueue([np.concatenate([r, [grip]]) for r in rows])

        _log.info(
            "executing %d planned waypoints (%.2fs) as %d rows at %g Hz",
            len(confs),
            planned_total,
            len(rows),
            self.driver_hz,
        )
        t_start = time.time()
        try:
            # Generous margin: the driver may pause, and a stall is reported
            # rather than hidden.
            self._drain(max(30.0, planned_total * 4 + 15.0), "trajectory")
        except FrankyExecutionTimeout as exc:
            self._mon_motion(len(rows), planned_total, time.time() - t_start, False)
            return {"success": False, "error": str(exc)}
        self._mon_motion(len(rows), planned_total, time.time() - t_start, True)
        return {"success": True}

    @staticmethod
    def _mon_motion(rows: int, planned_s: float, actual_s: float, ok: bool) -> None:
        try:
            from tiptop.monitor import hooks as mon

            mon.motion_done(
                n_waypoints=rows,
                planned_s=planned_s,
                actual_s=actual_s,
                converged=ok,
            )
        except Exception:
            pass

    def _resample(self, confs: np.ndarray, durations) -> np.ndarray:
        """Map a plan on its own time base onto the driver's fixed rate.

        Forwarding cuRobo rows verbatim would run the arm at
        ``driver_hz * interpolation_dt`` times the intended speed.
        """
        d = np.asarray(durations, dtype=np.float64).reshape(-1)
        if d.size == 0:
            d = np.full(len(confs), 1.0 / self.driver_hz)
        if d.size == 1:
            d = np.full(len(confs), float(d[0]))
        if d.size != len(confs):
            d = np.resize(d, len(confs))

        # Cumulative time of each planned waypoint, starting at 0.
        t_plan = np.concatenate([[0.0], np.cumsum(d[:-1])])
        total = float(t_plan[-1]) if len(t_plan) > 1 else float(d[0])
        if total <= 0:
            return confs

        n_out = max(2, int(round(total * self.driver_hz)))
        t_out = np.linspace(0.0, total, n_out)
        return np.stack(
            [np.interp(t_out, t_plan, confs[:, j]) for j in range(confs.shape[1])],
            axis=1,
        )

    def close(self) -> None:
        if self._loop is not None:
            self._loop.call_soon_threadsafe(self._loop.stop)
        _log.info("franky client closed")

    # ── snapshot for the planner ───────────────────────────────────────
    def latest_wire(self) -> dict:
        """Most recent driver frame, for writing an H5 snapshot.

        The driver connects to exactly one endpoint, so this object has to serve
        both roles: perception source AND RobotClient. Returns a shallow copy so
        a caller cannot mutate the live frame.
        """
        with self._cursor.lock:
            if not self._cursor.wire:
                raise RuntimeError(
                    "no observation yet — the driver has not sent a frame"
                )
            return dict(self._cursor.wire)

    def write_h5(self, out_path) -> "object":
        """Write the latest frame as a TiPToP H5 snapshot."""
        from tiptop.franky.frame_to_h5 import observation_to_h5

        return observation_to_h5(self.latest_wire(), out_path)
