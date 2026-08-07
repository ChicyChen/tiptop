"""Episode-aware browser dashboard for TiPToP real-robot runs.

Design constraints (same three as the cap-x monitor):

1. **Episode-first.** The driver increments ``episode_id``; a session is many
   episodes, so everything is keyed on it and episodes can be compared.
2. **Observability only.** Must never affect the robot: bounded queues,
   best-effort delivery, ``publish()`` never raises.
3. **Stdlib only** (``http.server`` + SSE), so it cannot conflict with the
   torch/cuRobo pins in this env.

``events.py`` / ``server.py`` / ``ui.py`` are shared with the cap-x monitor by
COPY, not import: the two repos must never depend on each other. ``hooks.py`` is
TiPToP-specific because the pipeline stages differ (VLM -> SAM2 -> M2T2 ->
cuTAMP -> execute).
"""

from tiptop.monitor.events import BUS, Event, EventKind, publish
from tiptop.monitor.server import MonitorServer, start_monitor

__all__ = [
    "BUS",
    "Event",
    "EventKind",
    "publish",
    "MonitorServer",
    "start_monitor",
]
