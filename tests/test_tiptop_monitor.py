"""TiPToP dashboard: episode-aware, and unable to affect the robot."""

from __future__ import annotations

import json
import urllib.request

import numpy as np
import pytest

from tiptop.monitor.events import BUS, Event, EventBus, EventKind


@pytest.fixture(autouse=True)
def fresh():
    BUS._history.clear()
    BUS.episodes.clear()
    BUS._current_episode = None
    yield


class TestEpisodeAware:
    def test_episodes_tracked_separately(self):
        from tiptop.monitor import hooks as h

        h.episode_start(7, task="pick up the larger block")
        h.grasps(31, 0.28, 1.2)
        h.episode_end(7)
        h.episode_start(8, task="pick up the smaller block")
        assert {k[1] for k in BUS.episodes} == {7, 8}
        assert BUS.episode(7)["task"] == "pick up the larger block"
        assert BUS.episode(8)["task"] == "pick up the smaller block"
        assert BUS.episode(7)["ended"] is not None
        assert BUS.episode(8)["ended"] is None

    def test_attempts_on_the_same_episode_do_not_create_a_second_row(self):
        from tiptop.monitor import hooks as h

        h.episode_start(7, task="t", attempt=1)
        h.episode_start(7, task="t", attempt=2)
        h.attempt(7, 2)
        assert [k[1] for k in BUS.episodes] == [7]
        assert BUS.episode(7)["task"] == "t"

    def test_episode_zero_is_valid(self):
        from tiptop.monitor import hooks as h

        h.episode_start(0, task="t")
        assert BUS.episode(0) is not None


class TestPipelineStages:
    """TiPToP's stages differ from cap-x's; each must be visible."""

    def test_detections_segmentation_grasps_are_counted_as_tools(self):
        from tiptop.monitor import hooks as h

        h.episode_start(1, task="t")
        h.detections([{"label": "yellow_block"}, {"label": "bowl"}], [], 2.1)
        h.segmentation(3, 0.4)
        h.grasps(31, 0.28, 1.1)
        tools = BUS.episode(1)["tools"]
        assert tools == {"VLM detect": 1, "SAM2": 1, "M2T2": 1}

    def test_successful_planning_is_a_code_event(self):
        from tiptop.monitor import hooks as h

        h.episode_start(1, task="t")
        h.planning(success=True, seconds=12.3, steps=25)
        assert BUS.episode(1)["code_blocks"] == 1
        assert "25 step" in BUS._history[-1].text

    def test_failed_planning_is_an_error_with_the_reason(self):
        from tiptop.monitor import hooks as h

        h.episode_start(1, task="t")
        h.planning(success=False, seconds=60.0, reason="no skeleton found")
        assert BUS.episode(1)["errors"] == 1
        assert "no skeleton found" in BUS._history[-1].text

    def test_snapshot_reports_depth_validity(self):
        from tiptop.monitor import hooks as h

        d = np.full((60, 80), np.nan, np.float32)
        d[10:50, 10:70] = 0.9
        h.episode_start(1, task="t")
        h.snapshot(rgb=np.zeros((60, 80, 3), np.uint8), depth=d, joints=np.zeros(7))
        e = BUS._history[-1]
        assert len(e.images) == 2, "rgb + depth heat-map"
        assert 0 < e.data["depth_valid_frac"] < 1
        assert e.data["depth_min_m"] == 0.9

    def test_snapshot_survives_all_invalid_depth(self):
        from tiptop.monitor import hooks as h

        h.snapshot(depth=np.zeros((20, 20), np.float32))
        assert BUS._history[-1].data["depth_valid_frac"] == 0.0

    def test_motion_reports_planned_vs_actual(self):
        from tiptop.monitor import hooks as h

        h.episode_start(1, task="t")
        h.motion_done(n_waypoints=46, planned_s=3.08, actual_s=3.21, converged=True)
        assert BUS.episode(1)["waypoints"] == 46
        assert "planned 3.08s" in BUS._history[-1].text

    def test_incomplete_motion_is_visible(self):
        from tiptop.monitor import hooks as h

        h.motion_done(n_waypoints=10, planned_s=1.0, actual_s=9.0, converged=False)
        assert "DID NOT COMPLETE" in BUS._history[-1].text


class TestCannotAffectTheRobot:
    def test_publish_survives_a_bad_payload(self):
        class Boom:
            def __repr__(self):
                raise RuntimeError("nope")

        BUS.publish(Event(kind=EventKind.NOTE, data={"x": Boom()}))

    def test_slow_subscriber_cannot_block(self):
        bus = EventBus(queue_size=4)
        q = bus.subscribe()
        for i in range(50):
            bus.publish(Event(kind=EventKind.NOTE, text=str(i)))
        assert q.qsize() <= 4

    def test_hooks_are_safe_with_no_server(self):
        from tiptop.monitor import hooks as h

        h.episode_start(1)
        h.detections([], [])
        h.segmentation(0)
        h.grasps(0)
        h.planning(success=False, seconds=1.0)
        h.error("boom")
        h.note("fine")


class TestHttp:
    def _serve(self):
        import socket

        from tiptop.monitor.server import MonitorServer

        s = socket.socket()
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
        s.close()
        return MonitorServer("127.0.0.1", port).start(), port

    def test_dashboard_is_branded_tiptop(self):
        _, port = self._serve()
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/", timeout=5) as r:
            body = r.read()
        assert b"TiPToP real robot" in body
        assert b"CaP-X" not in body

    def test_snapshot_endpoint(self):
        from tiptop.monitor import hooks as h

        _, port = self._serve()
        h.episode_start(7, task="pick up the block")
        with urllib.request.urlopen(
            f"http://127.0.0.1:{port}/api/snapshot", timeout=5
        ) as r:
            snap = json.loads(r.read())
        assert snap["episodes"][0]["task"] == "pick up the block"


class TestNoCrossRepoDependency:
    """cap-x and tiptop may share code by COPY but must never import each other."""

    @pytest.mark.parametrize(
        "rel",
        [
            "tiptop/monitor/events.py",
            "tiptop/monitor/server.py",
            "tiptop/monitor/ui.py",
            "tiptop/monitor/hooks.py",
            "tiptop/monitor/instrument.py",
        ],
    )
    def test_no_capx_import(self, rel):
        import pathlib

        src = pathlib.Path(rel).read_text()
        assert "capx" not in src, f"{rel} must not reference cap-x"
        assert "vlm_orchestrator" not in src
