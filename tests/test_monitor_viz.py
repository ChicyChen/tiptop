"""Per-stage diagnostic renderers.

These exist because every failure so far was only diagnosable visually: a VLM box
that misses its object, a SAM2 mask on a shadow, M2T2 grasps on a table lip, and
an association step that silently discards every grasp.
"""

from __future__ import annotations

import base64

import numpy as np
import pytest

from tiptop.monitor import viz


def _rgb(h=120, w=160):
    return np.full((h, w, 3), 90, np.uint8)


def _is_png_b64(s):
    return isinstance(s, str) and base64.b64decode(s)[:4] == b"\x89PNG"


class TestVlmBoxes:
    def test_renders_boxes(self):
        out = viz.vlm_boxes(
            _rgb(), [{"box_2d": [400, 400, 600, 600], "label": "blue_cube"}]
        )
        assert _is_png_b64(out)

    def test_empty_list_still_renders(self):
        assert _is_png_b64(viz.vlm_boxes(_rgb(), []))

    def test_malformed_box_is_skipped_not_fatal(self):
        out = viz.vlm_boxes(
            _rgb(), [{"box_2d": [1, 2], "label": "bad"}, {"label": "nobox"}]
        )
        assert _is_png_b64(out)


class TestSam2Masks:
    def test_accepts_n_1_h_w(self):
        m = np.zeros((2, 1, 120, 160), bool)
        m[0, 0, 10:40, 10:40] = True
        assert _is_png_b64(viz.sam2_masks(_rgb(), m, ["a", "b"]))

    def test_accepts_a_list_of_masks(self):
        m = [np.zeros((120, 160), bool), np.ones((120, 160), bool)]
        assert _is_png_b64(viz.sam2_masks(_rgb(), m))

    def test_mismatched_shape_is_ignored(self):
        assert viz.sam2_masks(_rgb(), np.zeros((1, 5, 5), bool)) is not None

    def test_none_returns_none(self):
        assert viz.sam2_masks(_rgb(), None) is None


class TestM2t2Grasps:
    def _cam(self):
        K = np.array([[100.0, 0, 80], [0, 100.0, 60], [0, 0, 1]])
        return K, np.eye(4)

    def test_projects_contacts(self):
        K, cfw = self._cam()
        c = np.array([[0.0, 0.0, 1.0], [0.1, 0.0, 1.0]])
        assert _is_png_b64(viz.m2t2_grasps(_rgb(), c, K, cfw))

    def test_kept_mask_colours_green_and_red(self):
        K, cfw = self._cam()
        c = np.array([[0.0, 0.0, 1.0], [0.1, 0.0, 1.0]])
        out = viz.m2t2_grasps(_rgb(), c, K, cfw, np.array([True, False]))
        assert _is_png_b64(out)

    def test_no_contacts_returns_none(self):
        K, cfw = self._cam()
        assert viz.m2t2_grasps(_rgb(), np.zeros((0, 3)), K, cfw) is None

    def test_points_behind_the_camera_are_dropped(self):
        K, cfw = self._cam()
        assert viz.m2t2_grasps(_rgb(), np.array([[0.0, 0.0, -1.0]]), K, cfw) is None


class TestFinalGrasp:
    def test_draws_axes(self):
        K = np.array([[100.0, 0, 80], [0, 100.0, 60], [0, 0, 1]])
        T = np.eye(4)
        T[:3, 3] = [0, 0, 1.0]
        assert _is_png_b64(viz.final_grasp(_rgb(), T, K, np.eye(4), "cube", 0.64))

    def test_behind_camera_returns_none(self):
        K = np.array([[100.0, 0, 80], [0, 100.0, 60], [0, 0, 1]])
        T = np.eye(4)
        T[:3, 3] = [0, 0, -1.0]
        assert viz.final_grasp(_rgb(), T, K, np.eye(4)) is None


class TestNeverRaises:
    """A rendering bug must not affect the robot or hide a run."""

    @pytest.mark.parametrize(
        "call",
        [
            lambda: viz.vlm_boxes(None, [{"box_2d": [1, 2, 3, 4]}]),
            lambda: viz.sam2_masks(None, "garbage"),
            lambda: viz.m2t2_grasps(None, "garbage", None, None),
            lambda: viz.final_grasp(None, "garbage", None, None),
        ],
    )
    def test_bad_input_returns_none(self, call):
        assert call() is None


class TestFailureIsLoud:
    """The two silent killers must become unmistakable errors."""

    def test_no_goal_predicate_is_an_error(self):
        from tiptop.monitor.events import BUS, EventKind
        from tiptop.monitor import hooks as h

        BUS._history.clear()
        h.no_goal([{"label": "orange_bottle"}])
        e = BUS._history[-1]
        assert e.kind == EventKind.ERROR
        assert "NO GOAL PREDICATE" in e.text
        assert "on(object, surface)" in e.text

    def test_zero_association_is_an_error(self):
        from tiptop.monitor.events import BUS, EventKind
        from tiptop.monitor import hooks as h

        BUS._history.clear()
        h.association({"blue_cube": 0, "bin": 0}, 0)
        e = BUS._history[-1]
        assert e.kind == EventKind.ERROR
        assert "NO GRASPS ASSOCIATED" in e.text

    def test_nonzero_association_is_a_tool_event(self):
        from tiptop.monitor.events import BUS, EventKind
        from tiptop.monitor import hooks as h

        BUS._history.clear()
        h.association({"blue_cube": 215}, 215)
        assert BUS._history[-1].kind == EventKind.TOOL
