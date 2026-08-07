"""Driver observation -> TiPToP H5 snapshot.

The point of this path is that TiPToP's existing ``tiptop_h5.py`` planner runs
unmodified on our cell: no ZED camera, and no FoundationStereo pass (the driver
already provides metric depth, matching the other baselines and our method).
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from tiptop.franky.frame_to_h5 import (
    observation_to_h5,
    wire_to_observation_arrays,
)


def _wire(**over):
    R = Rotation.from_euler("xyz", [2.6, 0.02, -1.57]).as_matrix()
    ext = np.eye(4)
    ext[:3, :3] = R
    ext[:3, 3] = [0.62, 0.04, 0.58]
    wire = {
        "observation/exterior_image_1_left_raw": np.zeros((720, 1280, 3), np.uint8),
        "observation/depth_external": np.full((720, 1280), 0.9, np.float32),
        "observation/camera_K": np.array([526.8, 0, 640, 0, 526.8, 360, 0, 0, 1.0]),
        "observation/camera_extrinsic": ext.reshape(-1),
        "observation/joint_position": np.array([0.0, -0.6, 0.0, -2.5, 0.0, 1.9, 0.0]),
    }
    wire.update(over)
    return wire


class TestRoundTripThroughTiptopsOwnReader:
    """The only test that really matters: what the planner ends up seeing."""

    def test_extrinsics_survive_exactly(self, tmp_path):
        from tiptop.tiptop_h5 import load_h5_observation

        R = Rotation.from_euler("xyz", [2.6, 0.02, -1.57]).as_matrix()
        ext = np.eye(4)
        ext[:3, :3] = R
        ext[:3, 3] = [0.62, 0.04, 0.58]
        p = observation_to_h5(
            _wire(**{"observation/camera_extrinsic": ext.reshape(-1)}),
            tmp_path / "obs.h5",
        )
        obs = load_h5_observation(p)
        assert np.allclose(obs.world_from_cam[:3, 3], ext[:3, 3], atol=1e-5), (
            "translation must round-trip: load_h5_observation subtracts a "
            "rig-specific offset that we pre-compensate for"
        )
        assert np.allclose(obs.world_from_cam[:3, :3], R, atol=1e-4), (
            "rotation must round-trip: the H5 format stores [w,x,y,z] while "
            "scipy uses [x,y,z,w]"
        )

    def test_joint_state_and_intrinsics_survive(self, tmp_path):
        from tiptop.tiptop_h5 import load_h5_observation

        obs = load_h5_observation(observation_to_h5(_wire(), tmp_path / "o.h5"))
        assert np.allclose(obs.q_init, [0.0, -0.6, 0.0, -2.5, 0.0, 1.9, 0.0])
        assert obs.frame.intrinsics[0, 0] == pytest.approx(526.8)
        assert obs.frame.intrinsics[0, 2] == pytest.approx(640.0)


class TestNoSilentCorruption:
    """Rather than substitute defaults, refuse: a wrong pose or a rectangle
    depth map yields 3D targets that are decimetres off but look plausible.
    """

    def test_missing_native_rgb_raises(self, tmp_path):
        w = _wire()
        del w["observation/exterior_image_1_left_raw"]
        # The 224x224 policy image must NOT be silently accepted as a stand-in.
        w["observation/exterior_image_1_left"] = np.zeros((224, 224, 3), np.uint8)
        with pytest.raises(KeyError, match="native-resolution RGB"):
            wire_to_observation_arrays(w)

    def test_missing_depth_raises(self):
        w = _wire()
        del w["observation/depth_external"]
        with pytest.raises(KeyError, match="metric depth"):
            wire_to_observation_arrays(w)

    def test_missing_extrinsics_raises(self):
        w = _wire()
        del w["observation/camera_extrinsic"]
        with pytest.raises(KeyError, match="camera extrinsics"):
            wire_to_observation_arrays(w)

    def test_all_nan_depth_raises(self):
        w = _wire(
            **{"observation/depth_external": np.full((720, 1280), np.nan, np.float32)}
        )
        with pytest.raises(ValueError, match="every depth pixel"):
            wire_to_observation_arrays(w)

    def test_partial_nan_depth_is_kept(self):
        """NaN holes are real sensor behaviour; the reader maps them to 0."""
        d = np.full((720, 1280), 0.9, np.float32)
        d[:100, :100] = np.nan
        arrays = wire_to_observation_arrays(
            _wire(**{"observation/depth_external": d})
        )
        assert np.isnan(arrays["depth"][:100, :100]).all(), (
            "do not clean depth here — the reader handles it and hides nothing"
        )


class TestResolutionPairing:
    """camera_K describes the DEPTH image; the cloud is projected in depth px."""

    def test_rgb_is_resized_to_depth_resolution(self):
        arrays = wire_to_observation_arrays(
            _wire(
                **{
                    "observation/exterior_image_1_left_raw": np.zeros(
                        (480, 864, 3), np.uint8
                    ),
                    "observation/depth_external": np.full(
                        (720, 1280), 0.9, np.float32
                    ),
                }
            )
        )
        assert arrays["rgb"].shape[:2] == (720, 1280)
        assert arrays["intrinsic_matrix"][0, 0] == pytest.approx(526.8), (
            "K must NOT be rescaled; RGB moves to match it"
        )

    def test_float_rgb_is_converted(self):
        arrays = wire_to_observation_arrays(
            _wire(
                **{
                    "observation/exterior_image_1_left_raw": np.full(
                        (720, 1280, 3), 0.5, np.float32
                    )
                }
            )
        )
        assert arrays["rgb"].dtype == np.uint8
        assert arrays["rgb"].max() > 1
