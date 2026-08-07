"""Turn a ``franky_service`` observation into the H5 snapshot TiPToP consumes.

``tiptop/tiptop_h5.py::load_h5_observation`` expects exactly these datasets:

===================  =========================================================
``rgb``              (h, w, 3) uint8
``depth``            (h, w) float32, metres
``intrinsic_matrix`` (3, 3) float32, for the DEPTH resolution
``pos_w``            (3,) camera position in world/base frame
``quat_w_ros``       (4,) camera orientation as [w, x, y, z]
``q_init``           (n,) arm joint positions at capture
===================  =========================================================

Every one of those is already on the driver's wire, so no ZED camera and no
FoundationStereo pass are involved — the driver's depth is used as-is, which is
what the other baselines and our own method do.

Two details that are easy to get wrong and would silently corrupt the scene:

* **Resolution pairing.** ``camera_K`` describes the depth image. RGB and depth
  are both 720x1280 on this cell, but if they ever differ the RGB must be
  resized to the depth resolution rather than the intrinsics being rescaled,
  because the point cloud is projected in depth pixels.
* **Quaternion order.** ``scipy`` uses [x, y, z, w]; this H5 format wants
  [w, x, y, z]. ``load_h5_observation`` re-orders it back, so writing the wrong
  order yields a plausible-looking but rotated scene.

``load_h5_observation`` also subtracts a fixed ``[0, 0, -0.015]`` calibration
offset from ``pos_w``. That offset belongs to the pi-sim-evals capture rig, so we
pre-compensate for it here (see ``H5_READER_Z_OFFSET_M``) and the extrinsics that
reach the planner are the driver's own.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Iterable, Sequence

import h5py
import numpy as np
from scipy.spatial.transform import Rotation

_log = logging.getLogger(__name__)

#: Candidate wire keys for the native-resolution RGB, best first. The driver
#: sends a 224x224 policy image under ``exterior_image_1_left``; that one is for
#: the VLA and is useless for 3D, so the ``_raw`` key is required.
DRIVER_RGB_KEYS: tuple[str, ...] = (
    "observation/exterior_image_1_left_raw",
    "observation/front_image_left_raw",
)

#: Candidate wire keys for metric depth, best first.
DRIVER_DEPTH_KEYS: tuple[str, ...] = (
    "observation/depth_external",
    "observation/depth_exterior_image_1_left",
)

_JOINT_KEYS: tuple[str, ...] = (
    "observation/joint_position",
    "observation/joint_positions",
)

_K_KEYS: tuple[str, ...] = ("observation/camera_K", "observation/camera_k")
_EXTRINSIC_KEYS: tuple[str, ...] = (
    "observation/camera_extrinsic",
    "observation/camera_pose",
)

#: ``load_h5_observation`` subtracts this from ``pos_w`` as a rig-specific
#: calibration fudge. We add it back so the planner sees the driver's real
#: extrinsics.
H5_READER_Z_OFFSET_M = np.array([0.0, 0.0, -0.015], dtype=np.float32)


def _first(wire: dict, keys: Iterable[str]) -> tuple[str | None, Any]:
    for k in keys:
        if k in wire and wire[k] is not None:
            return k, wire[k]
    return None, None


def _require(wire: dict, keys: Sequence[str], what: str) -> tuple[str, Any]:
    key, val = _first(wire, keys)
    if key is None:
        raise KeyError(
            f"observation carries no {what}: tried {list(keys)}. "
            f"Available keys: {sorted(wire)[:12]}..."
        )
    return key, val


def wire_to_observation_arrays(wire: dict) -> dict[str, np.ndarray]:
    """Extract and normalise the six arrays ``load_h5_observation`` wants.

    Raises rather than substituting defaults: a wrong camera pose or a
    rectangle-shaped depth map produces 3D targets that are decimetres off while
    still looking plausible in logs.
    """
    _, rgb_raw = _require(wire, DRIVER_RGB_KEYS, "native-resolution RGB")
    depth_key, depth_raw = _require(wire, DRIVER_DEPTH_KEYS, "metric depth")
    _, k_raw = _require(wire, _K_KEYS, "camera intrinsics")
    _, ext_raw = _require(wire, _EXTRINSIC_KEYS, "camera extrinsics")
    _, q_raw = _require(wire, _JOINT_KEYS, "arm joint positions")

    rgb = np.asarray(rgb_raw)
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError(f"RGB must be (h, w, 3); got {rgb.shape}")
    if rgb.dtype != np.uint8:
        rgb = (
            (rgb * 255.0).clip(0, 255).astype(np.uint8)
            if float(np.nanmax(rgb)) <= 1.0
            else rgb.astype(np.uint8)
        )

    depth = np.asarray(depth_raw, dtype=np.float32)
    if depth.ndim == 3 and depth.shape[-1] == 1:
        depth = depth[..., 0]
    if depth.ndim != 2:
        raise ValueError(f"depth must be (h, w); got {depth.shape}")

    # The intrinsics describe the DEPTH image. If RGB differs, resize RGB to
    # match -- never rescale K, because the cloud is projected in depth pixels.
    if rgb.shape[:2] != depth.shape[:2]:
        from PIL import Image

        _log.info(
            "resizing RGB %s -> depth resolution %s so it pairs with camera_K",
            rgb.shape[:2],
            depth.shape[:2],
        )
        rgb = np.asarray(
            Image.fromarray(rgb).resize(
                (depth.shape[1], depth.shape[0]), Image.Resampling.LANCZOS
            )
        )

    k = np.asarray(k_raw, dtype=np.float32).reshape(3, 3)

    ext = np.asarray(ext_raw, dtype=np.float32)
    if ext.size != 16:
        raise ValueError(f"camera_extrinsic must have 16 elements; got {ext.size}")
    world_from_cam = ext.reshape(4, 4)

    pos_w = world_from_cam[:3, 3].astype(np.float32)
    quat_xyzw = Rotation.from_matrix(world_from_cam[:3, :3]).as_quat()
    # H5 format stores [w, x, y, z]; scipy gives [x, y, z, w].
    quat_w_ros = np.array(
        [quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]], dtype=np.float32
    )

    # Undo the reader's rig-specific offset so the planner gets our extrinsics.
    pos_w = pos_w + H5_READER_Z_OFFSET_M

    q_init = np.asarray(q_raw, dtype=np.float32).reshape(-1)
    if q_init.size < 2:
        raise ValueError(f"q_init must be a joint vector; got shape {q_init.shape}")

    finite = np.isfinite(depth)
    _log.info(
        "H5 observation: rgb=%s depth=%s (valid %.1f%%, %.3f-%.3f m) "
        "q_init=%s cam_t=%s",
        rgb.shape,
        depth.shape,
        100.0 * finite.mean(),
        float(depth[finite].min()) if finite.any() else float("nan"),
        float(depth[finite].max()) if finite.any() else float("nan"),
        np.round(q_init, 3).tolist(),
        np.round(pos_w, 3).tolist(),
    )
    if not finite.any():
        raise ValueError(
            f"every depth pixel in '{depth_key}' is invalid — the camera or the "
            "driver's depth stream is broken; refusing to write a useless "
            "snapshot"
        )

    return {
        "rgb": rgb,
        "depth": depth,
        "intrinsic_matrix": k,
        "pos_w": pos_w,
        "quat_w_ros": quat_w_ros,
        "q_init": q_init,
    }


def observation_to_h5(wire: dict, out_path: str | Path) -> Path:
    """Write *wire* to an H5 file that ``tiptop_h5.py`` can load.

    NaN/inf depth is left untouched: ``load_h5_observation`` already maps
    non-finite values to 0.0 and truncates by ``perception.depth_trunc_m``, so
    cleaning it here would hide how much of the frame was actually invalid.
    """
    arrays = wire_to_observation_arrays(wire)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(out_path, "w") as f:
        for name, val in arrays.items():
            f.create_dataset(name, data=val)
    _log.info("wrote TiPToP H5 snapshot -> %s", out_path)
    return out_path
