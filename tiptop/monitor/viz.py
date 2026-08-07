"""Render per-stage diagnostic images for the TiPToP dashboard.

Every stage of the pipeline gets a picture, because the failures we have seen are
only diagnosable visually:

``vlm_boxes``   the VLM's ``box_2d`` rectangles over the RGB. A box that misses
                its object is the single most common root cause -- SAM2 can only
                segment inside the box it is given.
``sam2_masks``  the masks SAM2 returned, tinted per object, so a mask sitting on
                a shadow or on the robot arm is obvious.
``m2t2_grasps`` M2T2's grasp contacts projected back into the image. M2T2 is
                object-agnostic: it proposes grasps on any grippable geometry
                (table lips included), so seeing WHERE they land explains why an
                object may end up with none.
``association`` the join that actually decides success: grasps within
                ``contact_threshold_m`` of each object's masked points are kept,
                everything else is discarded. Green = kept, red = discarded.

All functions return base64 PNG strings and never raise: a rendering bug must
not affect the robot or hide a run.
"""

from __future__ import annotations

import base64
import io
import logging
from typing import Any, Sequence

import numpy as np

logger = logging.getLogger(__name__)

_PALETTE = [
    (255, 64, 64),
    (64, 255, 64),
    (64, 160, 255),
    (255, 220, 0),
    (255, 64, 255),
    (0, 255, 220),
    (255, 150, 60),
    (170, 120, 255),
]


def _b64(img) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def _as_pil(rgb: np.ndarray):
    from PIL import Image

    a = np.asarray(rgb)
    if a.dtype != np.uint8:
        a = (a * 255).clip(0, 255).astype(np.uint8) if a.max() <= 1.0 else a.astype(np.uint8)
    return Image.fromarray(a[:, :, :3])


def _fit(img, max_w: int = 900):
    if img.width > max_w:
        img = img.resize((max_w, int(img.height * max_w / img.width)))
    return img


def vlm_boxes(rgb: np.ndarray, bboxes: Sequence[dict]) -> str | None:
    """Draw the VLM's boxes. ``box_2d`` is [ymin,xmin,ymax,xmax] over 0..1000."""
    try:
        from PIL import ImageDraw

        img = _as_pil(rgb)
        W, H = img.size
        d = ImageDraw.Draw(img)
        for i, b in enumerate(bboxes or []):
            box = b.get("box_2d")
            if not box or len(box) != 4:
                continue
            ymin, xmin, ymax, xmax = box
            xy = [
                xmin / 1000 * W,
                ymin / 1000 * H,
                xmax / 1000 * W,
                ymax / 1000 * H,
            ]
            col = _PALETTE[i % len(_PALETTE)]
            d.rectangle(xy, outline=col, width=3)
            d.text((xy[0] + 4, max(0, xy[1] - 14)), str(b.get("label", "?")), fill=col)
        d.text((6, 6), f"VLM boxes: {len(bboxes or [])}", fill=(255, 255, 255))
        return _b64(_fit(img))
    except Exception:
        logger.debug("vlm_boxes render failed", exc_info=True)
        return None


def sam2_masks(rgb: np.ndarray, masks: Any, labels: Sequence[str] | None = None) -> str | None:
    """Tint each SAM2 mask. Accepts (N,1,H,W), (N,H,W) or a list of (H,W)."""
    try:
        from PIL import ImageDraw

        arr = _normalise_masks(masks)
        if arr is None:
            return None
        base = np.asarray(_as_pil(rgb)).copy()
        for i, mk in enumerate(arr):
            mk = mk.astype(bool)
            if mk.shape != base.shape[:2] or not mk.any():
                continue
            col = np.array(_PALETTE[i % len(_PALETTE)])
            base[mk] = (0.55 * col + 0.45 * base[mk]).astype(np.uint8)
        from PIL import Image

        img = Image.fromarray(base)
        d = ImageDraw.Draw(img)
        for i, mk in enumerate(arr):
            mk = mk.astype(bool)
            if not mk.any() or mk.shape != base.shape[:2]:
                continue
            ys, xs = np.nonzero(mk)
            name = labels[i] if labels and i < len(labels) else f"mask{i}"
            d.text(
                (xs.mean(), ys.mean()),
                f"{name} ({int(mk.sum())}px)",
                fill=_PALETTE[i % len(_PALETTE)],
            )
        d.text((6, 6), f"SAM2 masks: {len(arr)}", fill=(255, 255, 255))
        return _b64(_fit(img))
    except Exception:
        logger.debug("sam2_masks render failed", exc_info=True)
        return None


def m2t2_grasps(
    rgb: np.ndarray,
    contacts: np.ndarray,
    K: np.ndarray,
    cam_from_world: np.ndarray,
    kept_mask: np.ndarray | None = None,
) -> str | None:
    """Project grasp contacts into the image.

    ``kept_mask`` marks grasps that survived object association: green = kept,
    red = discarded. Without it every contact is drawn yellow.
    """
    try:
        from PIL import ImageDraw

        c = np.asarray(contacts, dtype=np.float64).reshape(-1, 3)
        if len(c) == 0:
            return None
        hom = np.c_[c, np.ones(len(c))]
        cam = (np.asarray(cam_from_world) @ hom.T).T[:, :3]
        front = cam[:, 2] > 1e-6
        if not front.any():
            return None
        uv = (np.asarray(K) @ cam[front].T).T
        uv = uv[:, :2] / uv[:, 2:3]
        keep = (
            np.asarray(kept_mask, dtype=bool)[front]
            if kept_mask is not None
            else None
        )
        img = _as_pil(rgb)
        d = ImageDraw.Draw(img)
        order = np.argsort(keep.astype(int)) if keep is not None else range(len(uv))
        for i in order:
            u, v = uv[i]
            if keep is None:
                col = (255, 220, 0)
            else:
                col = (0, 255, 0) if keep[i] else (255, 60, 60)
            d.ellipse([u - 2.5, v - 2.5, u + 2.5, v + 2.5], fill=col)
        n_keep = int(keep.sum()) if keep is not None else None
        d.text((6, 6), f"M2T2 grasp contacts: {len(uv)}", fill=(255, 255, 255))
        if n_keep is not None:
            d.text(
                (6, 24),
                f"GREEN kept after association: {n_keep}   RED discarded: {len(uv) - n_keep}",
                fill=(0, 255, 0) if n_keep else (255, 60, 60),
            )
        return _b64(_fit(img))
    except Exception:
        logger.debug("m2t2_grasps render failed", exc_info=True)
        return None


def final_grasp(
    rgb: np.ndarray,
    pose: np.ndarray,
    K: np.ndarray,
    cam_from_world: np.ndarray,
    label: str = "",
    confidence: float | None = None,
) -> str | None:
    """Draw the chosen grasp: its axes projected into the image."""
    try:
        from PIL import ImageDraw

        T = np.asarray(pose, dtype=np.float64).reshape(4, 4)
        origin = T[:3, 3]
        axes = [(T[:3, 0], (255, 60, 60)), (T[:3, 1], (60, 255, 60)), (T[:3, 2], (80, 160, 255))]
        cfw = np.asarray(cam_from_world)

        def proj(p):
            q = cfw @ np.r_[p, 1.0]
            if q[2] <= 1e-6:
                return None
            uv = np.asarray(K) @ q[:3]
            return uv[:2] / uv[2]

        o = proj(origin)
        if o is None:
            return None
        img = _as_pil(rgb)
        d = ImageDraw.Draw(img)
        for vec, col in axes:
            tip = proj(origin + 0.05 * vec)
            if tip is not None:
                d.line([o[0], o[1], tip[0], tip[1]], fill=col, width=3)
        d.ellipse([o[0] - 6, o[1] - 6, o[0] + 6, o[1] + 6], outline=(255, 255, 255), width=3)
        txt = f"CHOSEN GRASP {label}"
        if confidence is not None:
            txt += f"  conf={confidence:.3f}"
        d.text((6, 6), txt, fill=(255, 255, 255))
        d.text((6, 24), "blue = approach axis", fill=(80, 160, 255))
        return _b64(_fit(img))
    except Exception:
        logger.debug("final_grasp render failed", exc_info=True)
        return None


def _normalise_masks(masks: Any) -> np.ndarray | None:
    try:
        if masks is None:
            return None
        if isinstance(masks, (list, tuple)):
            arr = np.asarray([np.asarray(m).squeeze() for m in masks])
        else:
            arr = np.asarray(masks)
        if arr.ndim == 4:
            arr = arr[:, 0]
        if arr.ndim == 2:
            arr = arr[None]
        return arr if arr.ndim == 3 else None
    except Exception:
        return None
