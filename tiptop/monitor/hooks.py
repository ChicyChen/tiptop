"""Publishers for the TiPToP real-robot monitor.

The event bus, HTTP/SSE server and dashboard are shared with the cap-x monitor
(pure stdlib, no cross-repo import — the files are copied, so neither repo
depends on the other). Only these publishers are TiPToP-specific, because
TiPToP's pipeline stages differ from cap-x's:

    snapshot -> VLM detect -> SAM2 segment -> M2T2 grasps -> cuTAMP -> execute

Everything here is best-effort: observability must never affect the robot.
"""

from __future__ import annotations

import logging
from typing import Any

from .events import BUS, EventKind, publish

logger = logging.getLogger(__name__)


# ── episode lifecycle ──────────────────────────────────────────────────

def episode_start(episode: Any, task: str | None = None, attempt: int = 1) -> None:
    publish(
        EventKind.EPISODE_START,
        f"episode {episode} started (attempt {attempt})",
        episode=episode,
        data={"attempt": attempt},
    )
    if task:
        from .events import BUS as _B

        _B.clear_superseded(episode)
        publish(EventKind.TASK, task, episode=episode)


def episode_end(episode: Any, reason: str = "") -> None:
    publish(
        EventKind.EPISODE_END, reason or f"episode {episode} ended", episode=episode
    )


def attempt(episode: Any, n: int) -> None:
    publish(
        EventKind.NOTE,
        f"re-planning episode {episode} (attempt {n})",
        episode=episode,
        data={"attempt": n},
    )


# ── pipeline stages ────────────────────────────────────────────────────

def snapshot(*, rgb=None, depth=None, joints=None, h5_path: str | None = None) -> None:
    """What the planner will see: RGB + depth heat-map + valid-depth fraction."""
    import numpy as np

    images: list[str] = []
    data: dict[str, Any] = {"h5": h5_path}
    if rgb is not None:
        a = np.asarray(rgb)
        data["rgb_shape"] = list(a.shape)
        images += _encode([a])
    if depth is not None:
        d = np.asarray(depth, dtype="float32")
        if d.ndim == 3:
            d = d[:, :, 0]
        finite = np.isfinite(d) & (d > 0)
        data["depth_shape"] = list(d.shape)
        data["depth_valid_frac"] = round(float(finite.mean()), 3)
        if finite.any():
            lo, hi = float(d[finite].min()), float(d[finite].max())
            data["depth_min_m"], data["depth_max_m"] = round(lo, 3), round(hi, 3)
            norm = np.zeros_like(d)
            if hi > lo:
                norm[finite] = 1.0 - (d[finite] - lo) / (hi - lo)
            heat = np.stack(
                [norm * 255, norm * 160, np.where(finite, 40, 0)], axis=-1
            ).astype("uint8")
            images += _encode([heat])
    if joints is not None:
        data["joints"] = [round(float(x), 3) for x in np.asarray(joints).reshape(-1)]
    publish(EventKind.OBSERVATION, "snapshot for planning", data=data, images=images)


def detections(
    objects: list,
    atoms: list | None = None,
    seconds: float | None = None,
    image: str | None = None,
) -> None:
    labels = []
    for o in objects or []:
        labels.append(o.get("label") if isinstance(o, dict) else str(o))
    publish(
        EventKind.TOOL,
        f"detected {len(labels)}: {', '.join(str(x) for x in labels[:8])}",
        data={
            "tool": "VLM detect",
            "objects": labels,
            "atoms": atoms or [],
            "seconds": seconds,
        },
        images=[image] if image else [],
    )


def no_goal(objects: list) -> None:
    """The VLM produced no goal predicate -- cuTAMP cannot plan without one.

    TiPToP's only predicate is ``on(movable, surface)``. A bare "pick up X" has
    no destination, so it is NOT expressible and every attempt will fail
    identically. This is a task-phrasing problem, not a perception one, and it
    must be unmistakable in the dashboard.
    """
    labels = [o.get("label") if isinstance(o, dict) else str(o) for o in objects or []]
    publish(
        EventKind.ERROR,
        "NO GOAL PREDICATE: TiPToP only supports on(object, surface). "
        "A 'pick up X' task has no destination, so cuTAMP has nothing to "
        "satisfy — rephrase as e.g. 'put the X in/on the Y'.",
        data={"detected": labels, "atoms": []},
    )


def segmentation(
    n_masks: int, seconds: float | None = None, image: str | None = None
) -> None:
    publish(
        EventKind.TOOL,
        f"SAM2 produced {n_masks} mask(s)",
        data={"tool": "SAM2", "masks": n_masks, "seconds": seconds},
        images=[image] if image else [],
    )


def association(per_object: dict, total: int) -> None:
    """Grasps kept per object after the contact-distance join.

    This is where success is decided: a grasp survives only if its contact point
    is within ``contact_threshold_m`` of that object's MASKED points. If a mask
    is misplaced, every grasp is discarded even when M2T2 found good ones.
    """
    detail = ", ".join(f"{k}={v}" for k, v in per_object.items())
    if total == 0:
        publish(
            EventKind.ERROR,
            "NO GRASPS ASSOCIATED with any object — M2T2's grasps were all "
            "further than contact_threshold_m from every mask. Usually the mask "
            f"is in the wrong place. ({detail})",
            data={"tool": "association", "per_object": per_object, "total": 0},
        )
    else:
        publish(
            EventKind.TOOL,
            f"associated {total} grasp(s): {detail}",
            data={"tool": "association", "per_object": per_object, "total": total},
        )


def final_grasp(label: str, confidence: float, image: str) -> None:
    publish(
        EventKind.TOOL,
        f"chosen grasp on {label} (confidence {confidence:.3f})",
        data={"tool": "chosen grasp", "object": label, "confidence": confidence},
        images=[image],
    )


def grasp_association_image(image: str) -> None:
    publish(
        EventKind.TOOL,
        "grasp association: green kept, red discarded",
        data={"tool": "association view"},
        images=[image],
    )


def grasps(n_grasps: int, best_score: float | None = None, seconds: float | None = None) -> None:
    txt = f"M2T2 produced {n_grasps} grasp(s)"
    if best_score is not None:
        txt += f", best score {best_score:.3f}"
    publish(
        EventKind.TOOL,
        txt,
        data={
            "tool": "M2T2",
            "grasps": n_grasps,
            "best_score": best_score,
            "seconds": seconds,
        },
    )


def planning(
    *,
    success: bool,
    seconds: float,
    steps: int | None = None,
    reason: str | None = None,
) -> None:
    if success:
        publish(
            EventKind.CODE,
            f"cuTAMP found a plan: {steps} step(s) in {seconds:.1f}s",
            data={"planner": "cuTAMP", "steps": steps, "seconds": seconds},
        )
    else:
        publish(
            EventKind.ERROR,
            f"cuTAMP found no plan after {seconds:.1f}s: {reason or 'unknown'}",
            data={"planner": "cuTAMP", "seconds": seconds, "reason": reason},
        )


def executing(step: int, total: int, label: str, kind: str) -> None:
    publish(
        EventKind.NOTE,
        f"step {step}/{total}: {label} ({kind})",
        data={"step": step, "total": total, "kind": kind},
    )


def motion_done(
    *,
    n_waypoints: int,
    planned_s: float | None = None,
    actual_s: float | None = None,
    converged: bool | None = None,
) -> None:
    """One row per completed motion, not per enqueue batch."""
    txt = f"motion done · {n_waypoints} rows"
    if planned_s is not None and actual_s is not None:
        txt += f" · planned {planned_s:.2f}s, took {actual_s:.2f}s"
    if converged is False:
        txt += " · DID NOT COMPLETE"
    publish(
        EventKind.MOTION,
        txt,
        data={
            "waypoints": n_waypoints,
            "planned_s": planned_s,
            "actual_s": actual_s,
            "converged": converged,
        },
    )


def error(text: str, **data: Any) -> None:
    publish(EventKind.ERROR, text, data=data)


def note(text: str, **data: Any) -> None:
    publish(EventKind.NOTE, text, data=data)


# ── helpers ────────────────────────────────────────────────────────────

def _encode(images: Any) -> list[str]:
    """Best-effort base64 PNG list. Never raises."""
    if images is None:
        return []
    if not isinstance(images, (list, tuple)):
        images = [images]
    out: list[str] = []
    for im in list(images)[:4]:
        try:
            if isinstance(im, str):
                out.append(im if len(im) > 512 else "")
                continue
            import base64
            import io

            import numpy as np
            from PIL import Image

            arr = np.asarray(im)
            if arr.ndim == 2:
                arr = np.stack([arr] * 3, -1)
            if arr.dtype != np.uint8:
                a = arr.astype("float32")
                rng = float(a.max() - a.min()) or 1.0
                arr = (((a - a.min()) / rng) * 255).astype("uint8")
            img = Image.fromarray(arr[:, :, :3])
            img.thumbnail((480, 480))
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            out.append(base64.b64encode(buf.getvalue()).decode())
        except Exception:
            continue
    return [s for s in out if s]
