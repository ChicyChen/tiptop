"""Publish TiPToP's perception stages to the monitor by runtime patching.

``perception_wrapper`` and ``perception/m2t2.py`` are SHARED TiPToP modules used
by the sim path, so they are not edited. Instead the service calls
``instrument_perception()`` at startup, which wraps the three stage functions to
emit events. Failures here are swallowed: monitoring must never break a run.
"""

from __future__ import annotations

import time

import numpy as np

_installed = False

#: Set per-episode by the service so stage renderers can draw into the RGB.
_CTX: dict = {}


def set_frame_context(rgb=None, K=None, cam_from_world=None) -> None:
    """Give the instrumentation the image + camera for this attempt."""
    _CTX["rgb"] = rgb
    _CTX["K"] = K
    _CTX["cam_from_world"] = cam_from_world


def instrument_perception() -> None:
    """Wrap VLM detect / SAM2 segment / M2T2 grasps. Idempotent."""
    global _installed
    if _installed:
        return
    from tiptop.monitor import hooks as mon

    # ── VLM detect-and-translate ───────────────────────────────────────
    try:
        import tiptop.perception.gemini as g

        _orig_async = g.detect_and_translate_async

        async def detect_async(*a, **kw):
            t0 = time.perf_counter()
            bboxes, atoms = await _orig_async(*a, **kw)
            try:
                from tiptop.monitor import viz

                img = None
                if _CTX.get("rgb") is not None:
                    img = viz.vlm_boxes(_CTX["rgb"], bboxes)
                mon.detections(
                    bboxes, atoms, time.perf_counter() - t0, image=img
                )
                if not atoms:
                    # TiPToP's only predicate is on(movable, surface); a bare
                    # "pick up X" has no destination and yields NO goal, which
                    # cuTAMP cannot plan for. Say so loudly.
                    mon.no_goal(bboxes)
            except Exception:
                pass
            return bboxes, atoms

        g.detect_and_translate_async = detect_async
    except Exception:
        pass

    # ── SAM2 segmentation ──────────────────────────────────────────────
    try:
        import tiptop.perception.sam2 as s

        _orig_seg = s.sam2_segment_objects

        def segment(*a, **kw):
            t0 = time.perf_counter()
            masks = _orig_seg(*a, **kw)
            try:
                from tiptop.monitor import viz

                n = len(masks) if masks is not None else 0
                labels = None
                if len(a) > 1 and isinstance(a[1], (list, tuple)):
                    labels = [str(b.get("label", "?")) for b in a[1]]
                img = (
                    viz.sam2_masks(_CTX["rgb"], masks, labels)
                    if _CTX.get("rgb") is not None
                    else None
                )
                mon.segmentation(n, time.perf_counter() - t0, image=img)
            except Exception:
                pass
            return masks

        s.sam2_segment_objects = segment
        import tiptop.perception_wrapper as pw

        if hasattr(pw, "sam2_segment_objects"):
            pw.sam2_segment_objects = segment
    except Exception:
        pass

    # ── M2T2 grasp generation ──────────────────────────────────────────
    try:
        import tiptop.perception.m2t2 as m

        _orig_grasp = m.generate_grasps_async

        async def grasps_async(*a, **kw):
            t0 = time.perf_counter()
            out = await _orig_grasp(*a, **kw)
            try:
                n, best = _summarise_grasps(out)
                mon.grasps(n, best, time.perf_counter() - t0)
            except Exception:
                pass
            return out

        m.generate_grasps_async = grasps_async
        import tiptop.perception_wrapper as pw

        if hasattr(pw, "generate_grasps_async"):
            pw.generate_grasps_async = grasps_async
    except Exception:
        pass

    # ── the association step: which grasps survive? ────────────────────
    # This is where success is actually decided: grasps within
    # contact_threshold_m of an object's masked points are kept, the rest are
    # discarded. Wrapping process_scene lets us report the outcome per object
    # AND draw kept-vs-discarded, which is the single most diagnostic picture.
    try:
        import tiptop.tiptop_run as tr

        _orig_process = tr.process_scene_geometry

        def process_scene(*a, **kw):
            out = _orig_process(*a, **kw)
            try:
                _report_association(out)
            except Exception:
                pass
            return out

        tr.process_scene_geometry = process_scene
    except Exception:
        pass

    _installed = True


def _report_association(processed) -> None:
    """Publish per-object grasp counts + a kept/discarded visualisation."""
    from tiptop.monitor import hooks as mon
    from tiptop.monitor import viz

    grasps = getattr(processed, "grasps", None)
    if not isinstance(grasps, dict):
        return
    per_object = {}
    all_contacts, kept_flags = [], []
    best = (None, -1.0, None)
    for label, g in grasps.items():
        poses = g.get("poses")
        n = int(poses.shape[0]) if hasattr(poses, "shape") else 0
        per_object[str(label)] = n
        conf = g.get("confidences")
        if n and conf is not None and len(conf):
            i = int(np.argmax(conf))
            if float(conf[i]) > best[1]:
                best = (str(label), float(conf[i]), poses[i])
        c = g.get("contacts")
        if c is not None and len(c):
            all_contacts.append(np.asarray(c).reshape(-1, 3))
            kept_flags.append(np.ones(len(c), dtype=bool))

    total = sum(per_object.values())
    mon.association(per_object, total)

    ctx = _CTX
    if ctx.get("rgb") is None or ctx.get("K") is None:
        return
    if total and best[2] is not None:
        img = viz.final_grasp(
            ctx["rgb"], best[2], ctx["K"], ctx["cam_from_world"],
            label=best[0] or "", confidence=best[1],
        )
        if img:
            mon.final_grasp(best[0] or "", best[1], img)
    if all_contacts:
        cont = np.concatenate(all_contacts)
        keep = np.concatenate(kept_flags)
        img = viz.m2t2_grasps(
            ctx["rgb"], cont, ctx["K"], ctx["cam_from_world"], keep
        )
        if img:
            mon.grasp_association_image(img)


def _summarise_grasps(out) -> tuple[int, float | None]:
    """Count grasps and the best confidence from M2T2's return value."""
    try:
        grasps = out[0] if isinstance(out, (tuple, list)) else out
        scores = out[1] if isinstance(out, (tuple, list)) and len(out) > 1 else None
        n = sum(len(g) for g in grasps) if hasattr(grasps, "__iter__") else 0
        best = None
        if scores is not None:
            flat = [float(x) for s in scores for x in (s if hasattr(s, "__iter__") else [s])]
            best = max(flat) if flat else None
        return n, best
    except Exception:
        return 0, None
