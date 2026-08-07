"""Plan-loading adapter shared by the TiPToP franky service.

The one-shot CLI that used to live here was removed: the task must come from the
driver's per-frame ``prompt``, not a command-line flag, so
``tiptop/franky/serve_franky.py`` is the entry point. Only the plan adapter below
is still used.

Three phases, mirroring the sim/robolab flow but with our driver as the sensor
and the arm:

1. **Snapshot** — wait for a driver frame, write it as the H5 that
   ``tiptop_h5.py`` already consumes (no ZED, no FoundationStereo).
2. **Plan** — run the existing perception + cuTAMP pipeline, producing
   ``tiptop_plan.json``. Uses M2T2 for grasps and SAM2 for segmentation, exactly
   as in the sim eval.
3. **Execute** — replay the plan through ``execute_cutamp_plan`` with our
   ``FrankyClient`` injected.

``--plan-only`` stops after phase 2, which is how to validate perception and TAMP
on a new scene with zero risk to the arm.

Nothing in shared TiPToP is modified: ``execute_cutamp_plan`` already accepts an
injected client, and the planner is reached through its normal H5 entry point.

Usage
-----
    pixi run python -m tiptop.franky.run_franky \\
        --task-instruction "put the blue block in the bowl" --plan-only

    pixi run python -m tiptop.franky.run_franky \\
        --task-instruction "put the blue block in the bowl"
"""

from __future__ import annotations

from pathlib import Path


class _PlanArrays:
    """Adapter: the saved JSON stores plain arrays, but
    ``execute_cutamp_plan`` reads ``step["plan"].position.cpu().numpy()``.

    Exposes the tensor-ish surface it expects without importing torch or
    reaching into cuTAMP internals.
    """

    def __init__(self, positions, velocities) -> None:
        self.position = _AsNumpy(positions)
        self.velocity = _AsNumpy(velocities)


class _AsNumpy:
    def __init__(self, arr) -> None:
        self._arr = arr

    def cpu(self):
        return self

    def numpy(self):
        return self._arr

    def __len__(self) -> int:
        return len(self._arr)


def _load_plan_for_execution(plan_path: Path) -> list[dict]:
    """``load_tiptop_plan`` output -> the shape ``execute_cutamp_plan`` wants."""
    from tiptop.planning import load_tiptop_plan

    saved = load_tiptop_plan(plan_path)
    out: list[dict] = []
    for step in saved["steps"]:
        if step["type"] == "trajectory":
            out.append(
                {
                    "type": "trajectory",
                    "label": step["label"],
                    "plan": _PlanArrays(step["positions"], step["velocities"]),
                    "dt": step["dt"],
                }
            )
        elif step["type"] == "gripper":
            out.append(
                {
                    "type": "gripper",
                    "label": step["label"],
                    "action": step["action"],
                }
            )
        else:
            raise ValueError(f"unknown step type in saved plan: {step['type']}")
    return out


def _find_plan(plan_root: Path) -> Path | None:
    """``run_tiptop_h5`` writes into a timestamped subdirectory."""
    hits = sorted(plan_root.rglob("tiptop_plan.json"), key=lambda p: p.stat().st_mtime)
    return hits[-1] if hits else None


