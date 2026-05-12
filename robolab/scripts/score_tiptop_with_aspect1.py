"""Replay TiPToP per-trial outputs through the vlm-orchestrator Aspect-1
data shape.

TiPToP owns its own env loop (export H5 → plan → replay), so the
orchestrator's online logger never sees these episodes. This script
walks a tiptop osmo-results tree (or any per-trial-dir layout) and
writes per-episode ``metadata.json`` files in the shape the
``build_tiptop_lh_cs_jsons.py`` aggregator + the
``main_results`` dashboard expect.

Output layout (matches vlm-orchestrator's per-episode dir, mirrors
``score_capx_with_aspect1.py``):

    <out_root>/<task_slug>/episode_<NN>/
        metadata.json           # written by this script
        task_failures.jsonl     # stub: episode_start + episode_end only;
                                # tiptop's replay doesn't dump gt_state so
                                # aspect-1 detail events (WRONG_OBJECT_PICKED,
                                # OBJECT_REGRESSION, ...) cannot fire.
                                # Adding a gt_state dumper to
                                # replay_tiptop_plan_robolab.py would let
                                # us run a real TaskFailureLogger here.

Input layout (tiptop osmo eval; see osmo/run-tiptop-lh-cs.yaml):

    <tiptop_results>/<TaskName>/<run_ts>/trial_<N>/
        plan/<timestamp>/metadata.json
        replay/<TaskName>/result.json

Usage:
    python robolab/scripts/score_tiptop_with_aspect1.py \\
        --tiptop-output-dir /mnt/amlfs-04/home/siyic/tiptop-results \\
        --out-root ./outputs/tiptop_aspect1
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path


def _slugify(text: str) -> str:
    """Match vlm-orchestrator's instruction → directory slug convention."""
    return re.sub(r"[^\w\s-]", "", text or "").strip().replace(" ", "_") or "episode"


def _read_planning(trial_dir: Path) -> dict:
    """Return the planning metadata block (dict) or empty dict if missing.

    Tiptop's runner writes plan/<timestamp>/metadata.json with at minimum:
        {planning: {success, failure_reason, duration},
         perception: {grounded_atoms, duration},
         task_instruction}
    """
    plan_root = trial_dir / "plan"
    if not plan_root.is_dir():
        return {}
    # Most recent timestamped dir.
    ts_dirs = sorted(d for d in plan_root.iterdir() if d.is_dir())
    if not ts_dirs:
        return {}
    md_path = ts_dirs[-1] / "metadata.json"
    if not md_path.is_file():
        return {}
    try:
        return json.loads(md_path.read_text())
    except (OSError, json.JSONDecodeError):
        return ""


def _read_replay(trial_dir: Path) -> dict:
    """Return the replay result dict for the (single) env in this trial.

    Tiptop's replay_tiptop_plan_robolab.py writes
    <env_name>/result.json with {task, instruction, success, env_results,
    subtask_info, executed_steps, ...}.
    """
    rep_root = trial_dir / "replay"
    if not rep_root.is_dir():
        return {}
    # Walk the (single) env subdir.
    env_dirs = [d for d in rep_root.iterdir() if d.is_dir()]
    if not env_dirs:
        return {}
    rj = env_dirs[0] / "result.json"
    if not rj.is_file():
        return {}
    try:
        return json.loads(rj.read_text())
    except (OSError, json.JSONDecodeError):
        return {}


def _episode_row(plan_md: dict, replay_md: dict) -> dict:
    """Derive the (success, reason, steps, duration) tuple for one trial.

    success = planning success AND replay success.
    reason  = planning.failure_reason if planning failed,
              else replay's terminal failure summary if replay failed,
              else "success".
    steps   = executed_steps from replay (best proxy; if no plan ran, 0).
    duration_s = perception duration + cuTAMP duration (the only time
                 signals tiptop emits today; doesn't include Isaac Sim
                 replay wall-clock).
    """
    planning = plan_md.get("planning", {}) or {}
    perception = plan_md.get("perception", {}) or {}
    plan_ok = bool(planning.get("success", False))
    replay_ok = bool(replay_md.get("success", False))
    success = plan_ok and replay_ok

    if not plan_ok:
        reason = planning.get("failure_reason") or "planning failed"
    elif not replay_ok:
        # Replay's env_results carries the robolab termination summary.
        env_results = replay_md.get("env_results") or []
        first = env_results[0] if env_results else {}
        reason = (
            first.get("failure_reason")
            or first.get("reason")
            or "replay failed"
        )
    else:
        reason = "success"

    steps = int(replay_md.get("executed_steps") or 0)
    duration_s = float(perception.get("duration") or 0.0) + float(
        planning.get("duration") or 0.0
    )

    return {
        "success": success,
        "score": 1.0 if success else 0.0,
        "steps": steps,
        "duration_s": duration_s,
        "reason": str(reason)[:160],
        "planning_success": plan_ok,
        "replay_success": replay_ok,
    }


def _write_stub_task_failures(episode_dir: Path, instruction: str,
                              episode_id: int, success: bool) -> None:
    """Write a minimal task_failures.jsonl with just the header + episode
    boundary events. Real Aspect-1 events need per-step gt_state from
    Isaac Sim, which tiptop's replay doesn't currently dump."""
    path = episode_dir / "task_failures.jsonl"
    with path.open("w") as fh:
        fh.write(json.dumps({
            "type": "header",
            "schema_version": 1,
            "source": "tiptop",
            "note": "Aspect-1 events unavailable: tiptop replay does not dump gt_state.",
        }) + "\n")
        fh.write(json.dumps({
            "type": "episode_start",
            "step": 0,
            "instruction": instruction,
            "episode_id": episode_id,
        }) + "\n")
        fh.write(json.dumps({
            "type": "episode_end",
            "step": 0,
            "instruction": instruction,
            "episode_id": episode_id,
            "success": success,
        }) + "\n")


def _iter_trials(tiptop_root: Path):
    """Yield (task_name, run_ts, trial_idx, trial_dir) for every trial dir.

    Expected layout: <tiptop_root>/<TaskName>/<run_ts>/trial_<N>/...
    """
    if not tiptop_root.is_dir():
        return
    for task_dir in sorted(tiptop_root.iterdir()):
        if not task_dir.is_dir():
            continue
        for ts_dir in sorted(task_dir.iterdir()):
            if not ts_dir.is_dir():
                continue
            for trial_dir in sorted(ts_dir.iterdir()):
                if not trial_dir.is_dir() or not trial_dir.name.startswith("trial_"):
                    continue
                try:
                    idx = int(trial_dir.name.split("_", 1)[1])
                except ValueError:
                    continue
                yield task_dir.name, ts_dir.name, idx, trial_dir


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tiptop-output-dir",
        required=True,
        help="Root containing <TaskName>/<run_ts>/trial_<N>/ subdirs "
             "(e.g. /mnt/amlfs-04/home/<user>/tiptop-results).",
    )
    parser.add_argument(
        "--out-root",
        required=True,
        help="Where to write per-episode dirs (<out>/<task>/episode_<NN>/).",
    )
    parser.add_argument(
        "--use-instruction-slug",
        action="store_true",
        help="Use slugified instruction as task dir name (matches vlm-orch). "
             "Default uses the robolab class name (matches dashboard's batch JSONs).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Walk inputs and report counts without writing.",
    )
    args = parser.parse_args()

    tiptop_root = Path(args.tiptop_output_dir).resolve()
    out_root = Path(args.out_root).resolve()
    if not tiptop_root.is_dir():
        print(f"tiptop-output-dir not found: {tiptop_root}", file=sys.stderr)
        return 1

    # Group trials by task so episode_<N> numbering is sequential per task.
    grouped: dict[str, list[tuple[Path, int]]] = {}
    for task_name, run_ts, trial_idx, trial_dir in _iter_trials(tiptop_root):
        # If multiple run_ts exist for the same task, keep them all in order;
        # episode_NN indexing is monotonic across all trials we see.
        grouped.setdefault(task_name, []).append((trial_dir, trial_idx))

    total_trials = sum(len(v) for v in grouped.values())
    print(f"[scorer] found {total_trials} trial(s) across {len(grouped)} task(s)")
    if args.dry_run:
        for task_name, trials in grouped.items():
            print(f"  {task_name}: {len(trials)} trial(s)")
        return 0

    written = 0
    for task_name, trials in grouped.items():
        for ep_idx, (trial_dir, _native_trial_idx) in enumerate(sorted(trials)):
            plan_md = _read_planning(trial_dir)
            replay_md = _read_replay(trial_dir)

            instruction = (
                plan_md.get("task_instruction")
                or replay_md.get("instruction")
                or ""
            )
            slug = (
                _slugify(instruction)
                if args.use_instruction_slug
                else task_name
            )
            episode_dir = out_root / slug / f"episode_{ep_idx:03d}"
            episode_dir.mkdir(parents=True, exist_ok=True)

            row = _episode_row(plan_md, replay_md)

            metadata = {
                "episode_id": ep_idx,
                "instruction": instruction,
                "task_slug": slug,
                "task_class": task_name,
                "episode_dir": str(episode_dir),
                "trial_dir": str(trial_dir),
                "num_steps": row["steps"],
                "success": row["success"],
                "score": row["score"],
                "duration_s": row["duration_s"],
                "reason": row["reason"],
                "planning_success": row["planning_success"],
                "replay_success": row["replay_success"],
                "source": "tiptop",
                "end_timestamp": time.time(),
            }
            (episode_dir / "metadata.json").write_text(
                json.dumps(metadata, indent=2)
            )
            _write_stub_task_failures(
                episode_dir,
                instruction=instruction,
                episode_id=ep_idx,
                success=row["success"],
            )
            written += 1

    print(f"[scorer] wrote {written} episode dir(s) under {out_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
