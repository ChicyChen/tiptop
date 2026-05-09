#!/usr/bin/env python3
"""Run TiPToP H5 deeper-grasp planner over a RoboLab-120 manifest.

Safety/resume features:
  - Per-subprocess virtual-address-space cap via `prlimit` (default 24 GB)
    so a single pathological cuTAMP run cannot OOM the whole machine.
  - Resume-friendly: if a task already has a metadata.json with a terminal
    planning_success (True or False), skip it and carry forward the result.
  - Merge with any prior summary.json so aggregate metrics remain consistent.
"""
from __future__ import annotations
import argparse, json, os, shutil, subprocess, time
from pathlib import Path


def _load_prev_summary(path: Path) -> dict[str, dict]:
    if not path.exists():
        return {}
    try:
        rows = json.loads(path.read_text())
        return {r['env_name']: r for r in rows if isinstance(r, dict) and 'env_name' in r}
    except Exception:
        return {}


def _terminal_meta(task_dir: Path):
    metas = sorted(task_dir.glob('*/metadata.json'), key=lambda x: x.stat().st_mtime, reverse=True)
    for m in metas:
        try:
            md = json.loads(m.read_text())
        except Exception:
            continue
        pl = md.get('planning', {})
        ps = pl.get('success')
        if ps in (True, False):
            return m, md
    return None, None


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--manifest', required=True)
    p.add_argument('--output-dir', default='tiptop_robolab120_outputs_deeper')
    p.add_argument('--limit', type=int, default=None)
    p.add_argument('--timeout-s', type=int, default=900)
    p.add_argument('--max-planning-time', type=float, default=20)
    p.add_argument('--num-particles', type=int, default=64)
    p.add_argument('--opt-steps-per-skeleton', type=int, default=100)
    p.add_argument('--grasp-deeper-m', type=float, default=0.015)
    p.add_argument('--mem-cap-gb', type=int, default=24,
                   help='Per-subprocess virtual address space cap (GB) via prlimit.')
    p.add_argument('--retry-failed', action='store_true',
                   help='Retry tasks that previously recorded planning_success=False.')
    args = p.parse_args()

    here = Path(__file__).resolve().parent
    tiptop = here.parents[1]                          # tiptop repo root
    runner = here / 'tiptop_h5_gemini_deeper_runner.py'
    manifest = json.loads(Path(args.manifest).read_text())
    if args.limit:
        manifest = manifest[:args.limit]
    out = Path(args.output_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)

    prev = _load_prev_summary(out / 'summary.json')
    summary: list[dict] = []

    systemd_run = shutil.which('systemd-run')
    pixi = os.path.expanduser('~/.pixi/bin/pixi')
    mem_bytes = int(args.mem_cap_gb) * 1024 ** 3

    for i, item in enumerate(manifest, 1):
        env = item['env_name']
        instr = item['instruction']
        h5 = item['h5_path']
        task_out = out / env
        task_out.mkdir(parents=True, exist_ok=True)

        # Resume: check for terminal metadata.
        meta_path, md = _terminal_meta(task_out)
        if md is not None:
            ps = md.get('planning', {}).get('success')
            fr = md.get('planning', {}).get('failure_reason')
            ga = md.get('perception', {}).get('grounded_atoms') or []
            plan = (meta_path.parent / 'tiptop_plan.json').exists()
            if ps is True or (ps is False and not args.retry_failed):
                rec = prev.get(env) or {
                    'env_name': env, 'instruction': instr, 'h5_path': h5,
                    'returncode': 'skipped_existing', 'error': None,
                    'planning_success': ps, 'has_plan': plan,
                    'failure_reason': fr, 'num_grounded_atoms': len(ga),
                    'wall_s': 0, 'run_dir': str(meta_path.parent),
                }
                print(f'[{i}/{len(manifest)}] SKIP {env} (planning_success={ps}, has_plan={plan})', flush=True)
                summary.append(rec)
                (out / 'summary.json').write_text(json.dumps(summary, indent=2))
                continue

        # Build command with memory cap.
        inner = [pixi, 'run', 'python', str(runner),
                 '--h5-path', h5, '--task-instruction', instr,
                 '--output-dir', str(task_out),
                 '--max-planning-time', str(args.max_planning_time),
                 '--num-particles', str(args.num_particles),
                 '--opt-steps-per-skeleton', str(args.opt_steps_per_skeleton)]
        if systemd_run:
            # Cgroup-v2 MemoryMax limits RSS (not virtual), so CUDA mapping huge
            # virtual ranges still works. Disable swap usage to keep OOM decisive.
            cmd = [systemd_run, '--user', '--scope', '--quiet',
                   f'-p', f'MemoryMax={mem_bytes}',
                   f'-p', 'MemorySwapMax=0',
                   '--'] + inner
        else:
            cmd = inner
        envv = os.environ.copy()
        envv['TIPTOP_GRASP_DEEPER_M'] = str(args.grasp_deeper_m)
        print(f'[{i}/{len(manifest)}] RUN {env}: {instr}', flush=True)
        t = time.time(); rc = None; err = None
        try:
            proc = subprocess.run(cmd, cwd=tiptop, timeout=args.timeout_s,
                                  text=True, capture_output=True, env=envv)
            rc = proc.returncode
            (task_out / 'stdout.txt').write_text(proc.stdout)
            (task_out / 'stderr.txt').write_text(proc.stderr)
        except subprocess.TimeoutExpired as e:
            rc = 124; err = 'timeout'
            (task_out / 'stdout.txt').write_text(e.stdout or '')
            (task_out / 'stderr.txt').write_text(e.stderr or '')
        except Exception as e:
            rc = 1; err = f'launch_error:{e!r}'

        # Determine result.
        meta_path, md = _terminal_meta(task_out)
        ps = None; fr = None; ga = []; rd = None; plan = False
        if md is not None:
            ps = md.get('planning', {}).get('success')
            fr = md.get('planning', {}).get('failure_reason')
            ga = md.get('perception', {}).get('grounded_atoms') or []
            rd = meta_path.parent
            plan = (rd / 'tiptop_plan.json').exists()
        if err is None and rc not in (0, 139) and ps is None:
            # OOM/prlimit kill = rc 137 (-9) or >128; classify.
            err = f'process_killed_rc={rc}'
        rec = {'env_name': env, 'instruction': instr, 'h5_path': h5,
               'returncode': rc, 'error': err,
               'planning_success': ps, 'has_plan': plan,
               'failure_reason': fr, 'num_grounded_atoms': len(ga),
               'wall_s': round(time.time() - t, 2),
               'run_dir': str(rd) if rd else None}
        print('RESULT', rec, flush=True)
        summary.append(rec)
        (out / 'summary.json').write_text(json.dumps(summary, indent=2))
        with open(out / 'summary.jsonl', 'a') as f:
            f.write(json.dumps(rec) + '\n')

    n = len(summary)
    ok = sum(1 for r in summary if r.get('planning_success') is True)
    metrics = {'num_tasks': n, 'planning_successes': ok,
               'planning_success_rate': ok / n if n else 0,
               'results': summary}
    (out / 'metrics.json').write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))


if __name__ == '__main__':
    main()
