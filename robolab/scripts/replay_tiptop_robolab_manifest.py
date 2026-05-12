#!/usr/bin/env python3
"""Replay TiPToP plans in RoboLab, recording metrics and per-task video.

Supports parallel workers via a filesystem claim-lock: each runner tries to
atomically create `<task_out>/.claim` before launching Isaac. If the claim
already exists and is fresh, another worker is handling the task and this one
skips it. After completion the claim is removed. The sharding args still work
and can be combined (e.g. two shards where each also contends via claims).
"""
from __future__ import annotations
import argparse, json, os, subprocess, time
from pathlib import Path


def _try_claim(claim_path: Path, ttl_s: float = 3600) -> bool:
    """Atomically create claim_path. Returns True if we own the claim."""
    try:
        fd = os.open(str(claim_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.write(fd, f'{os.getpid()} {time.time()}\n'.encode())
        os.close(fd)
        return True
    except FileExistsError:
        # Stale-claim recovery.
        try:
            st = claim_path.stat()
            if time.time() - st.st_mtime > ttl_s:
                claim_path.unlink(missing_ok=True)
                return _try_claim(claim_path, ttl_s)
        except FileNotFoundError:
            return _try_claim(claim_path, ttl_s)
        return False


def _release_claim(claim_path: Path):
    try:
        claim_path.unlink(missing_ok=True)
    except Exception:
        pass


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--planning-summary', required=True)
    p.add_argument('--output-dir', default='tiptop_robolab120_replay_deeper')
    p.add_argument('--limit', type=int, default=None)
    p.add_argument('--timeout-s', type=int, default=900)
    p.add_argument('--stride', type=int, default=4)
    p.add_argument('--max-joint-step', type=float, default=0.03)
    p.add_argument('--gripper-steps', type=int, default=60)
    p.add_argument('--post-steps', type=int, default=120)
    p.add_argument('--camera', default='both')
    p.add_argument('--shards', type=int, default=1)
    p.add_argument('--shard-idx', type=int, default=0)
    p.add_argument('--worker-tag', default='w0')
    p.add_argument('--task-subdirs', nargs='+', default=None,
                   help='Forwarded to replay_tiptop_plan_robolab.py; lets LH-CS / LH-vague tasks resolve.')
    p.add_argument(
        '--robolab-python',
        default=os.environ.get('ROBOLAB_PYTHON', ''),
        help='Path to the robolab_valts venv python (e.g. /path/to/robolab_valts/.venv/bin/python). '
             'Falls back to the ROBOLAB_PYTHON env var, then to ../robolab_valts/.venv/bin/python '
             'relative to the TiPToP repo root.')
    args = p.parse_args()

    tiptop_repo = Path(__file__).resolve().parents[2]  # tiptop repo root
    replay = Path(__file__).resolve().parent / 'replay_tiptop_plan_robolab.py'
    if args.robolab_python:
        py = Path(args.robolab_python)
    else:
        py = tiptop_repo.parent / 'robolab_valts/.venv/bin/python'
    if not py.exists():
        raise SystemExit(
            f'robolab_valts python not found at {py}. Pass --robolab-python or set ROBOLAB_PYTHON.')
    items = json.loads(Path(args.planning_summary).read_text())
    if isinstance(items, dict):
        items = items.get('results', [])
    items = [x for x in items
             if x.get('has_plan') or (x.get('run_dir') and Path(x['run_dir'], 'tiptop_plan.json').exists())]
    if args.limit:
        items = items[:args.limit]
    if args.shards > 1:
        items = [it for i, it in enumerate(items) if i % args.shards == args.shard_idx]
        print(f'[shard {args.shard_idx}/{args.shards}] handling {len(items)} tasks', flush=True)

    out = Path(args.output_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    results = []
    tag = args.worker_tag
    rs_path = out / f'replay_summary.{tag}.json'
    rsl_path = out / f'replay_summary.{tag}.jsonl'
    rm_path = out / f'replay_metrics.{tag}.json'

    for i, it in enumerate(items, 1):
        env = it['env_name']
        run_dir = Path(it['run_dir'])
        plan = run_dir / 'tiptop_plan.json'
        task_out = out / env
        task_out.mkdir(parents=True, exist_ok=True)
        existing = task_out / 'result.json'
        if existing.exists():
            try:
                r = json.loads(existing.read_text())
                print(f'[{tag} {i}/{len(items)}] SKIP existing {env}', flush=True)
                results.append(r)
                continue
            except Exception:
                pass
        claim = task_out / '.claim'
        if not _try_claim(claim):
            print(f'[{tag} {i}/{len(items)}] SKIP claimed-by-other {env}', flush=True)
            continue
        cmd = [str(py), str(replay), '--headless', '--task', env, '--plan', str(plan),
               '--output-dir', str(task_out), '--camera', args.camera,
               '--stride', str(args.stride), '--max-joint-step', str(args.max_joint_step),
               '--gripper-steps', str(args.gripper_steps), '--post-steps', str(args.post_steps)]
        if args.task_subdirs:
            cmd += ['--task-subdirs', *args.task_subdirs]
        print(f'[{tag} {i}/{len(items)}] REPLAY {env}', flush=True)
        t = time.time(); rc = None; err = None
        try:
            proc = subprocess.run(cmd, cwd=tiptop_repo, timeout=args.timeout_s,
                                  text=True, capture_output=True)
            rc = proc.returncode
            (task_out / 'stdout.txt').write_text(proc.stdout)
            (task_out / 'stderr.txt').write_text(proc.stderr)
        except subprocess.TimeoutExpired as e:
            rc = 124; err = 'timeout'
            def _to_str(x):
                if x is None: return ''
                if isinstance(x, bytes):
                    try: return x.decode('utf-8', errors='replace')
                    except Exception: return ''
                return str(x)
            (task_out / 'stdout.txt').write_text(_to_str(e.stdout))
            (task_out / 'stderr.txt').write_text(_to_str(e.stderr))
        finally:
            _release_claim(claim)
        r = {'task': env, 'instruction': it.get('instruction'),
             'planning_run_dir': str(run_dir), 'plan': str(plan),
             'returncode': rc, 'error': err,
             'wall_s': round(time.time() - t, 2),
             'success': False, 'env_results': None, 'video': None,
             'worker': tag}
        if existing.exists():
            try:
                r.update(json.loads(existing.read_text()))
            except Exception as e:
                r['parse_error'] = str(e)
        else:
            r['failure_reason'] = 'no_result_json'
        print('REPLAY_RESULT', r, flush=True)
        results.append(r)
        rs_path.write_text(json.dumps(results, indent=2))
        with open(rsl_path, 'a') as f:
            f.write(json.dumps(r) + '\n')

    n = len(results)
    ok = sum(1 for r in results if r.get('success') is True)
    metrics = {'num_replayed': n, 'replay_successes': ok,
               'replay_success_rate': ok / n if n else 0, 'results': results,
               'worker': tag}
    rm_path.write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))


if __name__ == '__main__':
    main()
