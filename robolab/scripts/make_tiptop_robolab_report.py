#!/usr/bin/env python3
from __future__ import annotations
import argparse,json
from pathlib import Path
p=argparse.ArgumentParser(); p.add_argument('--planning-metrics',required=True); p.add_argument('--replay-metrics',required=True); p.add_argument('--out',default='TIPTOP_ROBOLAB120_REPORT.md'); args=p.parse_args()
plan=json.loads(Path(args.planning_metrics).read_text()); replay=json.loads(Path(args.replay_metrics).read_text()) if Path(args.replay_metrics).exists() else {'results':[]}
prs=plan.get('results',[]); rmap={r.get('task') or r.get('env_name'):r for r in replay.get('results',[])}
lines=['# TiPToP RoboLab-120 Report','',f"Planning tasks: {len(prs)}",f"Planning successes: {sum(1 for r in prs if r.get('planning_success') is True)}",f"Replay attempted: {len(rmap)}",f"RoboLab successes: {sum(1 for r in rmap.values() if r.get('success') is True)}",'', '## Per-task results','', '| Task | Planning | Replay/RoboLab | Failure reason | Plan s | Replay step |', '|---|---:|---:|---|---:|---:|']
for pr in prs:
 task=pr['env_name']; rr=rmap.get(task,{}); ps='✅' if pr.get('planning_success') is True else '❌'; rs='✅' if rr.get('success') is True else ('—' if not rr else '❌')
 reason=pr.get('failure_reason') or rr.get('failure_reason') or rr.get('error') or ''
 step='';
 try: step=(rr.get('env_results') or [{}])[0].get('step') or ''
 except Exception: pass
 lines.append(f"| {task} | {ps} | {rs} | {str(reason).replace('|','/')} | {pr.get('wall_s','')} | {step} |")
lines += ['', '## Failure cases', '']
for pr in prs:
 task=pr['env_name']; rr=rmap.get(task,{})
 if pr.get('planning_success') is not True or (rr and rr.get('success') is not True):
  lines.append(f"- **{task}**: planning={pr.get('planning_success')}, replay={rr.get('success') if rr else 'not_attempted'}, reason={pr.get('failure_reason') or rr.get('failure_reason') or rr.get('error') or 'unknown'}")
Path(args.out).write_text('\n'.join(lines)); print(args.out)
