#!/usr/bin/env python3
from __future__ import annotations
import argparse, os, sys
from pathlib import Path
# Allow sibling imports (tiptop_gemini_model_patch) when invoked as a script.
_here = Path(__file__).resolve().parent
if str(_here) not in sys.path: sys.path.insert(0, str(_here))
from tiptop_gemini_model_patch import apply_patch
apply_patch()
# Disable M2T2 hard-coded real-robot workspace bounds for RoboLab coordinates.
import tiptop.perception.m2t2 as _m2t2
_orig_generate_grasps_async = _m2t2.generate_grasps_async
async def _generate_grasps_async_no_bounds(*args, **kwargs):
    kwargs['apply_bounds'] = False
    return await _orig_generate_grasps_async(*args, **kwargs)
_m2t2.generate_grasps_async = _generate_grasps_async_no_bounds
import tiptop.perception_wrapper as _pw
_pw.generate_grasps_async = _generate_grasps_async_no_bounds
# Make gripper a tad deeper in grasp frame. TiPToP applies m2t2_to_tiptop_transform() to M2T2 poses.
# Increasing the z TCP offset here shifts the tool/gripper deeper along TiPToP's grasp approach frame.
_deeper_m = float(os.environ.get('TIPTOP_GRASP_DEEPER_M', '0.015'))
def _m2t2_to_tiptop_transform_deeper():
    import numpy as np
    from scipy.spatial.transform import Rotation
    base_to_tcp = np.eye(4)
    base_to_tcp[2, 3] = 0.1034 + _deeper_m
    to_tiptop_frame = np.eye(4)
    to_tiptop_frame[:3, :3] = Rotation.from_euler('xyz', np.array([np.pi, 0, -np.pi / 2])).as_matrix()
    return base_to_tcp @ to_tiptop_frame
_m2t2.m2t2_to_tiptop_transform = _m2t2_to_tiptop_transform_deeper
import tiptop.tiptop_run as _tr
_tr.m2t2_to_tiptop_transform = _m2t2_to_tiptop_transform_deeper
print(f'[tiptop-deeper-grasp] m2t2_to_tiptop TCP z offset = {0.1034 + _deeper_m:.4f} m (deeper +{_deeper_m:.4f} m)')
from tiptop.tiptop_h5 import run_tiptop_h5
p=argparse.ArgumentParser()
p.add_argument('--h5-path', required=True); p.add_argument('--task-instruction', required=True); p.add_argument('--output-dir', required=True)
p.add_argument('--max-planning-time', type=float, default=60.0); p.add_argument('--num-particles', type=int, default=128); p.add_argument('--opt-steps-per-skeleton', type=int, default=250)
args=p.parse_args()
run_tiptop_h5(h5_path=args.h5_path, task_instruction=args.task_instruction, output_dir=args.output_dir, max_planning_time=args.max_planning_time, num_particles=args.num_particles, opt_steps_per_skeleton=args.opt_steps_per_skeleton, rr_spawn=False)
