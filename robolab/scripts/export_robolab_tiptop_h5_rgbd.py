#!/usr/bin/env python3
"""Export RoboLab observations for TiPToP with RGB-D cameras only.

This registers standard RoboLab tasks with RGB+depth external/wrist cameras and
writes TiPToP H5 observations.  No object/world GT is exported.
"""
from __future__ import annotations

import argparse, json, sys, traceback
from pathlib import Path
import cv2  # noqa
import h5py, numpy as np
from isaaclab.app import AppLauncher

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--task', nargs='+', required=True)
p.add_argument('--output-dir', default='tiptop_robolab_h5')
p.add_argument('--camera', default='external_cam', choices=['external_cam','wrist_cam'])
# Task subfolders to glob for class definitions. Robolab's default
# (`benchmark`, `custom`) doesn't include the long-horizon families,
# so the canonical robolab-120 export keeps the default while LH-CS /
# LH-vague runs pass --task-subdirs to add e.g. `long_horizon/common_sense`.
p.add_argument('--task-subdirs', nargs='+', default=None,
               help='Robolab task subdirs to search (default: robolab.constants.DEFAULT_TASK_SUBFOLDERS)')
AppLauncher.add_app_launcher_args(p)
args,_=p.parse_known_args(); args.enable_cameras=True
app=AppLauncher(args).app

import isaaclab.sim as sim_utils
import isaaclab.envs.mdp as mdp
from isaaclab.managers import ObservationGroupCfg as ObsGroup, ObservationTermCfg as ObsTerm, SceneEntityCfg
from isaaclab.sensors import TiledCameraCfg
from isaaclab.utils import configclass
from robolab.constants import DEFAULT_TASK_SUBFOLDERS, TASK_DIR
from robolab.core.environments.factory import auto_discover_and_create_cfgs, get_envs
from robolab.core.environments.runtime import create_env
from robolab.core.observations.observation_utils import generate_obs_cfg
from robolab.robots.droid import DroidCfg, DroidJointPositionActionCfg, ProprioceptionObservationCfg, contact_gripper
from robolab.variations.backgrounds import HomeOfficeBackgroundCfg
from robolab.variations.lighting import SphereLightCfg

@configclass
class RgbdExternalCameraCfg:
    external_cam = TiledCameraCfg(
        prim_path='{ENV_REGEX_NS}/external_cam', height=720, width=1280, data_types=['rgb','depth'],
        spawn=sim_utils.PinholeCameraCfg(focal_length=2.1, focus_distance=28.0, horizontal_aperture=5.376, vertical_aperture=3.024),
        offset=TiledCameraCfg.OffsetCfg(pos=(0.05,0.57,0.66), rot=(-0.393,-0.195,0.399,0.805), convention='opengl'))

@configclass
class RgbdImageObsCfg(ObsGroup):
    external_cam = ObsTerm(func=mdp.observations.image, params={'sensor_cfg': SceneEntityCfg('external_cam'), 'data_type':'rgb', 'normalize':False})
    external_depth = ObsTerm(func=mdp.observations.image, params={'sensor_cfg': SceneEntityCfg('external_cam'), 'data_type':'depth', 'normalize':False})
    wrist_cam = ObsTerm(func=mdp.observations.image, params={'sensor_cfg': SceneEntityCfg('wrist_cam'), 'data_type':'rgb', 'normalize':False})
    wrist_depth = ObsTerm(func=mdp.observations.image, params={'sensor_cfg': SceneEntityCfg('wrist_cam'), 'data_type':'depth', 'normalize':False})
    def __post_init__(self):
        self.enable_corruption=False; self.concatenate_terms=False

@configclass
class DroidRgbdWristCfg(DroidCfg):
    wrist_cam = TiledCameraCfg(
        prim_path='{ENV_REGEX_NS}/robot/Gripper/Robotiq_2F_85/base_link/wrist_cam',
        height=720,
        width=1280,
        data_types=['rgb','depth'],
        spawn=sim_utils.PinholeCameraCfg(focal_length=2.8, focus_distance=28.0, horizontal_aperture=5.376, vertical_aperture=3.024),
        offset=TiledCameraCfg.OffsetCfg(pos=(0.011,-0.031,-0.074), rot=(-0.420,0.570,0.576,-0.409), convention='opengl'),
    )


def _to_np(x):
    return x.detach().cpu().numpy() if hasattr(x,'detach') else np.asarray(x)

def register(tasks, task_subdirs=None):
    # The `tasks` arg is filtered later via get_envs(task=...). The factory's
    # auto-discover entry no longer accepts a `tasks=` kwarg (robolab API drift,
    # 2026-05-11) — it always registers every task file in the given subdirs.
    subdirs = task_subdirs if task_subdirs is not None else DEFAULT_TASK_SUBFOLDERS
    ObservationCfg=generate_obs_cfg({'image_obs': RgbdImageObsCfg(), 'proprio_obs': ProprioceptionObservationCfg()})
    auto_discover_and_create_cfgs(task_dir=TASK_DIR, task_subdirs=subdirs, pattern='*.py', env_prefix='', env_postfix='', observations_cfg=ObservationCfg(), actions_cfg=DroidJointPositionActionCfg(), robot_cfg=DroidRgbdWristCfg, camera_cfg=[RgbdExternalCameraCfg], lighting_cfg=SphereLightCfg, background_cfg=HomeOfficeBackgroundCfg, contact_gripper=contact_gripper, dt=1/(60*2), render_interval=8, decimation=8, seed=1)

def intrinsics_from_cfg(width, height, focal_length, h_aperture, v_aperture):
    fx = focal_length / h_aperture * width
    fy = focal_length / v_aperture * height
    return np.array([[fx,0,width/2],[0,fy,height/2],[0,0,1]], dtype=np.float32)

def cam_pose_from_sensor(env, camera):
    cam = env.scene[camera]
    data = cam.data
    pos = _to_np(data.pos_w)[0].astype(np.float32)
    # TiPToP H5 loader expects [w,x,y,z] named quat_w_ros.
    if hasattr(data, 'quat_w_ros'):
        quat = _to_np(data.quat_w_ros)[0].astype(np.float32)
    else:
        quat = _to_np(data.quat_w_world)[0].astype(np.float32)
    return pos, quat

def intrinsics_from_sensor(env, camera):
    cam = env.scene[camera]
    if hasattr(cam.data, 'intrinsic_matrices'):
        return _to_np(cam.data.intrinsic_matrices)[0].astype(np.float32)
    if camera == 'external_cam':
        return intrinsics_from_cfg(1280,720,2.1,5.376,3.024)
    return intrinsics_from_cfg(1280,720,2.8,5.376,3.024)

def export_one(env, cfg, env_name, out_dir):
    obs,_=env.reset(); obs,_=env.reset()
    import torch
    action_dim=getattr(getattr(env,'action_manager',None),'total_action_dim',None) or env.action_space.shape[-1]
    for _ in range(3): obs,*_=env.step(torch.zeros((1,action_dim),device=env.device))
    image_obs=obs['image_obs']
    if args.camera == 'external_cam':
        rgb=_to_np(image_obs['external_cam'][0]); depth=_to_np(image_obs['external_depth'][0])
    else:
        rgb=_to_np(image_obs['wrist_cam'][0]); depth=_to_np(image_obs['wrist_depth'][0])
    k=intrinsics_from_sensor(env, args.camera)
    if depth.ndim==3 and depth.shape[-1]==1: depth=depth[...,0]
    rgb=np.clip(rgb,0,255).astype(np.uint8); depth=depth.astype(np.float32); depth[~np.isfinite(depth)]=0
    pos,quat=cam_pose_from_sensor(env, args.camera)
    q_init=_to_np(obs['proprio_obs']['arm_joint_pos'][0]).astype(np.float32)
    d=out_dir/env_name; d.mkdir(parents=True,exist_ok=True); h5=d/'obs.h5'
    with h5py.File(h5,'w') as f:
        f.create_dataset('rgb',data=rgb,compression='gzip'); f.create_dataset('depth',data=depth,compression='gzip')
        f.create_dataset('intrinsic_matrix',data=k); f.create_dataset('pos_w',data=pos); f.create_dataset('quat_w_ros',data=quat); f.create_dataset('q_init',data=q_init)
    meta={'env_name':env_name,'task_name':getattr(cfg,'_task_name',env_name),'instruction':cfg.instruction,'h5_path':str(h5),'allowed_inputs_only':['rgb','depth','intrinsic_matrix','pos_w','quat_w_ros','q_init'],'camera':args.camera}
    (d/'metadata.json').write_text(json.dumps(meta,indent=2)); print('EXPORT',env_name,h5); return meta

def main():
    out=Path(args.output_dir).resolve(); out.mkdir(parents=True,exist_ok=True); register(args.task, task_subdirs=args.task_subdirs); metas=[]
    for env_name in get_envs(task=args.task):
        env,cfg=create_env(env_name,device=args.device,num_envs=1,use_fabric=True,policy='tiptop_h5_export')
        try: metas.append(export_one(env,cfg,env_name,out))
        finally: env.close()
    (out/'manifest.json').write_text(json.dumps(metas,indent=2)); app.close()
if __name__=='__main__':
    try: main()
    except Exception as e: print('ERR',e); traceback.print_exc(); app.close(); sys.exit(1)
