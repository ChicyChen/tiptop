#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, os, re, sys, time, traceback
from pathlib import Path
import cv2  # must precede isaaclab
import numpy as np
from isaaclab.app import AppLauncher

p=argparse.ArgumentParser(description='Replay a TiPToP joint plan in RoboLab and record video.')
p.add_argument('--task', default='BananaInBowlTask')
p.add_argument('--plan', required=True)
p.add_argument('--output-dir', default='tiptop_replay_outputs/BananaInBowlTask')
p.add_argument('--camera', default='wrist_cam', choices=['wrist_cam','external_cam','both'])
p.add_argument('--stride', type=int, default=4, help='Replay every Nth trajectory waypoint to keep sim runtime reasonable.')
p.add_argument('--max-joint-step', type=float, default=0.0, help='If >0, interpolate commands so no arm joint target changes by more than this many radians per env step. This prevents sudden large position-target jumps during replay.')
p.add_argument('--settle-steps', type=int, default=20)
p.add_argument('--gripper-steps', type=int, default=20)
p.add_argument('--post-steps', type=int, default=80)
p.add_argument('--fps', type=float, default=15)
AppLauncher.add_app_launcher_args(p)
args,_=p.parse_known_args(); args.enable_cameras=True
app=AppLauncher(args).app

try:
 import torch
 import isaaclab.sim as sim_utils, isaaclab.envs.mdp as mdp
 from isaaclab.managers import ObservationGroupCfg as ObsGroup, ObservationTermCfg as ObsTerm, SceneEntityCfg
 from isaaclab.sensors import TiledCameraCfg
 from isaaclab.utils import configclass
 from robolab.constants import DEFAULT_TASK_SUBFOLDERS,TASK_DIR
 from robolab.core.environments.factory import auto_discover_and_create_cfgs
 from robolab.core.environments.runtime import create_env
 from robolab.core.observations.observation_utils import generate_obs_cfg, unpack_image_obs
 from robolab.robots.droid import DroidCfg,DroidJointPositionActionCfg,ProprioceptionObservationCfg,contact_gripper
 from robolab.variations.backgrounds import HomeOfficeBackgroundCfg
 from robolab.variations.lighting import SphereLightCfg
 from robolab.core.logging.results import get_all_env_subtask_infos

 @configclass
 class RgbdExternalCameraCfg:
  external_cam=TiledCameraCfg(prim_path='{ENV_REGEX_NS}/external_cam',height=720,width=1280,data_types=['rgb','depth'],spawn=sim_utils.PinholeCameraCfg(focal_length=2.1,focus_distance=28.0,horizontal_aperture=5.376,vertical_aperture=3.024),offset=TiledCameraCfg.OffsetCfg(pos=(0.05,0.57,0.66),rot=(-0.393,-0.195,0.399,0.805),convention='opengl'))
 @configclass
 class DroidRgbdWristCfg(DroidCfg):
  wrist_cam=TiledCameraCfg(prim_path='{ENV_REGEX_NS}/robot/Gripper/Robotiq_2F_85/base_link/wrist_cam',height=720,width=1280,data_types=['rgb','depth'],spawn=sim_utils.PinholeCameraCfg(focal_length=2.8,focus_distance=28.0,horizontal_aperture=5.376,vertical_aperture=3.024),offset=TiledCameraCfg.OffsetCfg(pos=(0.011,-0.031,-0.074),rot=(-0.420,0.570,0.576,-0.409),convention='opengl'))
 @configclass
 class RgbdObsCfg(ObsGroup):
  external_cam=ObsTerm(func=mdp.observations.image,params={'sensor_cfg':SceneEntityCfg('external_cam'),'data_type':'rgb','normalize':False})
  external_depth=ObsTerm(func=mdp.observations.image,params={'sensor_cfg':SceneEntityCfg('external_cam'),'data_type':'depth','normalize':False})
  wrist_cam=ObsTerm(func=mdp.observations.image,params={'sensor_cfg':SceneEntityCfg('wrist_cam'),'data_type':'rgb','normalize':False})
  wrist_depth=ObsTerm(func=mdp.observations.image,params={'sensor_cfg':SceneEntityCfg('wrist_cam'),'data_type':'depth','normalize':False})
  def __post_init__(self): self.enable_corruption=False; self.concatenate_terms=False
 Obs=generate_obs_cfg({'image_obs':RgbdObsCfg(),'proprio_obs':ProprioceptionObservationCfg()})
 auto_discover_and_create_cfgs(task_dir=TASK_DIR,task_subdirs=DEFAULT_TASK_SUBFOLDERS,tasks=[args.task],pattern='*.py',env_prefix='',env_postfix='',observations_cfg=Obs(),actions_cfg=DroidJointPositionActionCfg(),robot_cfg=DroidRgbdWristCfg,camera_cfg=[RgbdExternalCameraCfg],lighting_cfg=SphereLightCfg,background_cfg=HomeOfficeBackgroundCfg,contact_gripper=contact_gripper,dt=1/(60*2),render_interval=8,decimation=8,seed=1)
 env,cfg=create_env(args.task,device=args.device,num_envs=1,use_fabric=True,policy='tiptop_replay')
 out=Path(args.output_dir); out.mkdir(parents=True,exist_ok=True)
 plan=json.loads(Path(args.plan).read_text())
 obs,_=env.reset(); obs,_=env.reset()
 action_dim=getattr(env.action_manager,'total_action_dim',8)
 gripper=0.0
 current_q=np.array(plan.get('q_init', plan['steps'][0]['positions'][0]), dtype=np.float32)
 writers={}
 def np_img(t):
  a=t.detach().cpu().numpy() if hasattr(t,'detach') else np.asarray(t)
  if a.ndim==4: a=a[0]
  return a.astype(np.uint8)
 def frame_from_obs(obs):
  imgs=[]
  if args.camera in ('wrist_cam','both'): imgs.append(('wrist',np_img(obs['image_obs']['wrist_cam'])))
  if args.camera in ('external_cam','both'): imgs.append(('external',np_img(obs['image_obs']['external_cam'])))
  if len(imgs)==1: return imgs[0][1]
  # resize to same height and concatenate horizontally
  h=360; parts=[]
  for label,img in imgs:
   im=cv2.resize(img,(int(img.shape[1]*h/img.shape[0]),h)); cv2.putText(im,label,(10,30),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),2); parts.append(im)
  return np.concatenate(parts,axis=1)
 video_path=out/'replay.mp4'
 first=frame_from_obs(obs); H,W=first.shape[:2]
 writer=cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*'mp4v'), args.fps, (W,H))
 def write(obs): writer.write(cv2.cvtColor(frame_from_obs(obs), cv2.COLOR_RGB2BGR))
 def step_action(q7, grip, repeats=1):
  action=np.zeros((1,action_dim),np.float32); action[0,:7]=np.asarray(q7,np.float32); action[0,7]=float(grip)
  act=torch.tensor(action,device=env.device)
  last=None
  for _ in range(repeats):
   last=env.step(act)[0]; write(last)
  return last
 def command_q(target_q, grip):
  """Send target_q, optionally interpolating from current_q to avoid large target jumps."""
  global current_q
  target_q=np.asarray(target_q,dtype=np.float32)
  if args.max_joint_step and args.max_joint_step > 0:
   delta=target_q-current_q
   n=max(1,int(np.ceil(np.max(np.abs(delta))/args.max_joint_step)))
   last=None
   for i in range(1,n+1):
    q=current_q + delta*(i/n)
    last=step_action(q, grip, 1)
   current_q=target_q.copy()
   return last
  else:
   current_q=target_q.copy()
   return step_action(target_q, grip, 1)
 # Start explicitly from the plan/RoboLab home q_init and settle there before executing TiPToP motions.
 for _ in range(args.settle_steps): obs=step_action(current_q, gripper, 1)
 executed=0
 for si,s in enumerate(plan['steps']):
  if s['type']=='trajectory':
   pts=s.get('positions',[])
   idxs=list(range(0,len(pts),max(1,args.stride)))
   if pts and (len(pts)-1) not in idxs: idxs.append(len(pts)-1)
   for idx in idxs:
    obs=command_q(pts[idx], gripper); executed+=1
  elif s['type']=='gripper':
   gripper=1.0 if s.get('action')=='close' else 0.0
   q=plan['q_init']
   # hold current arm at last trajectory waypoint if available
   for prev in reversed(plan['steps'][:si]):
    if prev.get('positions'):
     q=prev['positions'][-1]; break
   # Move/hold at the current grasp/release arm pose while actuating gripper.
   command_q(q, gripper)
   for _ in range(args.gripper_steps): obs=step_action(current_q, gripper, 1); executed+=1
 for _ in range(args.post_steps): obs=step_action(current_q, gripper, 1)
 writer.release()
 results=env.get_env_results()
 subtask=get_all_env_subtask_infos(env)
 result={'task':args.task,'instruction':cfg.instruction,'success':bool(results[0].get('success',False)) if results else False,'env_results':results,'subtask_info':subtask,'video':str(video_path),'executed_steps':executed,'stride':args.stride,'max_joint_step':args.max_joint_step}
 (out/'result.json').write_text(json.dumps(result,indent=2,default=str))
 print(json.dumps(result,indent=2,default=str))
 env.close(); app.close()
except Exception as e:
 print('ERR',e); traceback.print_exc(); app.close(); sys.exit(1)
