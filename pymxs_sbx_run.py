import sys
# sys.path.insert(1, '/home/rc13011/projects/mxs/pymxs/models')
# sys.path.insert(1, '/home/tu18537/dev/mxs/pymxs/models')
sys.path.insert(1, '/app/models')
import copy
import datetime
import json
import os
import subprocess
import wandb
from wandb.integration.sb3 import WandbCallback
from contextlib import nullcontext

# from stable_baselines3 import PPO as MlAlg
from sbx import TQC as MlAlg
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from dm_control.utils import rewards
# from numba import jit

import gym
import gym_mxs
import numpy as np
from scipy.spatial.transform import Rotation

DEFAULT_X_LIMIT = 30 # m
DEFAULT_U_LIMIT = 25 # m/s
DEFAULT_TIME_LIMIT = 1000 # s*100
DEFAULT_CLIMB_WEIGHT = 1
DEFAULT_PITCH_WEIGHT = 0
TARGET_PITCH = 90 # deg

def get_pitch(qx,qy,qz,qw):
  if True:
    # Can't use scipy if 'jit'ting
    rot = Rotation.from_quat(np.array([qx,qy,qz,qw]).T)
    [yaw, pitch, roll] = rot.as_euler('zyx', degrees=False)
    # print([yaw,pitch,roll])
    if yaw != 0:
      if pitch > 0:
        pitch = np.pi/2 + (np.pi/2 - pitch)
      else:
        pitch = -np.pi/2 + (-np.pi/2 - pitch)
  if False:
    sinp = np.sqrt(1 + 2 * (qw*qy - qx*qz))
    cosp = np.sqrt(1 - 2 * (qw*qy - qx*qz))
    pitch = 2 * np.arctan2(sinp, cosp) - np.pi / 2
  return pitch

def create_reward_func(args):
  # Split these out so numba can jit the reward function

  def prop_hang_func(obs, reward_state):
    [x,y,z, u,v,w, qx,qy,qz,qw, p,q,r] = obs
    pitch = get_pitch(qx,qy,qz,qw)
    # intitial_position = env.initial_state[0]
    x_error = x 
    z_error = z
    pitch_error = np.degrees(pitch) - TARGET_PITCH
    # print(f"z_error: {z_error}, pitch_error: {pitch_error}, x_error: {x_error}")
    pitch_r = rewards.tolerance(pitch_error, bounds = (-5, 5), margin = 90)
    alt_r = rewards.tolerance(z_error, bounds = (-2.5, 2.5), margin = 15)
    x_r = rewards.tolerance(x_error, bounds = (-2.5, 2.5), margin =20.0)

    reward = (pitch_r * 0.4)  + (alt_r * 0.4) + (x_r * 0.2)

    return reward, False, None

  return prop_hang_func
 

# TODO rewrite to use bult in SB methods
def evaluate_model(model, env, output_path=False):
  obs = env.reset()
  done = False
  simtime = 0
  with open(output_path, "w") if output_path else nullcontext() as outfile:
    if outfile:
      outfile.write("time,x,y,z,u,v,w,qx,qy,qz,qw,p,q,r,alpha,airspeed,elevator,throttle\n")
    while not done:
      action, _state = model.predict(obs, deterministic=True)
      obs, reward, done, info = env.step(action)
      if outfile:
        outfile.write(f"{simtime},{env.render('ansi')[1:-1]}\n")
      simtime += env.dT

  return obs, reward, done, info, simtime

class LongitudinalStateWrapper(gym.ObservationWrapper):
  def __init__(self, env) -> None:
    super().__init__(env)
    self.observation_space = gym.spaces.Box(
            low=np.float32(-np.inf),
            high=np.float32(np.inf),
            shape=(7,),
            dtype=np.float32
        )

  def observation(self, obs):
    #           x       z       u       w      qy      qw        q
    return [obs[0], obs[2], obs[3], obs[5], obs[7], obs[9], obs[11]]


if __name__ == "__main__":
  import argparse

  parser = argparse.ArgumentParser()
  parser.add_argument("-n", "--run_name", help="Name for run. If not set, current timestamp will be used")

  output_args = parser.add_argument_group("Output options")
  output_args.add_argument("--no-save", dest="save", action="store_false", help="Don't save this run")
  output_args.add_argument("-d", "--directory", default="./runs", help="Destination for saving runs")
  output_args.add_argument("-o", "--output", action="store_true", help="Generate CSV for final output")
  output_args.add_argument("--plot", action="store_true", help="Show plots at end of training. (Will generate CSV as if -o specified)")
  output_args.add_argument("--ignore-dirty", action="store_true", help="Ignore dirty tree when saving run")

  training_args = parser.add_argument_group("Training options")
  training_args.add_argument("-s", "--steps", help="Total timesteps to train for", type=int, default=500_000)
  training_args.add_argument("-l", "--episode-length", help="Episode timestep limit", type=int, default=DEFAULT_TIME_LIMIT)
  training_args.add_argument("--use-reduced-observation", help="Use only longitudinal state observations", action="store_true")
  
  network_args = parser.add_argument_group("Network options")
  network_args.add_argument("--depth", help="Number of layers in network", type=int, default=2)
  network_args.add_argument("--width", help="Width of layers in network", type=int, default=64)

  reward_args = parser.add_argument_group("Reward function options")
  reward_args.add_argument("-x", "--x-limit", help="x coordinate limit", type=float, default=DEFAULT_X_LIMIT)
  reward_args.add_argument("-u", "--u-limit", help="u velocity limit", type=float, default=DEFAULT_U_LIMIT)
  reward_args.add_argument("-c", "--climb-weight", help="Weight for climb cost", type=float, default=DEFAULT_CLIMB_WEIGHT)
  reward_args.add_argument("-p", "--pitch-weight", help="Weight for pitch cost", type=float, default=DEFAULT_PITCH_WEIGHT)
  reward_args.add_argument("-w", "--waypoint-weight", help="Weight for waypoints", type=float, default=0)
  reward_args.add_argument("-f", "--waypoint-file", help="File for waypoints", default=0)
  reward_args.add_argument("-m", "--manoeuvre", help="Manoeuvre to use", type=str)
  reward_args.add_argument("--multi-manoeuvre", help="Train for multiple manoeuvres at once", action="store_true")

  args = parser.parse_args()
  
  if args.run_name is None:
    args.run_name = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")

  # Check if on clean commit
  # diff_result = subprocess.run(["git", "diff", "-s", "--exit-code", "HEAD"])
  # git_sha = subprocess.check_output(["git", "rev-parse", "HEAD"], encoding="UTF-8")
  # git_sha = git_sha.strip()
  # if diff_result.returncode == 0:
  #   args.commit = git_sha
  # else:
  #   if args.ignore_dirty or not args.save:
  #     args.commit = f"{git_sha}-dirty"
  #   else:
  #     print("Error: Current tree not committed.")
  #     print("Prevent saving with --no-save, or explicitly ignore dirty tree with --ignore-dirty")
  #     sys.exit(1)

  wandb.init(
    project="pymxs",
    sync_tensorboard=True,
    )

  reward_func = create_reward_func(args)

  n_envs = 8
  vec_env_cls = SubprocVecEnv
  wandb.config.update({"n_envs": n_envs})

  def make_env():
    env = gym.make('gym_mxs/MXS-v0', reward_func=reward_func, timestep_limit=1000)
    env= gym.wrappers.TimeLimit(env, max_episode_steps=1000)
    if args.use_reduced_observation:
      env = LongitudinalStateWrapper(env)
    return env
  


  env = make_vec_env(lambda: make_env(), n_envs=n_envs,vec_env_cls=vec_env_cls)
  # env = make_vec_env(env_id, n_envs=num_cpu, seed=0, vec_env_cls=SubprocVecEnv
  if args.use_reduced_observation:
    env = LongitudinalStateWrapper(env) # turn on


# not required 
  # if args.multi_manoeuvre:
  #   env = MultiManoeuvreWrapper(
  #     env,
  #     ["hover", "descent"],
  #     create_reward_func,
  #     args
  #   )

  net_arch = "small"
  net_arch = {
    "small": [64, 64],
    "medium": [256, 256],
    "large": [400, 300],
  }[net_arch]
  hyperparameter_defaults = {"learning_rate": 0.003}
  policy_kwargs = dict(net_arch=net_arch)
  print("policy_kwargs", policy_kwargs)
  hyperparameter_defaults["policy_kwargs"] = policy_kwargs


  # layers = [args.width] * args.depth does not work with sbx
  # net_arch = [dict(vf=layers, pi=layers)]
  # model = MlAlg("MlpPolicy", env, verbose=1, policy_kwargs=dict(net_arch=net_arch))
  model = MlAlg("MlpPolicy", env, verbose=1, tensorboard_log=f"./runs/{args.run_name}/tensorboard/", **hyperparameter_defaults)
  # model = MlAlg("MlpPolicy", env, verbose=1)
  model.learn(total_timesteps=args.steps, callback=WandbCallback())
  # model.learn(total_timesteps=args.steps)

  run_dir = f"{args.directory}/{args.run_name}"
  if args.save:
    os.makedirs(run_dir)
    model.save(f"{run_dir}/model.zip")
    with open(f"{run_dir}/metadata.json", "w") as f:
      json.dump(vars(args), f, indent=2)

  if args.output or args.plot:
    output_file = f"{run_dir}/output.csv"
    evaluate_model(model, env, output_file)
    
  if args.plot:
    subprocess.call(["python", f"{os.path.dirname(os.path.realpath(__file__))}/plotting/unified_plot.py", "-d", args.directory, args.run_name])
