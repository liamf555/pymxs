import importlib
import pprint
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
from sbx import TQC, PPO, SAC, DroQ, DQN 
import sbx
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
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

  def obstacle_avoidance_func(obs, reward_state):
    pass
   
 

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
  output_args.add_argument("--eval", action="store_true", help="Evaluate model after training")
  
  training_args = parser.add_argument_group("Training options")
  training_args.add_argument("-s", "--steps", help="Total timesteps to train for", type=int, default=10_000)
  training_args.add_argument("-l", "--episode-length", help="Episode timestep limit", type=int, default=DEFAULT_TIME_LIMIT)
  training_args.add_argument("--use-reduced-observation", help="Use only longitudinal state observations", action="store_true")
  training_args.add_argument("--algo", help="Algorithm to use", type=str, default="sac")
  
  reward_args = parser.add_argument_group("Reward function options")
  reward_args.add_argument("-m", "--manoeuvre", help="Manoeuvre to use", type=str)

  hyperparam_args = parser.add_argument_group("Hyperparam options")
  hyperparam_args.add_argument("--gamma", help="Discount factor", type=float)
  hyperparam_args.add_argument("--learning_rate", help="Learning rate", type=float)
  hyperparam_args.add_argument("--batch_size", help="Batch size", type=int)
  hyperparam_args.add_argument("--buffer_size", help="Buffer size", type=int)
  hyperparam_args.add_argument("--tau", help="Soft update coefficient", type=float)
  hyperparam_args.add_argument("--update-interval", help="Update interval", type=int)
  hyperparam_args.add_argument("--top_quantiles_to_drop_per_net", help="Number of top quatiles to drop per net", type=int)
  hyperparam_args.add_argument("--gradient_steps", help="Number of gradient steps", type=int)
  hyperparam_args.add_argument("--qf_learning_rate", help="QF learning rate", type=float)
  hyperparam_args.add_argument("--learning_starts", help="Learning starts", type=int)
  hyperparam_args.add_argument("--dropout_rate", help="Dropout rate", type=float)
  hyperparam_args.add_argument("--net_arch", help="Network architecture", type=str)
  
  args, unknown = parser.parse_known_args()

  hyperparam_args = {arg.dest: getattr(args, arg.dest) for arg in hyperparam_args._group_actions if getattr(args, arg.dest) is not None}

  if args.run_name is None:
    args.run_name = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")

  if args.directory is not None:
    if not os.path.exists(args.directory+"/wandb"):
      os.makedirs(args.directory+"/wandb")

  wandb.init(
    project="pymxs",
    sync_tensorboard=True,
    dir=args.directory,
    )
  reward_func = create_reward_func(args)

  env = gym.make('gym_mxs/MXS-v0', reward_func=reward_func, timestep_limit=1000)
  env= gym.wrappers.TimeLimit(env, max_episode_steps=1000)
  if args.use_reduced_observation:
    env = LongitudinalStateWrapper(env)

  if args.gradient_steps:
    hyperparam_args["policy_delay"] = int(args.gradient_steps)

  # create dict of hyperparams from args
  if args.net_arch:
    net_arch = {
    "small": [64, 64],
    "medium": [256, 256],
    "large": [400, 300],
  }[args.net_arch]
    policy_kwargs = dict(net_arch=net_arch)
    hyperparam_args["policy_kwargs"] = policy_kwargs
    
  pprint.pprint(hyperparam_args)
  
  ## get aglorithm from args and create arlgorithm object
  algorithm = args.algo

  def get_alg(algorithm):
    algos = {
      "sac": SAC,
      "tqc": TQC,
      "ppo": PPO,
      "droq": DroQ,
      "dqn": DQN
    }
    return algos[algorithm]
  
  MlAlg = get_alg(algorithm)
 
  run_dir = f"{args.directory}/{args.run_name}"

  # model = MlAlg("MlpPolicy", env, verbose=1, policy_kwargs=dict(net_arch=net_arch))
  model = MlAlg("MlpPolicy", env, verbose=1, tensorboard_log=f"{run_dir}/tensorboard/")
  # model = MlAlg("MlpPolicy", env, verbose=1)
  model.learn(total_timesteps=args.steps, callback=WandbCallback())
  # model.learn(total_timesteps=args.steps)

  if args.eval:
    mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=10)
    print(f"mean_reward:{mean_reward:.2f} +/- {std_reward}")
    wandb.log({"eval_reward": mean_reward})
  
  if args.save:
    # os.makedirs(run_dir, )
    model.save(f"{run_dir}/model.zip")
    wandb.save(f"{run_dir}/model.zip")
    with open(f"{run_dir}/metadata.json", "w") as f:
      json.dump(vars(args), f, indent=2)
    wandb.save(f"{run_dir}/metadata.json")
    

  if args.output or args.plot:
    output_file = f"{run_dir}/output.csv"
    evaluate_model(model, env, output_file)
    
  if args.plot:
    subprocess.call(["python", f"{os.path.dirname(os.path.realpath(__file__))}/plotting/unified_plot.py", "-d", args.directory, args.run_name])
