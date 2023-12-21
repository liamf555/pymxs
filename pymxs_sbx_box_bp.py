import importlib
import pprint
import sys

# sys.path.insert(1, '/home/rc13011/projects/mxs/pymxs/models')
sys.path.insert(1, '/home/tu18537/dev/mxs/pymxs/models')
# sys.path.insert(1, '/app/models')
import copy
import datetime
import json
import os
import subprocess
import string
import random
import shutil
import wandb
from wandb.integration.sb3 import WandbCallback
from contextlib import nullcontext

# from stable_baselines3 import PPO as MlAlg
from sbx import TQC, PPO, SAC, DroQ, DQN 
import sbx
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.atari_wrappers import MaxAndSkipEnv
from stable_baselines3.common.callbacks import EvalCallback
from dm_control.utils import rewards
# from numba import jit

import gymnasium as gym
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
def evaluate_model(model, env, output_path=False, n_episodes=1):
    print(f"Output path: {output_path}")

    total_reward = 0
    total_steps = 0
#   with open(output_path, "w") if output_path else nullcontext() as outfile:
#     if outfile:
#       outfile.write("episode,time,x,y,z,u,v,w,qx,qy,qz,qw,p,q,r,alpha,airspeed,elevator,throttle\n")
    for episode in range(n_episodes):
        obs = env.reset()
        print(obs)
        done = False
        simtime = 0
        episode_reward = 0
        episode_steps = 0
        while not done:
            action, _state = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            # if outfile:
            #     print(f"calling render: {env.render()}")
            #     outfile.write(f"{episode},{simtime},{env.render()[1:-1]}\n")
            simtime += env.dT
            episode_reward += reward
            episode_steps += 1
            total_reward += episode_reward
            total_steps += episode_steps

    avg_reward = total_reward / n_episodes
    avg_steps = total_steps / n_episodes
    return avg_reward, avg_steps


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
  output_args.add_argument("--model_save_freq", help="Frequency to save model", type=int, default=100_000)
  
  training_args = parser.add_argument_group("Training options")
  training_args.add_argument("-s", "--steps", help="Total timesteps to train for", type=int, default=1_000_000)
  training_args.add_argument("-l", "--episode-length", help="Episode timestep limit", type=int, default=DEFAULT_TIME_LIMIT)
  training_args.add_argument("--algo", help="Algorithm to use", type=str, default="sac")
  training_args.add_argument("--env", help="Environment to use", type=str, default="gym_mxs/MXSBox2D-v0")
  training_args.add_argument("--training_render_mode", help="Render mode for training", type=str, default="ansi")
  training_args.add_argument("--frame_skip", help="Number of frames to skip per action", type=int, default=1)
  training_args.add_argument("--n_vec_env", help="Number of vectorized environments", type=int, default=1)
  training_args.add_argument("--config_dir", help="Directory to load obstacle configs from", type=str, default="./config/")
  
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
  hyperparam_args.add_argument("--train_freq", help="Train frequency", type=int)

  config_args = parser.add_argument_group("Config options")
  config_args.add_argument("--dryden", help="Use Dryden turbulence model", type =str)
  config_args.add_argument("--steady", help="Use steady wind model", nargs='*', type =int)
  config_args.add_argument("--airspeed", help="Airspeed", nargs='*', type =float)
 
  args, unknown = parser.parse_known_args()

  hyperparam_args = {arg.dest: getattr(args, arg.dest) for arg in hyperparam_args._group_actions if getattr(args, arg.dest) is not None}
  config_args = {arg.dest: getattr(args, arg.dest) for arg in config_args._group_actions if getattr(args, arg.dest) is not None}

  if args.run_name is None:
    # args.run_name = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    random_str = ''.join(random.choices(string.ascii_uppercase + string.digits, k=5))
    args.run_name = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S-%f") + "-" + random_str

  if args.env == "gym_mxs/MXSBox2DLand-v0":
    config_file = args.config_dir + "box_env_config_land.json"
    eval_config_file = args.config_dir + "box_env_eval_config_land.json"
  else:
    config_file = args.config_dir + "box_env_config.json"
    eval_config_file = args.config_dir + "box_env_eval_config.json"

  with open(config_file) as f:
    env_config = json.load(f)

  with open(eval_config_file) as f:
    eval_env_config = json.load(f)

  # print(env_config)
  
  for key, value in config_args.items():
    value = None if value == "None" else value
    env_config["domain"][key] = value

  # print(env_config)

  if args.directory is not None:
    if not os.path.exists(args.directory+"/wandb"):
      os.makedirs(args.directory+"/wandb")

  wandb.init(
    project="pymxs",
    sync_tensorboard=True,
    dir=args.directory,
    save_code=True,
    )

  def make_env(training = True, config=env_config, render_mode=args.training_render_mode):
    env = gym.make(args.env, training=True, config = config, render_mode=render_mode)
    # env = Monitor(env, args.directory)
    env= gym.wrappers.TimeLimit(env, max_episode_steps=2000)
    env = MaxAndSkipEnv(env, skip=args.frame_skip)
    return env

  if args.n_vec_env > 1:
    env = make_vec_env(lambda: make_env(), n_envs=args.n_vec_env)
  else:
    env = make_env()

  eval_env = make_vec_env(lambda: make_env(training=False, config=eval_env_config, render_mode=None), n_envs=10)

  run_dir = f"{args.directory}/{args.run_name}"
  eval_log_dir = f"{run_dir}/eval_logs"
  os.makedirs(eval_log_dir, exist_ok=True)

  eval_callback = EvalCallback(
    eval_env, best_model_save_path=eval_log_dir, log_path=eval_log_dir,
    eval_freq=max(5000 // args.n_vec_env, 1), n_eval_episodes= 50, deterministic=True, render=False)

  # env = make_vec_env(lambda: make_env(), n_envs=12)

  wandb.config.update(args)

  def flatten_dict(d, parent_dict=None):
    if parent_dict is None:
        parent_dict = {}
    for k, v in d.items():
        if isinstance(v, dict):
            flatten_dict(v, parent_dict)
        else:
            parent_dict[k] = v
    return parent_dict

  env_config_flatten = flatten_dict(env_config)
  wandb.config.update(env_config_flatten, allow_val_change=True)

  if args.gradient_steps:
    hyperparam_args["policy_delay"] = int(args.gradient_steps)
    wandb.config.policy_delay = int(args.gradient_steps)

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
  pprint.pprint(wandb.config)
  
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
 
  # model = MlAlg("MlpPolicy", env, verbose=1, policy_kwargs=dict(net_arch=net_arch))
  model = MlAlg("MlpPolicy", env, verbose=1, tensorboard_log=f"{run_dir}/tensorboard/", **hyperparam_args)
  # model = MlAlg("MlpPolicy", env, verbose=1)
  model_save_freq = args.model_save_freq / args.frame_skip
  model.learn(total_timesteps=args.steps, callback=[WandbCallback(model_save_path=run_dir, verbose=1, model_save_freq=model_save_freq), eval_callback])
  # model.learn(total_timesteps=args.steps)

  if args.eval:
    mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=10)
    print(f"mean_reward:{mean_reward:.2f} +/- {std_reward}")
    wandb.log({"eval_reward": mean_reward})
  
  if args.save:
    # os.makedirs(run_dir, )
    model.save(f"{run_dir}/final_model.zip")
    wandb.save(f"{run_dir}/final_model.zip")
    with open(f"{run_dir}/metadata.json", "w") as f:
      json.dump(vars(args), f, indent=2)
    wandb.save(f"{run_dir}/metadata.json")
    with open(f"{run_dir}/box_env_config.json", "w") as f:
      json.dump(env_config, f, indent=2)
    # copy config file to run dir
    # shutil.copy(config_file, f"{run_dir}/box_env_config.json")
    
    

#   if args.output or args.plot:
#     output_file = f"{run_dir}/output.csv"
#     evaluate_model(model, model.get_env(), output_file)
    
  if args.plot:
    subprocess.call(["python", f"{os.path.dirname(os.path.realpath(__file__))}/plotting/unified_plot.py", "-d", args.directory, args.run_name])
