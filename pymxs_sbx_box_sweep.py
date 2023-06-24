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
from dm_control.utils import rewards
# from numba import jit

import gymnasium as gym
import gym_mxs
import numpy as np
from scipy.spatial.transform import Rotation

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
  training_args.add_argument("-l", "--episode-length", help="Episode timestep limit", type=int, default=2000)
  training_args.add_argument("--algo", help="Algorithm to use", type=str, default="sac")
  training_args.add_argument("--env", help="Environment to use", type=str, default="gym_mxs/MXSBox2D-v0")
  training_args.add_argument("--training_render_mode", help="Render mode for training", type=str, default="ansi")
  training_args.add_argument("--frame_skip", help="Number of frames to skip per action", type=int, default=10)
  training_args.add_argument("--n_vec_env", help="Number of vectorized environments", type=int, default=1)
  training_args.add_argument("--config_dir", help="Directory to load obstacle configs from", type=str, default="./config/")
  
  hyperparam_args = parser.add_argument_group("Hyperparam options")
  hyperparam_args.add_argument("--gamma", help="Discount factor", type=float)
  hyperparam_args.add_argument("--learning_rate", help="Learning rate", type=float)
  hyperparam_args.add_argument("--batch_size", help="Batch size", type=int)
  hyperparam_args.add_argument("--buffer_size", help="Buffer size", type=int)
  hyperparam_args.add_argument("--tau", help="Soft update coefficient", type=float)
  hyperparam_args.add_argument("--update_interval", help="Update interval", type=int)
  hyperparam_args.add_argument("--top_quantiles_to_drop_per_net", help="Number of top quatiles to drop per net", type=int)
  hyperparam_args.add_argument("--gradient_steps", help="Number of gradient steps", type=int)
  hyperparam_args.add_argument("--qf_learning_rate", help="QF learning rate", type=float)
  hyperparam_args.add_argument("--learning_starts", help="Learning starts", type=int)
  hyperparam_args.add_argument("--dropout_rate", help="Dropout rate", type=float)
  hyperparam_args.add_argument("--net_arch", help="Network architecture", type=str)
  hyperparam_args.add_argument("--train_freq", help="Train frequency", type=int)
  hyperparam_args.add_argument("--clip_range", help="Clip range", type=float)
  hyperparam_args.add_argument("--max_grad_norm", help="Max gradient norm", type=float)
  hyperparam_args.add_argument("--n_epochs", help="Number of epochs", type=int)
  hyperparam_args.add_argument("--ent_coef", help="Entropy coefficient", type=float)
  hyperparam_args.add_argument("--n_steps", help="Number of steps", type=int)
  hyperparam_args.add_argument("--gae_lambda", help="GAE lambda", type=float)
  hyperparam_args.add_argument("--vf_coef", help="Value function coefficient", type=float)
  hyperparam_args.add_argument("--n_quantiles", help="Number of quantiles", type=int)

  config_args = parser.add_argument_group("Config options")
  config_args.add_argument("--dryden", help="Use Dryden turbulence model", type =str)
  config_args.add_argument("--steady", help="Use steady wind model", nargs='*', type =int)
  config_args.add_argument("--airspeed", help="Airspeed", nargs='*', type =float)

  args, unknown = parser.parse_known_args()

  hyperparam_args = {arg.dest: getattr(args, arg.dest) for arg in hyperparam_args._group_actions if getattr(args, arg.dest) is not None}
  print(hyperparam_args)
  config_args = {arg.dest: getattr(args, arg.dest) for arg in config_args._group_actions if getattr(args, arg.dest) is not None}
  if args.run_name is None:
    args.run_name = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")

  config_file = args.config_dir + "box_env_config.json"
  with open(config_file) as f:
    env_config = json.load(f)


  for key, value in config_args.items():
    value = None if value == "None" else value
    env_config["domain"][key] = value
  
  if args.directory is not None:
    if not os.path.exists(args.directory+"/wandb"):
      os.makedirs(args.directory+"/wandb")

  wandb.init(
    project="pymxs",
    sync_tensorboard=True,
    dir=args.directory,
    save_code=True,
    )


  def make_env():
    env = gym.make(args.env, training=True, config = env_config, render_mode=args.training_render_mode)
    # env = Monitor(env, args.directory)
    env= gym.wrappers.TimeLimit(env, max_episode_steps=2000)
    env = MaxAndSkipEnv(env, skip=args.frame_skip)
    return env

  if args.n_vec_env > 1:
    env = make_vec_env(lambda: make_env(), n_envs=args.n_vec_env)
  else:
    env = make_env()

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

  policy_kwargs = dict()

  if args.gradient_steps:
    hyperparam_args["policy_delay"] = int(args.gradient_steps)
    wandb.config.policy_delay = int(args.gradient_steps)

  # create dict of hyperparams from args
  if args.net_arch:
    net_arch = {
    "small": [64, 64],
    "medium": [256, 256],
    "big": [400, 300],
  }[args.net_arch]
    policy_kwargs["net_arch"] = net_arch
    del hyperparam_args["net_arch"]

  if MlAlg == PPO:
    if args.batch_size > args.n_steps:
      hyperparam_args["batch_size"] = args.n_steps

  if args.dropout_rate:
    policy_kwargs["dropout_rate"] = args.dropout_rate

  hyperparam_args["policy_kwargs"] = policy_kwargs
    
  pprint.pprint(hyperparam_args)
  
 
 
  run_dir = f"{args.directory}/{args.run_name}"

  # model = MlAlg("MlpPolicy", env, verbose=1, policy_kwargs=dict(net_arch=net_arch))
  model = MlAlg("MlpPolicy", env, verbose=0, tensorboard_log=f"{run_dir}/tensorboard/", **hyperparam_args)
  # model = MlAlg("MlpPolicy", env, verbose=1)
  model_save_freq = args.model_save_freq / args.frame_skip
  model.learn(total_timesteps=args.steps, callback=WandbCallback(model_save_path=run_dir, verbose=0, model_save_freq=model_save_freq))
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
    # copy config file to run dir
    # shutil.copy(config_file, f"{run_dir}/box_env_config.json")
    with open(f"{run_dir}/box_env_config.json", "w") as f:
      json.dump(env_config, f, indent=2)
    
    

  if args.output or args.plot:
    output_file = f"{run_dir}/output.csv"
    evaluate_model(model, env, output_file)
    
  if args.plot:
    subprocess.call(["python", f"{os.path.dirname(os.path.realpath(__file__))}/plotting/unified_plot.py", "-d", args.directory, args.run_name])
