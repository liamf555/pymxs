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
import imageio
import subprocess
import wandb
import argparse
from argparse import Namespace
from wandb.integration.sb3 import WandbCallback
from contextlib import nullcontext

# from stable_baselines3 import PPO as MlAlg
from sbx import TQC, PPO, SAC, DroQ, DQN 
import sbx
from stable_baselines3.common.env_util import make_vec_env
# from stable_baselines3.common.vec_env import VecFrameStack
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecFrameStack, VecNormalize
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.utils import set_random_seed
from dm_control.utils import rewards
# from numba import jit

import gymnasium as gym
from pathlib import Path
import gym_mxs
import numpy as np
from scipy.spatial.transform import Rotation


parser = argparse.ArgumentParser(description='Load trained model and plot results')
parser.add_argument('--dir_path', type=Path)
parser.add_argument('--render_mode', type = str, default='ansi')
args= parser.parse_args()



dir_path = args.dir_path
json_path = dir_path / 'metadata.json'
config_file = dir_path / "box_env_config.json"

# load json file with parameters
with open(json_path) as json_file:
    params = json.load(json_file)

with open(config_file) as f:
    env_config = json.load(f)

metadata = Namespace(**params)

algorithm = metadata.algo

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

print(f"Config: {metadata}")
print(f"Config: {env_config}")

env = gym.make(metadata.env, training=False, config = env_config, render_mode=args.render_mode)

model = MlAlg.load(dir_path / 'model.zip', env=env)

episode_rewards, episode_lengths = evaluate_policy(model, env, n_eval_episodes=10, deterministic=True, render=True, return_episode_rewards=True)


print(f"mean_reward:{np.mean(episode_rewards):.2f} +/- {np.std(episode_rewards)}")
print(f"mean_length:{np.mean(episode_lengths):.2f} +/- {np.std(episode_lengths)}")

# from stable_baselines3.common.vec_env import VecVideoRecorder, DummyVecEnv

# video_folder = "logs/videos/"
# video_length = 100

# vec_env = DummyVecEnv([lambda: gym.make(metadata.env, training=False, render_mode= "rgb_array")])

# obs = vec_env.reset()

# # Record the video starting at the first step
# vec_env = VecVideoRecorder(vec_env, video_folder,
#                        record_video_trigger=lambda x: x == 0, video_length=video_length,
#                        name_prefix=f"trained-agent-{metadata.env}")

# vec_env.reset()
# for _ in range(video_length + 1):
#   action = [vec_env.action_space.sample()]
#   obs, _, _, _ = vec_env.step(action)
# # Save the video
# vec_env.close()