import sys
# sys.path.insert(1, '/home/rc13011/projects/mxs/pymxs/models')
sys.path.insert(1, '/home/tu18537/dev/mxs/pymxs/models')
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
  # @jit
  def descent_reward_func(obs, max_z):
    [x,y,z, u,v,w, qx,qy,qz,qw, p,q,r] = obs

    if max_z == None:
      max_z = 0
    # Update max_z
    max_z = max(z, max_z)

    pitch = get_pitch(qx,qy,qz,qw)

    if x < 0 or u > u_limit or u < 0 or pitch > np.radians(89) or pitch < np.radians(-270):
      return -1000, True, max_z

    if x > x_limit:
      total_ke_sqd = u**2+v**2+w**2
      ke_fraction = total_ke_sqd / (15**2)

      climb = max_z - z

      pitch_error = abs(pitch)

      return (1 - pitch_weight*pitch_error) * z / (ke_fraction * (1+climb*climb_weight)), True, max_z

    reward = 0
    if waypoint_weight != 0:
      for [wp_x,wp_z] in waypoints:
        if wp_x < x:
          continue
        reward += waypoint_weight / (np.hypot(x-wp_x, z-wp_z) + 0.01)

    return reward, False, max_z

  def within(value, lower, upper):
    if value < lower:
      return False
    if value > upper:
      return False
    return True

  def hover_reward_func(obs, reward_state):
    if reward_state is None:
      reward_state = 0
    reward_state += 1

    [x,y,z, u,v,w, qx,qy,qz,qw, p,q,r] = obs
    pitch = get_pitch(qx,qy,qz,qw)

    is_hover = within(q, -0.01, 0.01) \
      and within(pitch, np.radians(85), np.radians(95)) \
      and within(u, -0.1, 0.1) \
      and within(w, -0.1, 0.1) \

    if is_hover:
      # Reward is based on hover position
      return 110 + z, True, reward_state

    q_progress = 1 / (1 + abs(q))
    pitch_progress = 1 / (1+abs(np.radians(90) - pitch))
    u_progress = 1 / (1 + abs(u))
    w_progress = 1 / (1 + abs(w))
    hover_progress = q_progress * pitch_progress * u_progress * w_progress

    if reward_state >= 250:
      return 100 * hover_progress, True, None

    return 0, False, reward_state
  
  def prop_hang_func(obs, reward_state):
    [x,y,z, u,v,w, qx,qy,qz,qw, p,q,r] = obs
    pitch = get_pitch(qx,qy,qz,qw)
    intitial_position = env.initial_state[0]
    x_error = x - intitial_position[0]
    z_error = z - intitial_position[2]
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
  training_args.add_argument("-s", "--steps", help="Total timesteps to train for", type=int, default=250_000)
  training_args.add_argument("-l", "--episode-length", help="Episode timestep limit", type=int, default=DEFAULT_TIME_LIMIT)
  training_args.add_argument("--use-reduced-observation", help="Use only longitudinal state observations", action="store_true")
  
  network_args = parser.add_argument_group("Network options")
  network_args.add_argument("--depth", help="Number of layers in network", type=int, default=2)
  network_args.add_argument("--width", help="Width of layers in network", type=int, default=64)

  sweep_args = parser.add_argument_group("Sweep options")
  sweep_args.add_argument("--gamma", help="Discount factor", type=float, default=0.99)
  sweep_args.add_argument("--learning_rate", help="Learning rate", type=float, default=0.00025)
  sweep_args.add_argument("--batch_size", help="Batch size", type=int, default=64)
  # sweep_args.add_argument("--buffer-size", help="Buffer size", type=int, default=1_000_000)
  sweep_args.add_argument("--tau", help="Soft update coefficient", type=float, default=0.005)
  # sweep_args.add_argument("--update-interval", help="Update interval", type=int, default=4)
  sweep_args.add_argument("--top_quantiles_to_drop_per_net", help="Number of top quatiles to drop per net", type=int, default=0)
  sweep_args.add_argument("--gradient_steps", help="Number of gradient steps", type=int, default=1)
  sweep_args.add_argument("--qf_learning_rate", help="QF learning rate", type=float, default=0.00025)
  sweep_args.add_argument("--learning_starts", help="Learning starts", type=int, default=1000)
  sweep_args.add_argument("--dropout_rate", help="Dropout rate", type=float, default=0.0)
  sweep_args.add_argument("--net_arch", help="Network architecture", type=str, default="medium")
  args = parser.parse_args()
  
  if args.run_name is None:
    args.run_name = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")

  # Attempt to load any waypoint file
  wandb.init(
    project="pymxs",
    sync_tensorboard=True,
    )

  reward_func = create_reward_func(args)

  env = gym.make('gym_mxs/MXS-v0', reward_func=reward_func, timestep_limit=1000)
  env= gym.wrappers.TimeLimit(env, max_episode_steps=1000)
  if args.use_reduced_observation:
    env = LongitudinalStateWrapper(env) # turn on


  # n_envs = 12
  # wandb.config.update({"n_envs": n_envs})

  # def make_env():
  #   env = gym.make('gym_mxs/MXS-v0', reward_func=reward_func, timestep_limit=1000)
  #   env= gym.wrappers.TimeLimit(env, max_episode_steps=1000)
  #   if args.use_reduced_observation:
  #     env = LongitudinalStateWrapper(env)
  #   return env
  


  # env = make_vec_env(lambda: make_env(), n_envs=n_envs)

  hyperparameter_defaults = dict(
    gamma=args.gamma,
    tau=args.tau,
    gradient_steps = args.gradient_steps,
    learning_rate=args.learning_rate,
    # qf_learning_rate=args.qf_learning_rate,
    batch_size=args.batch_size,
    learning_starts=args.learning_starts,
    top_quantiles_to_drop_per_net=args.top_quantiles_to_drop_per_net,
    # dropout_rate=args.dropout_rate,
    )
  
  # wandb.config.update(hyperparameter_defaults)

  policy_delay = int(hyperparameter_defaults["gradient_steps"])

  net_arch = args.net_arch

  net_arch = {
    "small": [64, 64],
    "medium": [256, 256],
    "large": [400, 300],
  }[net_arch]
  
  policy_kwargs = dict(net_arch=net_arch)
  print("policy_kwargs", policy_kwargs)
  hyperparameter_defaults["policy_delay"] = policy_delay
  hyperparameter_defaults["policy_kwargs"] = policy_kwargs
  wandb.config.update(hyperparameter_defaults)

  # layers = [args.width] * args.depth does not work with sbx
  # net_arch = [dict(vf=layers, pi=layers)]c
  # model = MlAlg("MlpPolicy", env, verbose=1, policy_kwargs=dict(net_arch=net_arch))
  model = MlAlg("MlpPolicy", env, verbose=1, tensorboard_log=f"./runs/{args.run_name}/tensorboard/", **hyperparameter_defaults)
  model.learn(total_timesteps=args.steps, callback=WandbCallback())

  # run_dir = f"{args.directory}/{args.run_name}"
  # if args.save:
  #   os.makedirs(run_dir)
  #   model.save(f"{run_dir}/model.zip")
  #   with open(f"{run_dir}/metadata.json", "w") as f:
  #     json.dump(vars(args), f, indent=2)

  # if args.output or args.plot:
  #   output_file = f"{run_dir}/output.csv"
  #   evaluate_model(model, env, output_file)
    
  # if args.plot:
    # subprocess.call(["python", f"{os.path.dirname(os.path.realpath(__file__))}/plotting/unified_plot.py", "-d", args.directory, args.run_name])
