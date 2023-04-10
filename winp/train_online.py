import sys
# sys.path.insert(1, '/home/rc13011/projects/mxs/pymxs/models')
sys.path.insert(1, '/home/tu18537/dev/mxs/pymxs/models')
import copy
import datetime
import json
import os
import subprocess
from contextlib import nullcontext

from dm_control.utils import rewards
# from numba import jit

import os
import pickle
import shutil
import time

import numpy as np
import tqdm

import gym
import gym_mxs
import wandb
from absl import app, flags
from flax.training import checkpoints
from ml_collections import config_flags
from rl.agents import SACLearner
from rl.data import ReplayBuffer
from rl.evaluation import evaluate
from rl.wrappers import wrap_gym
import abc

import gym
import gym_mxs

import numpy as np
from scipy.spatial.transform import Rotation

DEFAULT_X_LIMIT = 50 # m
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

def create_reward_func():
  # Split these out so numba can jit the reward function

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
   
    x_error = x 
    z_error = z
    pitch_error = np.degrees(pitch) - TARGET_PITCH
    # print(f"z_error: {z_error}, pitch_error: {pitch_error}, x_error: {x_error}")

    pitch_r = rewards.tolerance(pitch_error, bounds = (-5, 5), margin = 90)
    alt_r = rewards.tolerance(z_error, bounds = (-2.5, 2.5), margin = 15)
    x_r = rewards.tolerance(x_error, bounds = (-2.5, 2.5), margin =20.0)

    reward = (pitch_r * 0.4)  + (alt_r * 0.4) + (x_r * 0.2)

    return reward, False, None

  manoeuvre = "hang"


  if manoeuvre == "hover":
    return hover_reward_func
  elif manoeuvre == "hang":
    return prop_hang_func
  

#! /usr/bin/env python

log_interval = 10000
eval_interval = 10000
start_training = 10000


FLAGS = flags.FLAGS

flags.DEFINE_string('env_name', 'Ardupilot-v0', 'Environment name.')
flags.DEFINE_string('save_dir', './tmp/', 'Tensorboard logging dir.')
flags.DEFINE_integer('seed', 42, 'Random seed.')
flags.DEFINE_integer('eval_episodes', 1,
                     'Number of episodes used for evaluation.')
flags.DEFINE_integer('log_interval', 1000, 'Logging interval.')
flags.DEFINE_integer('eval_interval', 1000, 'Eval interval.')
flags.DEFINE_integer('batch_size', 256, 'Mini batch size.')
flags.DEFINE_integer('max_steps', int(1e6), 'Number of training steps.')
flags.DEFINE_integer('start_training', int(1000),
                     'Number of training steps to start training.')
flags.DEFINE_boolean('tqdm', True, 'Use tqdm progress bar.')
flags.DEFINE_boolean('wandb', False, 'Log wandb.')
flags.DEFINE_boolean('save_video', False, 'Save videos during evaluation.')
flags.DEFINE_float('action_filter_high_cut', None, 'Action filter high cut.')
flags.DEFINE_integer('action_history', 1, 'Action history.')
flags.DEFINE_integer('control_frequency', 20, 'Control frequency.')
flags.DEFINE_integer('utd_ratio', 1, 'Update to data ratio.')
flags.DEFINE_string('task', 'knife', 'Task to train on')
flags.DEFINE_boolean('real_robot', False, 'Use real robot.')
config_flags.DEFINE_config_file(
    'config',
    'configs/sac_config.py',
    'File path to the training hyperparameter configuration.',
    lock_config=False)

class Agent:
    """Abstract class for all agents."""

    @abc.abstractmethod
    def act(self, obs: np.ndarray, **_kwargs) -> np.ndarray:
        """Issues an action given an observation.

        Args:
            obs (np.ndarray): the observation for which the action is needed.

        Returns:
            (np.ndarray): the action.
        """
        pass

    def plan(self, obs: np.ndarray, **_kwargs) -> np.ndarray:
        """Issues a sequence of actions given an observation.

        Unless overridden by a child class, this will be equivalent to :meth:`act`.

        Args:
            obs (np.ndarray): the observation for which the sequence is needed.

        Returns:
            (np.ndarray): a sequence of actions.
        """

        return self.act(obs, **_kwargs)

    def reset(self):
        """Resets any internal state of the agent."""
        pass
class AcroAgent(Agent):
    """An agent that performs aerobatic actions to populate the replay buffer

    Args:
        env (gym.Env): the environment on which the agent will act.

    """
    def __init__(self, env: gym.Env):
        self.env = env
        

    def act(self, *_args, **_kwargs) -> np.ndarray:
        return self.env.manoeuvre_instance.do_manoeuvre()

def main(_):
    wandb.init(project='pymxs')
    wandb.config.update(FLAGS)

    reward_func = create_reward_func()

    env = gym.make('gym_mxs/MXS-v0', reward_func=reward_func, timestep_limit=1000)

    env = gym.wrappers.TimeLimit(env, 1000)
    env = wrap_gym(env, rescale_actions=True)
    
    env = gym.wrappers.RecordEpisodeStatistics(env, deque_size=1)
    # print(vars(env))
    # env = gym.wrappers.RecordVideo(
    #     env,
    #     f'videos/train_{FLAGS.action_filter_high_cut}',
    #     episode_trigger=lambda x: True)
    env.seed(FLAGS.seed)

    # if not FLAGS.real_robot:
    #     eval_env = make_mujoco_env(
    #         FLAGS.env_name,
    #         control_frequency=FLAGS.control_frequency,
    #         action_filter_high_cut=FLAGS.action_filter_high_cut,
    #         action_history=FLAGS.action_history)
    #     eval_env = wrap_gym(eval_env, rescale_actions=True)
    #     eval_env = gym.wrappers.RecordVideo(
    #         eval_env,
    #         f'videos/eval_{FLAGS.action_filter_high_cut}',
    #         episode_trigger=lambda x: True)
    #     eval_env.seed(FLAGS.seed + 42)

    kwargs = dict(FLAGS.config)
    agent = SACLearner.create(FLAGS.seed, env.observation_space,
                              env.action_space, **kwargs)

    chkpt_dir = 'saved/checkpoints'
    os.makedirs(chkpt_dir, exist_ok=True)
    buffer_dir = 'saved/buffers'

    last_checkpoint = checkpoints.latest_checkpoint(chkpt_dir)

    if last_checkpoint is None:
        start_i = 0
        replay_buffer = ReplayBuffer(env.observation_space, env.action_space,
                                     FLAGS.max_steps)
        replay_buffer.seed(FLAGS.seed)
    else:
        start_i = int(last_checkpoint.split('_')[-1])

        agent = checkpoints.restore_checkpoint(last_checkpoint, agent)

        with open(os.path.join(buffer_dir, f'buffer_{start_i}'), 'rb') as f:
            replay_buffer = pickle.load(f)

    previous_time = 0
    # total_alt_error = 0

    # total_pitch_error = 0
    # total_x_error = 0
    # total_y_error = 0

    # total_obs_vel_x =0
    # total_obs_vel_y =0
    # total_obs_vel_z =0
    # total_obs_p =0
    # total_obs_q =0
    # total_obs_r =0
    
    ep_steps = 0
    reset_flag = True

    # observation, done = env.reset(), False
    for i in tqdm.tqdm(range(start_i, FLAGS.max_steps),
                       smoothing=0.1,
                       disable=not FLAGS.tqdm):
        # print(i)
        if reset_flag:
            observation, done = env.reset(), False
            reset_flag = False
            # print("reset")
        start_time = time.perf_counter()
        if i < start_training:
        # if env.acro_mode:
            action = env.action_space.sample()
            # print("Acro agent acting")
            # action = acro_agent.act()
        else:
            # print("Agent acting")
            # print(f" obs {observation}")
            action, agent = agent.sample_actions(observation)

        next_observation, reward, done, info = env.step(action)
 
        ep_steps += 1
    
        if not done or 'TimeLimit.truncated' in info:
            mask = 1.0
        else:
            mask = 0.0

        replay_buffer.insert(
            dict(observations=observation,
                 actions=action,
                 rewards=reward,
                 masks=mask,
                 dones=done,
                 next_observations=next_observation))
        observation = next_observation
        # print(observation)

        if done:
            # env.mavlink_mission_set_current()
            # env.increment_manoeuvre()
            # env.increment_waypoint()
            print("Done")
            reset_flag = True
            # observation, done = env.reset(), False
            # if not env.acro_mode:
            if i > start_training:
                print("Decoding")
                for k, v in info['episode'].items():
                    decode = {'r': 'episode_reward', 'l': 'length', 't': 'time'}
                    wandb.log({f'results/{decode[k]}': v}, step=i)
  
                ep_steps = 0
                
    

        if i >= start_training:
            # print("Agent training")
            # env.hold()
            # env.reset_action()
            batch = replay_buffer.sample(FLAGS.batch_size * FLAGS.utd_ratio)
            agent, update_info = agent.update(batch, FLAGS.utd_ratio)
            # print("Agent trained")

            if i % log_interval == 0:
                for k, v in update_info.items():
                    wandb.log({f'training/{k}': v}, step=i)

        if i % eval_interval == 0:
            # if not FLAGS.real_robot:
            #     eval_info = evaluate(agent,
            #                          eval_env,
            #                          num_episodes=FLAGS.eval_episodes)
            #     for k, v in eval_info.items():
            #         wandb.log({f'evaluation/{k}': v}, step=i)

            checkpoints.save_checkpoint(chkpt_dir,
                                        agent,
                                        step=i + 1,
                                        keep=20,
                                        overwrite=True)

            try:
                shutil.rmtree(buffer_dir)
            except:
                pass

            os.makedirs(buffer_dir, exist_ok=True)
            with open(os.path.join(buffer_dir, f'buffer_{i+1}'), 'wb') as f:
                pickle.dump(replay_buffer, f)

if __name__ == '__main__':
    app.run(main)

