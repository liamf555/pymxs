
try:
    import gymnasium as gym
    from gymnasium import spaces
    GYM_FLAG = False
except:
    import gym
    from gym import spaces
    GYM_FLAG = True
import sys
import math

import numpy as np
sys.path.insert(1, '/home/tu18537/dev/mxs/pymxs/models')

from pyaerso import AffectedBody, AeroBody, Body
from gym_mxs.model import Combined, calc_state, inertia, trim_points

from gym_mxs.wind import DrydenGustModel


AIRCRAFT_HEIGHT = 270 #mm
AIRCRAFT_LENGTH = 1000 #mm



# gust_vector = np.zeros
# gust_model = DrydenGustModel(dt= 0.01, b =1.1)

class MxsEnv(gym.Env):
    metadata = {
        "render_modes": ["ansi", "human"],
        "render_fps": 4,
    }

    def __init__(self, render_mode=None, reward_func=lambda obs: 0.5, timestep_limit=100, scenario=None, **kwargs):
        self.observation_space = spaces.Box(
            low=np.float32(-np.inf),
            high=np.float32(np.inf),
            shape=(13,),
            dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=np.array([np.radians(-60),0], dtype=np.float32),
            high=np.array([np.radians(60),1], dtype=np.float32),
            shape=(2,),
            dtype=np.float32
        )
        
        self.selected_trim_point = 15

        mass = 1.221
        position,velocity,attitude,rates = calc_state(
            np.radians(trim_points[self.selected_trim_point][0]),
            self.selected_trim_point,
            [0,0,0],
            False
        )

        self.dT = 0.01
        self.wind_i = 0

        
        self.initial_state = [position,velocity,attitude,rates]

        body = Body(mass,inertia,*self.initial_state)
        
    # vehicle = AffectedBody(aerobody,[Lift(),Drag(),Moment()])
        # if self.dryden:
        #     aerobody = AeroBody(body, self.create_WindModel())
        # else:
        aerobody = AeroBody(body)
        self.vehicle = AffectedBody(aerobody,[Combined(self.dT)])

        self.steps = 0

        self.reward_func = reward_func
        self.reward_state = None
        self.timestep_limit = timestep_limit

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode
        self.scenario = scenario

    def _get_obs(self):
        return np.array(self.vehicle.statevector)
    
    # def body_to_world(self, quat):
    #     # A method to create a rotation matrix from a quaternion 

    #     pitch = self.get_pitch(*quat)

        # get dryden gust in body frame
    @property
    def gust_vector(self):
        return self._gust_vector
    
    @gust_vector.setter
    def gust_vector(self, value):
        self._gust_vector = value
    
    def create_WindModel(self):
        return self.WindModel(self)
    
    class WindModel:
            def __init__(self, mxs):
                self.mxs = mxs
                 
            def get_wind(self,position, attitude):

                wind = self.mxs.calc_wind(attitude)
                # print(wind)
                return wind
            def step(self,dt):
                pass

    def calc_wind(self, attitude):
        pitch = self.get_pitch(*attitude)

        steady_wind = self.steady_vector
        # print(steady_wind)

        # print(self.outer_class.gust_vector)
        if self.config["domain"]["dryden"] is not None:
            gust_u = self.gust_vector[0][self.wind_i]
            gust_w = self.gust_vector[1][self.wind_i]
            self.wind_i += 1
            # print(self.wind_i)
            gust_world = self.body_to_world_frame([gust_u, gust_w], pitch)
            gust_world = np.array([gust_world[0], 0., gust_world[1]])
        else:
            gust_world = np.zeros(3)

        # print(f"steady_wind: {steady_wind}")
        # print(f"gust_world: {gust_world}")

    
        wind = steady_wind + gust_world

        return wind
    
    def reset(self, seed=None, return_info=False, options=None):
        self.vehicle.statevector = [
            *self.initial_state[0],
            *self.initial_state[1],
            *self.initial_state[2],
            *self.initial_state[3]
        ]

        # self.vehicle.set_state(np.array([[
        #     *self.initial_state[0],
        #     *self.initial_state[1],
        #     *self.initial_state[2],
        #     *self.initial_state[3]
        # ]]).T)

        self.elevator = np.radians(trim_points[self.selected_trim_point][1])
        self.throttle = trim_points[self.selected_trim_point][2]

        self.steps = 0
        self.reward_state = None

        observation = self._get_obs()
        info = {}
        return (observation,info) if return_info else observation

    def step(self, action):
        self.elevator = np.clip(self.elevator + action[0] * self.dT, np.radians(-30), np.radians(30))
        self.throttle = action[1]
        self.vehicle.step(self.dT,[0,self.elevator,self.throttle,0])
        observation = self._get_obs()

        reward, ep_done, self.reward_state = self.reward_func(observation, self.reward_state)
        self.steps += 1
        # done = ep_done or self.steps >= self.timestep_limit
        done = ep_done
        # print(observation)
        return observation, reward, done, {}
    

    def render(self, mode):
        if mode == "ansi":
            elements = ", ".join([f"{v:.{4}f}" for v in self._get_obs()])
            return f"[{elements},{self.vehicle.airstate[0]},{self.vehicle.airstate[2]},{self.elevator},{self.throttle}]"

    def close(self):
        pass

    def body_to_world_frame(self, body_velocity, aircraft_angle):
        # print(f"body_velocity: {body_velocity}")
        # print(f"aircraft_angle: {aircraft_angle}")
    # Calculate the rotation matrix for the aircraft's orientation
        rot_matrix = [
            [math.cos(aircraft_angle), -math.sin(aircraft_angle)],
            [math.sin(aircraft_angle), math.cos(aircraft_angle)]
        ]

        # Transform the body frame linear velocity to the world frame
        world_velocity_x = rot_matrix[0][0] * body_velocity[0] + rot_matrix[0][1] * body_velocity[1]
        world_velocity_y = rot_matrix[1][0] * body_velocity[0] + rot_matrix[1][1] * body_velocity[1]

        return world_velocity_x, world_velocity_y

    def get_pitch(self, qx,qy,qz,qw):
        if True:
            # Can't use scipy if 'jit'ting
        #     rot = Rotation.from_quat(np.array([qx,qy,qz,qw]).T)
        #     [yaw, pitch, roll] = rot.as_euler('zyx', degrees=False)
        #     # print([yaw,pitch,roll])
        #     if yaw != 0:
        #         if pitch > 0:
        #             pitch = np.pi/2 + (np.pi/2 - pitch)
        #         else:
        #             pitch = -np.pi/2 + (-np.pi/2 - pitch)
        # if False:
            sinp = np.sqrt(1 + 2 * (qw*qy - qx*qz))
            cosp = np.sqrt(1 - 2 * (qw*qy - qx*qz))
            pitch = 2 * np.arctan2(sinp, cosp) - np.pi / 2
        return pitch