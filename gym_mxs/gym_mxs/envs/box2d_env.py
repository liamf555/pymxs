import math
from matplotlib.path import Path
from numpy import random as random
try:
    import gymnasium as gym
    from gymnasium import spaces
    GYM_FLAG = False
except ImportError:
    import gym
    from gym import spaces
    GYM_FLAG = True

import sys
from . import MxsEnv
import Box2D as b2
import json
import wandb
from earcut import earcut
from scipy import signal
import pygame
from pygame.locals import K_p, KEYDOWN, QUIT
import numpy as np
sys.path.insert(1, '/home/tu18537/dev/mxs/pymxs/models')

from pyaerso import AffectedBody, AeroBody, Body
from gym_mxs.model import Combined, calc_state, inertia, trim_points
from gym_mxs.wind import DrydenGustModel

from scipy.spatial.transform import Rotation

# Constants for rendering
# PPM = 5  # Pixels per meter (scaling factor)
WINDOW_WIDTH = 800 * 1.5
WINDOW_HEIGHT = 600 * 1.5
SCALE = 4
GEOM_SCALE = 16.187 /2

# GEOMETRY_SCALE = 7.5

def multiply_dict_values(d, scalar, excluded_keys=None):
    if excluded_keys is None:
        excluded_keys = []
    result = {}
    for key, value in d.items():
        if key not in excluded_keys:
            if isinstance(value, list):
                result[key] = [x * scalar for x in value]
            else:
                result[key] = value * scalar
        else:
            result[key] = value
    return result



# obstacle course params




MIN_WALL_SPACING = 0  
INITIAL_WALL_OFFSET = 0


# LIDAR params
N_LIDAR_RAYS = 10
LIDAR_OFFSET_VECTOR = b2.b2Vec2(4, 0)
LIDAR_RANGE = 50*GEOM_SCALE #approximate range of lidar in m
LIDAR_FOV = 20
ANGLE_INCREMENT = LIDAR_FOV / (N_LIDAR_RAYS - 1)

#reward params
K1 = 1.0
K2 = 0.25

# aircraft gemotry
PLANE_OUTLINE_PATH = "M -8.4344006,0.8833226 L -3.6174367,1.4545926 C -2.6957014,1.5861425 -1.2977255,1.7000225 -0.44895008,0.98453256 C 0.97534922,0.9358126 2.1554971,0.9295626 3.4694746,0.8473026 C 3.4694746,0.8473026 4.1040207,0.8167026 4.1204559,0.5018026 C 4.1306045,0.3072626 4.2764544,-1.2268074 1.7485665,-1.3031174 L 1.7604066,-1.0355474 L 1.3209316,-1.0233574 L 1.3822972,-1.7538274 C 1.9074643,-1.7412074 2.0141441,-2.5891474 1.4111688,-2.6446878 C 0.80819248,-2.7002378 0.8023354,-1.8387774 1.1839183,-1.7720774 L 1.0908357,-1.0522274 L -5.2189818,-0.91913738 L -12.198397,-0.80283738 C -12.198397,-0.80283738 -12.820582,-0.84082738 -12.643322,-0.31380735 C -12.466063,0.2132026 -11.622877,3.1026526 -11.622877,3.1026526 L -10.120232,3.1500026 C -10.120232,3.1500026 -9.8463164,3.1552526 -9.6753635,2.8748926 C -9.5044154,2.5944926 -8.4343678,0.8834126 -8.4343678,0.8834126 Z"
MAIN_WING_PATH="M 0.32346345,0.1815526 C 1.8962199,0.1638926 1.9691414,-0.33848735 0.34369001,-0.39724735 C -2.0368286,-0.46197735 -3.4920188,-0.15280735 -3.3975903,-0.13907735 C -1.5720135,0.1264326 -0.81500941,0.1943226 0.32346345,0.1815526 Z"
TAIL_PLANE_PATH="M -8.9838929,0.4470726 C -7.9395132,0.4475726 -7.8954225,0.0758826 -8.975461,0.01829265 C -10.557021,-0.05024735 -11.520801,0.1663226 -11.457966,0.1773326 C -10.24323,0.3898926 -9.739887,0.4467426 -8.9838897,0.4471126 Z"
class MxsEnvBox2D(MxsEnv):


    def __init__(self, render_mode, training, use_lidar=False, acl = False, context = np.array([0.,0.,0.]), config=None, output_dir=None):
        self.render_mode = render_mode
        super().__init__(render_mode= render_mode)

        self.acl_on = acl

        self.config = config
        #load config file
    
        self.obstacle_config = self.config["obstacles"]
        # multiply each value in obstacle_config dict by the geom scale
        self.geom_scale = self.config["geometry"]["geom_scale"]
        self.obstacle_config = multiply_dict_values(self.obstacle_config, self.geom_scale, excluded_keys=["n_obstacle_walls"])
        self.lidar_config = self.config["lidars"]
        if self.acl_on:
            self.context = context

        self.window_dims = (
            self.config["rendering"]["window_width"],
            self.config["rendering"]["window_height"],
        )
        self.scale = self.config["rendering"]["scale"]

        self.middle_y = (self.window_dims[1] / 2) / self.scale

        self.steps = 0
        self.training = training

        if not self.training:
            self.first_render = True
            self.episode_counter = 1

        self.controls_high = np.array([np.radians(60), 1])
        self.controls_low = np.array([np.radians(-60), 0])


        self.completion_counter = 0
        self.wall_counter = 0
        self.alt_counter = 0
        self.speed_counter = 0
        self.pitch_counter = 0

        # if self.turbulence:
        #     self.gust_model = DrydenGustModel(dt=self.dT, b=1.1, Va = 13, intensity="light")

        self.reward_state = None

        self.use_lidar = use_lidar

        if self.use_lidar is True:
            n_obs = 8 + self.config["lidars"]["n_lidar_rays"]
        else:
            n_obs = 11

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode
        # self.scenario = scenario

        self.paused = False

        # create bx2d world 
    
        # reset world
        # x z u w qy qw q l1 l2 l3 l4 l5 l6 l7 l8 l9 l10 elevator throttle
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(n_obs,), dtype=np.float32)

        self.action_space = spaces.Box(
            low=np.array([-1, -1], dtype=np.float32),
            high=np.array([1,1], dtype=np.float32),
            shape=(2,),
            dtype=np.float32
            
        )

        self._gust_vector = np.zeros((2,3))
        self.steady_vector = np.zeros(3)
        self.wind_i = 0
        # print(f"action space: {self.action_space}")
        # breakpoint()

        self.initial_ac_pos = (0, (self.window_dims[1] / 2) / self.scale)
        self.render_screen = None
        self.clock = None
        self.x_pos_prev = 0
        self.last_obstacle_x = 0
        self.termination = False
        self.n_episodes = 0

        

    def _get_obs(self):
        return np.array(self.vehicle.statevector)
    
    def _reset_aircraft_initial_state(self):

        if self.config["domain"]["airspeed"] is not None or not all(v == 0 for v in self.config["domain"]["airspeed"]):
            if len(self.config["domain"]["airspeed"])> 1:                
                self.selected_trim_point = np.random.choice(self.config["domain"]["airspeed"])
            elif len(self.config["domain"]["airspeed"])== 1:
                self.selected_trim_point = self.config["domain"]["airspeed"][0]
        else:
            self.selected_trim_point = 15

        # print(f"selected trim point: {self.selected_trim_point}")

        alpha = np.radians(trim_points[self.selected_trim_point][0])

        if self.config["domain"]["steady"] is not None or all(v != 0 for v in self.config["domain"]["steady"]):
            if len(self.config["domain"]["steady"])> 1:
                self.steady_vector = np.array([np.random.uniform(self.config["domain"]["steady"][0], self.config["domain"]["steady"][1]), 0, 0])
            elif len(self.config["domain"]["steady"])== 1:
                self.steady_vector = np.array([self.config["domain"]["steady"][0], 0, 0])
       
        mass = 1.221
        # print(f"steady vector: {self.steady_vector}")
        position,velocity,attitude,rates = calc_state(
            alpha,
            self.selected_trim_point,
            self.steady_vector, 
            False
        ) 

        self.initial_state = [position,velocity,attitude,rates]

        body = Body(mass,inertia,*self.initial_state)

        if self.config["domain"]["dryden"] is not None:
            # print("dryden")
            self.wind_i = 0
            self.gust_model = DrydenGustModel(V_a=self.selected_trim_point, dt = self.dT, intensity=self.config["domain"]["dryden"])
            self.gust_model.simulate()
            self.gust_vector = self.gust_model.vel_lin
        else: 
            self.gust_vector = np.zeros((2,3))
        
    # vehicle = AffectedBody(aerobody,[Lift(),Drag(),Moment()])
        if self.config["domain"]["dryden"] is not None or (self.config["domain"]["steady"] is not None or all(v != 0 for v in self.config["domain"]["steady"])):
            # print("wind")
            aerobody = AeroBody(body, self.create_WindModel())
        else:
            aerobody = AeroBody(body)
        self.vehicle = AffectedBody(aerobody,[Combined(self.dT)])

    
    def _get_full_obs(self, obs):
        # obs = self.long_obs(obs)

        position = (self.aircraft.position - self.initial_ac_pos) / self.geom_scale
        velocity = self.aircraft.linear_velocity / self.geom_scale
        angle = self.aircraft.angle
        angular_velocity = self.aircraft.angular_velocity

        wall_pos = self.calc_wall_position_delta()
        if self.use_lidar:
            wall_obs = [l.fraction for l in self.lidars]
        else:
            # wall_pos = self.calc_wall_position_delta()
            wall_height = self.get_next_wall_gap_height()
            wall_obs = [*wall_pos, wall_height]
        # print(f"Position: {position}")
        # print("Wall position delta: ", wall_pos)
        observation = np.array([*position, *velocity, angle, angular_velocity, *wall_obs, self.elevator, self.throttle], dtype=np.float32)
        return observation
    
    def long_obs(self, obs):
    #           x       z       u       w      qy      qw        q
        return [obs[0], obs[2], obs[3], obs[5], obs[7], obs[9], obs[11]]

    def reset(self, seed=None, return_info=GYM_FLAG, options=None):
        self.vehicle.statevector = [
            *self.initial_state[0],
            *self.initial_state[1],
            *self.initial_state[2],
            *self.initial_state[3]
        ]

        if self.acl_on:
            self.obstacle_config["gap_height"] = [self.context[0], self.obstacle_config["gap_height"][1]]
            self.obstacle_config["gap_width"] = [self.obstacle_config["gap_width"][0], self.context[1]]
            self.obstacle_config["gap_offset"] = [self.obstacle_config["gap_offset"][0], self.context[1]]
            # wandb.log({
            #     "Gap height mean": self.obstacle_config["gap_height"][0],
            #     "Gap width mean": self.obstacle_config["gap_width"][0],
            #     "Gap offset SD": self.obstacle_config["gap_offset"][1]
            #     })
            wandb.log({
                "Gap height mean": self.obstacle_config["gap_height"][0],
                "Gap width SD": self.obstacle_config["gap_width"][1],
                "Gap offset SD": self.obstacle_config["gap_offset"][1]
                })

        self._reset_aircraft_initial_state()

        self.elevator = np.radians(trim_points[self.selected_trim_point][1])
        self.throttle = trim_points[self.selected_trim_point][2]

        observation = self._get_obs()
        # print(f"Initial velocity: {observation[3]}")

        self.aircraft_initial_angle = self.get_pitch(*self.initial_state[2])
        self.aircraft_initial_velocity = self.get_velocity(observation)[0]

        
        self.wall_i = 0
        self.reset_world()
        observation = self._get_full_obs(observation)
        self.simtime = 0

        

        self.steps = 0
        self.reward_state = None
        self.scroll_offset = 0

        
        self.termination = False
        self.success = False
        

        info = {}

        if self.render_mode == "human":
            self._render_frame()

        # print(observation, info)

        return (observation,info) if not return_info else observation

    def step(self, action, return_info=GYM_FLAG):
        # print(f"action: {action}")

        controls = self.action_mapper(
            action, 
            self.action_space.low, self.action_space.high,
            self.controls_low, self.controls_high)

        # print(f"controls: {controls}")
        # get actions
        self.elevator = np.clip(self.elevator + controls[0] * self.dT, np.radians(-30), np.radians(30))
        self.throttle = controls[1]

        self.vehicle.step(self.dT,[0,self.elevator,self.throttle,0])
        #get observation from mxs
        observation = self._get_obs()
        # print(f"Velocity: {observation[3]}")

        #[x,y,z, u,v,w, qx,qy,qz,qw, p,q,r] 
        # print(f"mxs vel: {observation[3:6]}")
        linear_velocities, angular_velocity = self.get_velocity(observation)

        self.pitch = self.get_pitch(*observation[6:10])

        linear_velocities = self.body_to_world_frame(linear_velocities, self.pitch)

        linear_velocities = (linear_velocities[0], linear_velocities[1])

        # print(f"linear vel: {linear_velocities[0]/GEOM_SCALE}")

        self.aircraft.linear_velocity = linear_velocities
 
        self.aircraft.angular_velocity = angular_velocity
        
        # step box2d
        # print(f"linear vel: {linear_velocities}")
        self.world.Step(self.dT, 8 , 3) #TODO: what are these numbers?
        # print(f"box2d vel: {self.aircraft.linear_velocity}")
        # print(linear_velocities[0]/GEOM_SCALE)
        self.pitch = math.degrees(self.get_pitch(*observation[6:10]))

        if self.use_lidar:
            self.update_lidar()

        observation = self._get_full_obs(observation)
        # print(f"observation: {observation}")

        reward = self.obstacle_avoid_box2d(observation)
        # print(f"reward: {reward}")
        self.steps += 1
        # done = ep_done or self.steps >= self.timestep_limit
        done = self.termination 

        if self.aircraft.position.x * self.scale > self.window_dims[0] / 2:
            self.scroll_offset = self.aircraft.position.x * self.scale - self.window_dims[0] / 2

        if self.render_mode == "human":
            self._render_frame()
        # print(observation)

        if GYM_FLAG:
            info = {"success": self.success}
            return observation, reward, done, info
        else:
            return observation, reward, done, False, {}

        # return observation, reward, done, False, {}
    
    def get_velocity(self, state):
        """ Returns the u and w velocities and q angular velocity
        of the aircraft, based on whether the env has standard or longitunda only
        observation space."""
        # print(f"Linear vel: {state[3]*GEOM_SCALE}, {state[5]*ef}")
        return (state[3]*self.geom_scale, state[5]*self.geom_scale), state[11]
        
    def action_mapper(self, action, input_low, input_high, output_low, output_high):
    
        return (action - input_low) * (output_high - output_low) / (input_high - input_low) + output_low
    
    def update_lidar(self):
        pos = self.aircraft.position
        angle = self.aircraft.angle
        offset_vector = b2.b2Vec2(self.lidar_config["lidar_offset_vector"])  # Define the offset vector for the front of the aircraft

        # Rotate the offset vector according to the body's angle
        rotated_offset_vector = b2.b2Vec2(
            offset_vector.x * math.cos(angle) - offset_vector.y * math.sin(angle),
            offset_vector.x * math.sin(angle) + offset_vector.y * math.cos(angle),
        )

        # Calculate the starting point of the lidar beams
        lidar_start_point = pos + rotated_offset_vector

        for i in range(self.lidar_config["n_lidar_rays"]):
            self.lidars[i].fraction = 1.0
            angle_offset = (0.5 * i / 10.0) - 5.0  # Adjust the starting angle of the lidar beams
            # angle_offset = (1 * i / N_LIDAR_RAYS) - 5.25
            # angle_offset = (ANGLE_INCREMENT * i) - (LIDAR_FOV / 2)
            start_angle = angle + angle_offset
            # end_angle = angle + 1.5 * i / 10.0 # why these magic numbers?

            self.lidars[i].p1 = lidar_start_point
            self.lidars[i].p2 = (
                pos[0] + math.sin(start_angle) * (self.lidar_config["lidar_range"] *self.geom_scale),
                pos[1] - math.cos(start_angle) * (self.lidar_config["lidar_range"] *self.geom_scale),
            )
            self.world.RayCast(self.lidars[i], self.lidars[i].p1, self.lidars[i].p2)
    
    def game_over(self):
        # print("Contact")
        if self.contact_listener.game_over_collision:
            self.contact_listener.game_over_collision = False
            return True
        return False
    
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
    
    def obstacle_avoid_box2d(self, obs):
        k1 = self.config["reward"]["k1"]
        penalisation = self.config["reward"]["penalisation"]
        # print(f"obs: {obs}")

        # print(f"obs: {obs}")
        x_pos = obs[0]
        alt = obs[1]
        u_vel = obs[2]
        
        desired_alt = 0
        x_pos_prev = self.x_pos_prev
        # pitch = self.pitch
        pitch = self.aircraft.angle
        # print(f"pitch: {math.degrees(pitch)}")
        pitch_rate = self.aircraft.angular_velocity

        # print(f"X pos: {x_pos}")
        # print(f"X pos prev: {x_pos_prev}")
        # print(f"Alt: {alt}")
        # print(f"Desired Alt: {desired_alt}")
        # print(f"self.last_obstacle_x: {self.last_obstacle_x}")
        # print(f"wall max height: {MAX_WALL_HEIGHT/GEOM_SCALE}")

    

        # intitial_position = env.initial_state[0]

        # # print(f"z_error: {z_error}, pitch_error: {pitch_error}, x_error: {x_error}")
        # pitch_r = rewards.tolerance(pitch_error, bounds = (-5, 5), margin = 90)
        # alt_r = rewards.tolerance(z_error, bounds = (-2.5, 2.5), margin = 15)
        # x_r = rewards.tolerance(x_error, bounds = (-2.5, 2.5), margin =20.0)

        # reward = (pitch_r * 0.4)  + (alt_r * 0.4) + (x_r * 0.2)

        rw_1 = k1* ((x_pos - x_pos_prev)/self.dT)
        # print(f"rw_1: {rw_1}")
        reward = rw_1
        self.x_pos_prev = x_pos

        # alitude maintenance
        # rw2 = -K2 * abs(desired_alt - alt)

        # print(f"rw2: {rw2}")
        # reward += rw2

        if self.game_over():
            # print("Hit wall")
            # reward = -100
            self.termination = True
            self.wall_counter += 1


        if self.wall_i == self.obstacle_config["n_obstacle_walls"]:
            # print(f"Passed obstacle at {x_pos}")
            self.termination = True
            self.success = True
            # print("Finished!")
            self.completion_counter += 1
        if abs(math.degrees(pitch)) > 120:
            # print("Too pitchy")
            reward = penalisation
            self.termination = True
            self.pitch_counter += 1
        # print(f"Reward {reward}")
        if u_vel > 24.5:
            # print("Too fast")
            reward = penalisation
            self.termination = True
            self.speed_counter += 1
        if abs(alt) > 25:
            # print("Too high")
            reward = penalisation
            self.termination = True
            self.alt_counter += 1
        if self.termination and self.training:
            self.n_episodes += 1
            wandb.log({
                "n_episodes": self.n_episodes,
                "Terminal position": x_pos,
                "Terminal pitch counter": self.pitch_counter,
                "Terminal wall counter": self.wall_counter,
                "Terminal speed counter": self.speed_counter,
                "Terminal alt counter": self.alt_counter,
                "Success counter": self.completion_counter
            })


            # wandb.log({"Terminal position": x_pos}, step = self.n_episodes)
            # wandb.log({}, step = self.n_episodes)
            # wandb.log({"Terminal wall counter": self.wall_counter},step = self.n_episodes)
            # wandb.log({"Terminal speed counter": self.speed_counter},step = self.n_episodes)
            # wandb.log({"Terminal alt counter": self.alt_counter},step = self.n_episodes)
            # wandb.log({"Terminal completion counter": self.completion_counter},step = self.n_episodes)
        # if reward < 0:
            # print(obs)
            # # print(f"altitude: {alt}")
            # # print(f"alt error: {desired_alt - alt}")
            # print(f" x pos: {x_pos}")
            # print(f" x pos prev: {x_pos_prev}")
            # print(f"reward: {reward}")
            # print(f"rw_1: {rw_1}")
            # print(f"rw2: {rw2}")
        
        return reward
    
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
    
    def reset_world(self):
        self.world = None

        self.world = b2.b2World(gravity=(0, 0))

        self.create_aircraft_and_obstacles()

        # body_def = b2.b2BodyDef()
        # body_def.type = b2.b2_kinematicBody
        # body_def.position = b2.b2Vec2(15.5, MIDDLE_Y-1)

        # top_wall_body = self.world.CreateBody(body_def)
        # wall_shape = b2.b2PolygonShape()
        # wall_shape.SetAsBox(SCALE*AIRCRAFT_LENGTH/1000, SCALE*AIRCRAFT_HEIGHT/1000)
        # top_wall_body.CreateFixture(shape=wall_shape)

        self.contact_listener = self.ContactListener(self.aircraft.aircraft_body)
        self.world.contactListener = self.contact_listener

        class LidarCallback(b2.b2RayCastCallback):
                def ReportFixture(self, fixture, point, normal, fraction):
                    if (fixture.filterData.categoryBits & 1) == 0:
                        return -1
                    self.p2 = point
                    self.fraction = fraction
                    return fraction

        self.lidars = [LidarCallback() for _ in range(self.lidar_config["n_lidar_rays"])]
        self.update_lidar()

    def create_aircraft_and_obstacles(self):
        self.aircraft = MXSGeometry(self.world, initial_position=self.initial_ac_pos, initial_angle=self.aircraft_initial_angle, initial_velocity=self.aircraft_initial_velocity)
        obstacle_geom = ObstacleGeometry(self.world, initial_position=self.initial_ac_pos, wall_config=self.obstacle_config, geom_scale=self.geom_scale)
        self.last_obstacle_x = obstacle_geom.last_wall_x
        # print(f"Wall positions: {obstacle_geom.wall_positions}")
        # print(f"Wall heights: {obstacle_geom.wall_heights}")
        # print(f"Wall widths: {obstacle_geom.wall_widths}")
        self.wall_positions = obstacle_geom.wall_positions
        self.wall_heights = obstacle_geom.wall_heights
        self.wall_widths = obstacle_geom.wall_widths
        self.x_pos_prev = 0

        cloud1 = Cloud(100, 100, self.world_to_screen)
        cloud2 = Cloud(300, 200, self.world_to_screen, cloud_size=70)
        cloud3 = Cloud(600, 150, self.world_to_screen, cloud_size=60)
        cloud4 = Cloud(800, 100, self.world_to_screen, cloud_size=50)
        cloud5 = Cloud(1000, 200, self.world_to_screen, cloud_size=70)
        cloud6 = Cloud(1200, 150, self.world_to_screen, cloud_size=60)
        # cloud7 = Cloud(2800, 100, self.world_to_screen, cloud_size=50)
        # cloud8 = Cloud(3200, 200, self.world_to_screen, cloud_size=70)
        # cloud9 = Cloud(3600, 150, self.world_to_screen, cloud_size=60)
        # cloud10 = Cloud(4000, 100, self.world_to_screen, cloud_size=50)

        self.clouds = [cloud1, cloud2, cloud3, cloud4, cloud5, cloud6]

    # def get_next_obstacle_position(self):
    #     """ A function to get the absolute position of the next obstacle """
    #     obstacle_i = self.obstacle_i
    #     return self.obstacle_positions[obstacle_i]
    
    def calc_wall_position_delta(self):
        """ A function to calculate the position delta between the aircraft and the next obstacle """
        aircraft_position = self.aircraft.position/self.geom_scale 
        next_obstacle_position = self.get_next_wall_position()
        # print(f"Wall position: {next_obstacle_position}")
        delta_x = next_obstacle_position[0] - aircraft_position[0]
        delta_y = next_obstacle_position[1] - aircraft_position[1]

        return (delta_x, delta_y)
  
    def get_next_wall_position(self):
        """A method to return the position of the next obstacle relative to the aircraft,
        based on the aicraft's current position"""
        aircraft_position = self.aircraft.position/self.geom_scale
        if aircraft_position[0] > (self.wall_positions[self.wall_i][0] + self.wall_widths[self.wall_i]):
            # print("Wall passed")
            if self.wall_i < (len(self.wall_positions)):
            # breakpoint()
                self.wall_i += 1
            
        if self.wall_i == len(self.wall_positions):
            return self.wall_positions[self.wall_i-1]
      

        # wall_centre_x = self.wall_positions[self.wall_i][0]
        # wall_centre_y = self.wall_positions[self.wall_i][1]
        
        return self.wall_positions[self.wall_i]
    
    def get_next_wall_gap_height(self):
        """A method to return the height of the next obstacle relative to the aircraft,
        based on the aicraft's current position"""
        if self.wall_i == len(self.wall_positions):
            return self.wall_heights[self.wall_i-1]

        return self.wall_heights[self.wall_i]


    class ContactListener(b2.b2ContactListener):
        def __init__(self, aircraft_body):
            self.aircraft_body = aircraft_body
            super().__init__()
            self.game_over_collision = False

        def BeginContact(self, contact):
            body_a, body_b = contact.fixtureA.body, contact.fixtureB.body
            if (body_a == self.aircraft_body and body_b.type == b2.b2_staticBody) or (body_a.type == b2.b2_staticBody and body_b == self.aircraft_body):
                self.game_over_collision = True
                # pass

    def render(self):
        if self.render_mode == "ansi" and self.training == False:
            with open(self.output_path, "w") if self.output_path else nullcontext() as outfile:
                if self.first_render:
                    outfile.write("episode,time,x,y,z,u,v,w,qx,qy,qz,qw,p,q,r,alpha,airspeed,elevator,throttle\n")
                    self.first_render = False
                elements = ", ".join([f"{v:.{4}f}" for v in self._get_obs()])
                outfile.write(f"{self.episode_count},{self.simtime},{[{elements},{self.vehicle.airstate[0]},{self.vehicle.airstate[2]},{self.elevator},{self.throttle}][1:-1]}\n")
                self.simtime += self.dT








            # return f"[{elements},{self.vehicle.airstate[0]},{self.vehicle.airstate[2]},{self.elevator},{self.throttle}]"
        
    def _render_frame(self):
        if self.render_screen is None and self.render_mode == "human":
            print("Initializing pygame")
            pygame.init()
            self.scroll = 0
            self.scroll_offset = 0
            self.render_screen = pygame.display.set_mode((self.window_dims[0], self.window_dims[1]), 0, 32)
            pygame.display.set_caption("Box2D World Rendering")
            
            self.surf = pygame.Surface(
                (self.window_dims[1] + max(0.0, self.scroll) * self.scale, self.window_dims[0])
                )
            pygame.transform.scale(self.surf, (self.scale, self.scale))
        # self.clock = pygame.time.Clock()
        
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        pos = self.aircraft.position
        # print(f"{pos.x} {pos.y}")
        self.scroll = pos.x - self.window_dims[0] / self.scale / 5
        self.render_screen.fill((135, 206, 235))
    # Draw the world first
        self.draw_world()
        if self.termination:
            if self.success:
                self.draw_message("Success")
            else:
                self.draw_message("Wasted")
            pygame.time.delay(1000)


        pygame.display.flip()
        self.clock.tick(60)

    def close(self):
        if self.render_screen is not None:
            # pygame.display.quit()
            pygame.quit

    def draw_world(self):
        # Draw clouds
        [cloud.draw(self.render_screen, self.scroll_offset, 0) for cloud in self.clouds]
    # Draw bodies
        for body in self.world.bodies:
            for fixture in body.fixtures:
                if type(fixture.shape) == b2.b2PolygonShape:
                    color = (255, 0, 0)  # Default color: blue
                    if body == self.aircraft.aircraft_body:  # If the body is the aircraft, set the color to green
                        color = (0, 255, 0)
                    self.draw_polygon(fixture.shape, body, color)

        if self.use_lidar:
        # Draw lidar rays
            for lidar in self.lidars:
                if hasattr(lidar, "p1") and hasattr(lidar, "p2"):
                    start_screen = self.world_to_screen(lidar.p1)
                    end_screen = self.world_to_screen(lidar.p2)
                    pygame.draw.line(self.render_screen, (255, 0, 0), start_screen, end_screen, 1)
        else:
            if self.wall_i < len(self.wall_positions):
            # Draw a cross representing the cetnre of teh next obstacle
                next_wall_position = [p * self.geom_scale for p in self.get_next_wall_position()]
                # print(f"Next wall position: {next_wall_position}")
                next_wall_position_screen = self.world_to_screen(next_wall_position)
                pygame.draw.line(self.render_screen, (0, 0, 255), (next_wall_position_screen[0] - 10, next_wall_position_screen[1]), (next_wall_position_screen[0] + 10, next_wall_position_screen[1]), 1)

                # Draw a vertical line
                pygame.draw.line(self.render_screen, (0, 0, 255), (next_wall_position_screen[0], next_wall_position_screen[1] - 10), (next_wall_position_screen[0], next_wall_position_screen[1] + 10), 1)

        
        # cloud1.draw(self.render_screen, self.scroll, 0)
        # cloud2.draw(self.render_screen, self.scroll, 0)
        # cloud3.draw(self.render_screen, self.scroll, 0)


        pygame.display.flip()

    
    def draw_cloud(self, x, y, scroll_x, scroll_y, cloud_size=50):
        color = (255, 255, 255)  # white color for clouds

        screen_x = x
        screen_y = y 

        for _ in range(5):
            offset_x = random.randint(-cloud_size // 2, cloud_size // 2)
            offset_y = random.randint(-cloud_size // 4, cloud_size // 4)
            ellipse_width = random.randint(cloud_size // 2, cloud_size)
            ellipse_height = random.randint(cloud_size // 4, cloud_size // 2)
            pygame.draw.ellipse(self.render_screen, color, (screen_x + offset_x, screen_y + offset_y, ellipse_width, ellipse_height))


    

    
    
    def draw_polygon(self, polygon, body, color):
        vertices = [(body.transform * v) * self.scale for v in polygon.vertices]
        vertices = [(v[0] - self.scroll_offset, self.window_dims[1] - v[1]) for v in vertices]
        pygame.draw.polygon(self.render_screen, color, vertices)

    def world_to_screen(self, world_point):
        screen_x = int(world_point[0] * self.scale - self.scroll_offset)
        screen_y = int(self.window_dims[1] - world_point[1] * self.scale)
        return (screen_x, screen_y)
    
    def draw_message(self, message):
        font = pygame.font.Font(None, 64)  # You can adjust the font size (64 in this case) as needed
        text_surface = font.render(message, True, (0, 0, 0))  # You can change the color (red in this case)
        text_rect = text_surface.get_rect()
        text_rect.center = (self.window_dims[0] // 2, self.window_dims[1] // 2)
        self.render_screen.blit(text_surface, text_rect)
        pygame.display.flip()
        
class MXSGeometry():
    """A class to store and process the geometry of the MXS aircraft"""

    PLANE_OUTLINE_PATH = "M -8.4344006,0.8833226 L -3.6174367,1.4545926 C -2.6957014,1.5861425 -1.2977255,1.7000225 -0.44895008,0.98453256 C 0.97534922,0.9358126 2.1554971,0.9295626 3.4694746,0.8473026 C 3.4694746,0.8473026 4.1040207,0.8167026 4.1204559,0.5018026 C 4.1306045,0.3072626 4.2764544,-1.2268074 1.7485665,-1.3031174 L 1.7604066,-1.0355474 L 1.3209316,-1.0233574 L 1.3822972,-1.7538274 C 1.9074643,-1.7412074 2.0141441,-2.5891474 1.4111688,-2.6446878 C 0.80819248,-2.7002378 0.8023354,-1.8387774 1.1839183,-1.7720774 L 1.0908357,-1.0522274 L -5.2189818,-0.91913738 L -12.198397,-0.80283738 C -12.198397,-0.80283738 -12.820582,-0.84082738 -12.643322,-0.31380735 C -12.466063,0.2132026 -11.622877,3.1026526 -11.622877,3.1026526 L -10.120232,3.1500026 C -10.120232,3.1500026 -9.8463164,3.1552526 -9.6753635,2.8748926 C -9.5044154,2.5944926 -8.4343678,0.8834126 -8.4343678,0.8834126 Z"
    MAIN_WING_PATH="M 0.32346345,0.1815526 C 1.8962199,0.1638926 1.9691414,-0.33848735 0.34369001,-0.39724735 C -2.0368286,-0.46197735 -3.4920188,-0.15280735 -3.3975903,-0.13907735 C -1.5720135,0.1264326 -0.81500941,0.1943226 0.32346345,0.1815526 Z"
    TAIL_PLANE_PATH="M -8.9838929,0.4470726 C -7.9395132,0.4475726 -7.8954225,0.0758826 -8.975461,0.01829265 C -10.557021,-0.05024735 -11.520801,0.1663226 -11.457966,0.1773326 C -10.24323,0.3898926 -9.739887,0.4467426 -8.9838897,0.4471126 Z"    

    def __init__(self, world=None, render_mode="human", initial_position=(0, 0), initial_angle=0, initial_velocity=(0, 0)):
        self.world = world
        self.create_aircraft_body(initial_position, initial_angle, initial_velocity)

    def _parse_path(self, path):
        vertices = []
        codes = []
        parts = path.split()
        code_map = {
            'M': Path.MOVETO,
            'C': Path.CURVE4,
            'L': Path.LINETO,
        }
        i = 0
        while i < len(parts) - 1:
            if parts[i] in code_map:
                path_code = code_map[parts[i]]
                code_len = 1
            else:
                path_code = code_map['l']
                code_len = 0
            npoints = Path.NUM_VERTICES_FOR_CODE[path_code]
            codes.extend([path_code] * npoints)
            vertices.extend([[*map(float, y.split(','))]
                            for y in parts[i+code_len:][:npoints]])
            i += npoints + code_len
        return vertices, codes

    def _path_to_vertices(self, path):
        vertices, _ = self._parse_path(path)
        vertices = self.scaled_vertices(vertices, scale=0.5)

        return [b2.b2Vec2(x, y) for x, y in vertices]

    def update_velocities(self, linear_velocity, angular_velocity):
        self.aircraft_body.linearVelocity = linear_velocity
        self.aircraft_body.angularVelocity = angular_velocity

    @property
    def position(self):
        return self.aircraft_body.position
    
    @property
    def angle(self):
        return self.aircraft_body.angle
    
    @property
    def linear_velocity(self):
        return self.aircraft_body.linearVelocity
    
    @property
    def angular_velocity(self):
        return self.aircraft_body.angularVelocity
    
    @property
    def angle(self):
        return self.aircraft_body.angle


    @linear_velocity.setter
    def linear_velocity(self, linear_velocity):
        self.aircraft_body.linearVelocity = linear_velocity

    @angular_velocity.setter
    def angular_velocity(self, angular_velocity):
        self.aircraft_body.angularVelocity = angular_velocity


    def scaled_vertices(self, vertices, scale=1):
        return [(x * scale, y * scale) for x, y in vertices]

    def create_aircraft_body(self, initial_position, initial_angle, initial_velocity):
   

        fuselage_vertices = self._path_to_vertices(PLANE_OUTLINE_PATH)
        main_wing_vertices = self._path_to_vertices(MAIN_WING_PATH)
        tail_plane_vertices = self._path_to_vertices(TAIL_PLANE_PATH)
        aircraft_vertices = fuselage_vertices + main_wing_vertices + tail_plane_vertices
        aircraft_vertices_flat = [coord for vertex in aircraft_vertices for coord in (vertex.x, vertex.y)]
        

        aircraft_body_def = b2.b2BodyDef()
        aircraft_body_def.type = b2.b2_dynamicBody
        aircraft_body_def.position = b2.b2Vec2(0, initial_position[1] )
        aircraft_body_def.angle = initial_angle
        aircraft_body_def.linearVelocity = initial_velocity
        aircraft_body_def.linear_damping = 0.0

        # Create the aircraft body first
        self.aircraft_body = self.world.CreateBody(aircraft_body_def)

        triangles_indices = earcut.earcut(aircraft_vertices_flat)

        for i in range(0, len(triangles_indices), 3):
            triangle_shape = b2.b2PolygonShape()
            triangle_shape.vertices = [
                    aircraft_vertices[triangles_indices[i]],
                    aircraft_vertices[triangles_indices[i + 1]],
                    aircraft_vertices[triangles_indices[i + 2]],
                ]
            

            # Create the fixture on the aircraft body inside the loop
            self.aircraft_body.CreateFixture(shape=triangle_shape)
class ObstacleGeometry():
    """A class to store and process the geometry of the obstacles"""

    def __init__(self, world=None, initial_position=(0, 0), wall_config=None, geom_scale = 1):
        self.world = world
        self.initial_ac_pos = initial_position
        self._wall_positions = []
        self._wall_heights = []
        self._wall_widths = []
        self.config = wall_config
        self.gap_height_params = self.config['gap_height']
        self.gap_width_params = self.config['gap_width']
        self.gap_offset_params = self.config['gap_offset']
        self.geom_scale = geom_scale

        self.create_obstacles()
       


    def define_placement(self):
        # first_wall_distance = random.uniform(MIN_FIRST_WALL_DISTANCE, MAX_FIRST_WALL_DISTANCE)
        first_wall_distance = self.config['first_wall_distance']
        # wall_positions = [first_wall_distance] + [random.uniform(MIN_WALL_DISTANCE + MIN_WALL_SPACING, MAX_WALL_DISTANCE) for _ in range(N_OBSTACLE_WALLS - 1)]
        wall_positions = [first_wall_distance] + [self.config["wall_distance"] for _ in range(self.config["n_obstacle_walls"] - 1)]
        cumulative_wall_positions = [sum(wall_positions[:i+1]) for i in range(len(wall_positions))]

        # print(cumulative_wall_positions)

        return cumulative_wall_positions, first_wall_distance
    
    def create_obstacles(self):
        cumulative_wall_positions, first_wall_distance = self.define_placement()
        for i in range(self.config["n_obstacle_walls"]):
            self.create_wall(cumulative_wall_positions, first_wall_distance, i)
    
    def create_wall(self, cumulative_wall_positions, first_wall_distance, i):
        # gap_height = random.uniform(MIN_GAP_HEIGHT, MAX_GAP_HEIGHT)
        # wall_gap_width = random.uniform(MIN_GAP_WIDTH, MAX_GAP_WIDTH)
        # gap_offset = random.uniform(MIN_GAP_OFFSET, MAX_GAP_OFFSET)
        gap_height = max(0.4, random.normal(self.gap_height_params[0], self.gap_height_params[1]))
        wall_gap_width = max(0.1,random.normal(self.gap_width_params[0], self.gap_width_params[1]))
        # wall_gap_width = self.gap_width_params[0]
        gap_offset = random.normal(self.gap_offset_params[0], self.gap_offset_params[1])
        # gap_height = random.uniform(self.gap_height_params[0], self.gap_height_params[1]) * GEOM_SCALE
        # wall_gap_width = random.uniform(self.gap_width_params[0], self.gap_width_params[1]) * GEOM_SCALE
        # gap_offset = random.uniform(-self.gap_offset_params[1], self.gap_offset_params[1]) * GEOM_SCALE
        
        gap_center_y = self.initial_ac_pos[1] + gap_offset

        # print(gap_height, wall_gap_width, gap_offset, gap_center_y)

        top_wall_y = gap_center_y + gap_height / 2
        wall_body_def = b2.b2BodyDef()
        wall_body_def.type = b2.b2_staticBody
        wall_body_def.position = b2.b2Vec2(INITIAL_WALL_OFFSET + cumulative_wall_positions[i], top_wall_y + (self.config["max_wall_height"] - gap_height) / 2)

        self._wall_positions.append((wall_body_def.position.x/self.geom_scale, gap_center_y/self.geom_scale))
        self._wall_heights.append(gap_height/self.geom_scale)
        self._wall_widths.append(wall_gap_width/self.geom_scale)

        top_wall_body = self.world.CreateBody(wall_body_def)
        wall_shape = b2.b2PolygonShape()
        wall_shape.SetAsBox(wall_gap_width / 2, (self.config["max_wall_height"] - gap_height) / 2)
        top_wall_body.CreateFixture(shape=wall_shape)

        # Bottom part of the wall
        bottom_wall_y = gap_center_y - gap_height / 2
        wall_body_def = b2.b2BodyDef()
        wall_body_def.type = b2.b2_staticBody
        wall_body_def.position = b2.b2Vec2(INITIAL_WALL_OFFSET + cumulative_wall_positions[i], bottom_wall_y - (self.config["max_wall_height"] - gap_height) / 2)

        bottom_wall_body = self.world.CreateBody(wall_body_def)
        wall_shape = b2.b2PolygonShape()
        wall_shape.SetAsBox(wall_gap_width / 2, (self.config["max_wall_height"] - gap_height) / 2)
        bottom_wall_body.CreateFixture(shape=wall_shape)

        # if i is last i
        if i == self.config["n_obstacle_walls"] - 1:
            # print(f"final wall{INITIAL_WALL_OFFSET + cumulative_wall_positions[i] + wall_gap_width}")
            # get last wall x + a little bit extra to account for plane fully passing through
            self._last_wall_x = (INITIAL_WALL_OFFSET + cumulative_wall_positions[i] + wall_gap_width + 10*self.geom_scale)/self.geom_scale
            # print(self._last_wall_x)

    @property
    def last_wall_x(self):
        return self._last_wall_x
    
    @property
    def wall_positions(self):
        return self._wall_positions
    
    @property
    def wall_heights(self):
        return self._wall_heights
    
    @property
    def wall_widths(self):
        return self._wall_widths
    
class Cloud:
    def __init__(self, x, y, world_to_screen, cloud_size=50):
        self.x = x 
        self.y = y
        self.cloud_size = cloud_size
        self.world_to_screen = world_to_screen
        self.ellipses = []

        for _ in range(5):
            offset_x = random.randint(-cloud_size // 2, cloud_size // 2)
            offset_y = random.randint(-cloud_size // 4, cloud_size // 4)
            ellipse_width = random.randint(cloud_size // 2, cloud_size)
            ellipse_height = random.randint(cloud_size // 4, cloud_size // 2)
            self.ellipses.append((offset_x, offset_y, ellipse_width, ellipse_height))

    def draw(self, screen, scroll_x, scroll_y):
        color = (255, 255, 255)  # white color for clouds


        screen_x = self.x
        screen_y = self.y

        screen_x, screen_y = self.world_to_screen((screen_x, screen_y))

        for ellipse in self.ellipses:
            pygame.draw.ellipse(screen, color, (screen_x + ellipse[0], screen_y + ellipse[1], ellipse[2], ellipse[3]))