import math
from matplotlib.path import Path
import random
try:
    import gymnasium as gym
    from gymnasium import spaces
    GYM_FLAG = False
except:
    import gym
    from gym import spaces
    GYM_FLAG = True
import sys
from . import MxsEnv
import Box2D as b2
import wandb
from earcut import earcut
import pygame
from pygame.locals import K_p, KEYDOWN, QUIT
import numpy as np
sys.path.insert(1, '/home/tu18537/dev/mxs/pymxs/models')

from pyaerso import AffectedBody, AeroBody, Body
from gym_mxs.model import Combined, calc_state, inertia, trim_points

from scipy.spatial.transform import Rotation

from dm_control.utils import rewards


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

# MIDDLE_Y = (WINDOW_HEIGHT / 2) / SCALE

  # size of walls

# LIDAR params
#reward params

PLANE_OUTLINE_PATH = "M -8.4344006,0.8833226 L -3.6174367,1.4545926 C -2.6957014,1.5861425 -1.2977255,1.7000225 -0.44895008,0.98453256 C 0.97534922,0.9358126 2.1554971,0.9295626 3.4694746,0.8473026 C 3.4694746,0.8473026 4.1040207,0.8167026 4.1204559,0.5018026 C 4.1306045,0.3072626 4.2764544,-1.2268074 1.7485665,-1.3031174 L 1.7604066,-1.0355474 L 1.3209316,-1.0233574 L 1.3822972,-1.7538274 C 1.9074643,-1.7412074 2.0141441,-2.5891474 1.4111688,-2.6446878 C 0.80819248,-2.7002378 0.8023354,-1.8387774 1.1839183,-1.7720774 L 1.0908357,-1.0522274 L -5.2189818,-0.91913738 L -12.198397,-0.80283738 C -12.198397,-0.80283738 -12.820582,-0.84082738 -12.643322,-0.31380735 C -12.466063,0.2132026 -11.622877,3.1026526 -11.622877,3.1026526 L -10.120232,3.1500026 C -10.120232,3.1500026 -9.8463164,3.1552526 -9.6753635,2.8748926 C -9.5044154,2.5944926 -8.4343678,0.8834126 -8.4343678,0.8834126 Z"
MAIN_WING_PATH="M 0.32346345,0.1815526 C 1.8962199,0.1638926 1.9691414,-0.33848735 0.34369001,-0.39724735 C -2.0368286,-0.46197735 -3.4920188,-0.15280735 -3.3975903,-0.13907735 C -1.5720135,0.1264326 -0.81500941,0.1943226 0.32346345,0.1815526 Z"
TAIL_PLANE_PATH="M -8.9838929,0.4470726 C -7.9395132,0.4475726 -7.8954225,0.0758826 -8.975461,0.01829265 C -10.557021,-0.05024735 -11.520801,0.1663226 -11.457966,0.1773326 C -10.24323,0.3898926 -9.739887,0.4467426 -8.9838897,0.4471126 Z"


class MxsEnvBox2DLand(MxsEnv):
    def __init__(self, render_mode, training, use_lidar=True, acl = False, context = np.array([0.,0.,0.]), config=None, output_dir=None):
        self.render_mode = render_mode
        super().__init__(render_mode= render_mode)

        self.acl_on = acl

        self.config = config

        self.obstacle_config = self.config['obstacles']
        self.geom_scale = self.config['geometry']['geom_scale']
        self.obstacle_config = multiply_dict_values(self.obstacle_config, self.geom_scale, excluded_keys=[])
        print(self.obstacle_config)
        self.lidar_config = self.config["lidars"]
        self.aircraft_alt = self.config['geometry']['aircraft_alt'] * self.geom_scale

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
        self.position_counter = 0

        self.reward_state = None

        self.use_lidar = use_lidar

        self.n_lidar_rays = self.config["lidars"]["n_lidar_rays"]

        if self.use_lidar is True:
            n_obs = 8 + self.n_lidar_rays
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

        # print(f"action space: {self.action_space}")
        # breakpoint()

        self.initial_ac_pos = (0, (self.window_dims[1] / 2) / self.scale)
        self.render_screen = None
        self.clock = None
        self.x_pos_prev = 0
        self.last_obstacle_x = 0
        self.termination = False
        self.n_episodes = 0


    
    def _get_full_obs(self, obs):
        # obs = self.long_obs(obs)

        position = (self.aircraft.position - self.initial_ac_pos) / self.geom_scale
        position = (position[0], position[1] + self.aircraft_alt / self.geom_scale)
        # print(f"Position: {position}")
        velocity = self.aircraft.linear_velocity / self.geom_scale
        angle = self.aircraft.angle
        angular_velocity = self.aircraft.angular_velocity

        if self.use_lidar:
            wall_obs = [l.fraction for l in self.lidars]
        else:
            gap_pos = self.calc_gap_position_delta()
            gap_height = self.get_wall_height()
            gap_width = self.get_gap_width()
            wall_obs = [gap_pos, gap_height, gap_width]
        # print(f"Wall obs: {wall_obs}")
        # print(f"Position: {position}")
        # print("Wall position delta: ", wall_pos)
        observation = np.array([*position, *velocity, angle, angular_velocity, *wall_obs, self.elevator, self.throttle], dtype=np.float32)
        return observation
    
    def obstacle_avoid_box2d(self, obs):
        # x, z, u, w, pitch, pitch_rate, gap_pos, gap_height, gap_width, elevator, throttle
        proximity_weight = 0.5
        angle_limit = 5 
        velocity_limit = 5
        # print(f"obs: {obs}")

        # print(f"obs: {obs}")
        x_pos = obs[0]
        alt = obs[1]
        u_vel = obs[2]
        w_vel = obs[3]
        pitch = self.aircraft.angle
        pitch_degrees = math.degrees(pitch)
        obstacle_centre_x = obs[6]
        

        # print(f"Pitch: {pitch_degrees}")
        # print(f"Velocity: {np.sqrt(u_vel**2 + w_vel**2)}")
        # print(f"Alt: {alt}")
        # print(f"X pos: {x_pos}")
        # print(f"Obstacle centre x: {obstacle_centre_x}")


        velocity = np.sqrt(u_vel**2 + w_vel**2)

        has_landed = alt <= 0.1

        if has_landed:
            if abs(pitch_degrees) < angle_limit and velocity < velocity_limit:
                self.success = True
                self.completion_counter += 1
            self.termination = True

        

        distance_to_centre = abs(x_pos - obstacle_centre_x)
        distance_reward = 1 / (distance_to_centre + 1)

        pitch_factor = max(0, 1 - abs(pitch_degrees) / angle_limit)
        velocity_factor = max(0, 1 - velocity / velocity_limit)

        landing_quality = pitch_factor * velocity_factor
        
        reward = proximity_weight * distance_reward + (1 - proximity_weight) * landing_quality
        
        if self.game_over():
            # print("Hit wall")
            reward = -1000
            self.termination = True
            self.wall_counter += 1

        if abs(math.degrees(pitch)) > 170:
            # print("Too pitchy")
            reward = -1000
            self.termination = True
            self.pitch_counter += 1
        # print(f"Reward {reward}")
        if u_vel > 24.5:
            # print("Too fast")
            reward = -1000
            self.termination = True
            self.speed_counter += 1
        if abs(alt) > (self.aircraft_alt /self.geom_scale) + 10 or x_pos > (self.get_gap_position() + 10):
            # print("Too high")
            reward = -1000
            self.termination = True
            self.alt_counter += 1

        if self.termination and self.training:
            wandb.log({"Terminal position": x_pos})
            wandb.log({"Terminal velocity": velocity})
            wandb.log({"Terminal pitch": pitch_degrees})
            wandb.log({"Terminal distance to centre": distance_to_centre})
            wandb.log({"Wall counter": self.wall_counter})
            wandb.log({"Pitch counter": self.pitch_counter})
            wandb.log({"Speed counter": self.speed_counter})
            wandb.log({"Position counter": self.alt_counter})
            wandb.log({"Completion counter": self.completion_counter})
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
    
    def obstacle_avoid_box2d_shaped(self, obs):
        # x, z, u, w, pitch, pitch_rate, gap_pos, gap_height, gap_width, elevator, throttle

        x_pos = obs[0]
        alt = obs[1]
        # print(f"Alt: {alt}")
        u_vel = obs[2]
        w_vel = obs[3]
        pitch = self.aircraft.angle
        pitch_degrees = math.degrees(pitch)
        pitch_rate = np.degrees(obs[5])
        obstacle_centre_x = self.calc_gap_position_delta()
        gap_width = self.gap_width

        pitch_error_degrees = abs(pitch_degrees)  
        alt_error = abs(alt) 
        x_error = obstacle_centre_x  
        velocity_error = np.sqrt(u_vel**2 + w_vel**2)

        # print(f"obstacle_centre_x: {obstacle_centre_x}")
        # print(f"alt_error: {alt_error}")
        # print(f"velocity_error: {velocity_error}")
        # print(f"x_error: {x_error}")
        # print(f"pitch_error_degrees: {pitch_error_degrees}")

        # print(f"Pitch error: {pitch_error_degrees}")
        # print(f"Alt error: {alt_error}")

        # print(f"X error: {obstacle_centre_x}")
        # print(f"Velocity error: {velocity_error}")
        # print(obstacle_centre_x)
        # print(alt_error)
        reward = 0
        shaping = (
            - 100 * np.sqrt(obstacle_centre_x**2  + alt_error**2)
            - 10 * velocity_error
        )

        if self.prev_shaping is not None:
            reward = shaping - self.prev_shaping
        self.prev_shaping = shaping

        gap_min = self.gap_centre_x - self.gap_width / 2
        gap_max = self.gap_centre_x + self.gap_width / 2
        final_reward = 0

        pos_r = 0
        vel_r = 0
        pitch_r = 0

        # minimise throttle? or maybe time? 
        if alt < 0.1:
            self.termination = True
            reward = -5000
            if (gap_min < x_pos < gap_max):
                # print("Success")
                self.completion_counter += 1
                self.success = True
            
                pos_r = rewards.tolerance(x_error, bounds=(0, 0.5), margin=2)
                vel_r = rewards.tolerance(velocity_error, bounds=(0, 2.5), margin=20)
                pitch_r = rewards.tolerance(pitch_error_degrees, bounds=(0, 5), margin=90)
                # alt_r = rewards.tolerance(alt_error, bounds=(0, 1), margin=10)

                reward = (pos_r * vel_r * pitch_r) * 10000
                final_reward = reward

        if self.game_over():
            # print("Hit wall")
            reward = - 5000
            self.termination = True
            self.wall_counter += 1

        if abs(pitch_degrees) > 180:
            # print("Too pitchy")
            reward = -5000
            self.termination = True
            self.pitch_counter += 1
        # print(f"Reward {reward}")
        if u_vel > 24.5:
            # print("Too fast")
            reward = -5000
            self.termination = True
            self.speed_counter += 1
        if alt > (self.aircraft_alt /self.geom_scale) + 25:
            # print("Too high or too far")
            reward = -5000
            self.termination = True
            self.alt_counter += 1
        if x_pos > (self.get_gap_position() + 25):
            # print("Too far")
            reward = -5000
            self.termination = True
            self.position_counter += 1

        if self.termination and self.training:
            # print(pitch_degrees)
            wandb.log({"Terminal position": x_pos})
            wandb.log({"Terminal velocity": velocity_error})
            wandb.log({"Terminal pitch": pitch_degrees})
            wandb.log({"Terminal distance to centre": x_error})
            wandb.log({"Wall counter": self.wall_counter})
            wandb.log({"Pitch counter": self.pitch_counter})
            wandb.log({"Speed counter": self.speed_counter})
            wandb.log({"Alt counter": self.alt_counter})
            wandb.log({"Position counter": self.position_counter})
            wandb.log({"Completion counter": self.completion_counter})
            wandb.log({"Final reward": final_reward})
            wandb.log({"Position reward": pos_r})
            wandb.log({"Velocity reward": vel_r})
            wandb.log({"Pitch reward": pitch_r})

           
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
    
    def draw_world(self):
        # Draw clouds
        [cloud.draw(self.render_screen, self.scroll_offset, 0) for cloud in self.clouds]
    # Draw bodies
        for body in self.world.bodies:
            for fixture in body.fixtures:
                if type(fixture.shape) == b2.b2PolygonShape:
                    color = (150,75,0)  # Default color: blue
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
        # draw floor bwtweeen blocks
        gap_centre_x = self.get_gap_position() * self.geom_scale
        gap_width = self.get_gap_width() * self.geom_scale
        # get floor of world by using airctaft starting alt
        floor_y = -10*self.geom_scale

        floor_position_screen = self.world_to_screen([gap_centre_x - gap_width / 2, floor_y])
        pygame.draw.line(self.render_screen, (0, 255, 0), floor_position_screen, (floor_position_screen[0] + gap_width * self.geom_scale, floor_position_screen[1]), 1)

        # else:
        #     if self.wall_i < len(self.wall_positions):
        #     # Draw a cross representing the cetnre of teh next obstacle
        #         next_wall_position = [p * GEOM_SCALE for p in self.get_next_wall_position()]
        #         # print(f"Next wall position: {next_wall_position}")
        #         next_wall_position_screen = self.world_to_screen(next_wall_position)
        #         pygame.draw.line(self.render_screen, (0, 0, 255), (next_wall_position_screen[0] - 10, next_wall_position_screen[1]), (next_wall_position_screen[0] + 10, next_wall_position_screen[1]), 1)

        #         # Draw a vertical line
        #         pygame.draw.line(self.render_screen, (0, 0, 255), (next_wall_position_screen[0], next_wall_position_screen[1] - 10), (next_wall_position_screen[0], next_wall_position_screen[1] + 10), 1)
    
    def calc_gap_position_delta(self):
        """ A function to calculate the position delta between the aircraft and the next obstacle """
        aircraft_position = self.aircraft.position/self.geom_scale 
        gap_position = self.get_gap_position()
        # print(f"Wall position: {next_obstacle_position}")
        delta_x = gap_position - aircraft_position[0]
        # print(f"Delta x: {delta_x}")

        return (delta_x)
    
    def get_gap_width(self):
        """ A function to calculate the width of the gap in the next obstacle """
        return self.obstacle_geom.gap_width
    
    def get_gap_position(self):
        """A method to return the position of the next obstacle relative to the aircraft,
        based on the aicraft's current position"""
     
        return self.obstacle_geom.gap_position
    
    def get_wall_height(self):
        """A method to return the height of the next obstacle relative to the aircraft,
        based on the aicraft's current position"""

        return self.obstacle_geom.wall_height
    
    def create_aircraft_and_obstacles(self):
        self.aircraft = MXSGeometry(self.world, initial_position=self.initial_ac_pos, initial_angle=self.aircraft_initial_angle, initial_velocity=self.aircraft_initial_velocity)
        self.obstacle_geom = ObstacleGeometry(self.world, initial_position=self.initial_ac_pos, obstacle_config=self.obstacle_config, geom_scale=self.geom_scale, initial_alt=self.config['geometry']['aircraft_alt'])
        # self.last_obstacle_x = self.obstacle_geom.last_wall_x
        # print(f"Wall positions: {obstacle_geom.wall_positions}")
        # print(f"Wall heights: {obstacle_geom.wall_heights}")
        # print(f"Wall widths: {obstacle_geom.wall_widths}")
        # self.wall_positions = self.obstacle_geom.wall_positions
        self.gap_centre_x = self.obstacle_geom.gap_position
        self.gap_width = self.obstacle_geom.gap_width
        # self.wall_heights = self.obstacle_geom.wall_heights
        # self.wall_widths = self.obstacle_geom.wall_widths
        # self.x_pos_prev = 0

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

    def reset(self, seed=None, return_info=GYM_FLAG, options=None):
        self.vehicle.statevector = [
            *self.initial_state[0],
            *self.initial_state[1],
            *self.initial_state[2],
            *self.initial_state[3]
        ]

        self.elevator = np.radians(trim_points[self.selected_trim_point][1])
        self.throttle = trim_points[self.selected_trim_point][2]

        observation = self._get_obs()

        self.aircraft_initial_angle = self.get_pitch(*self.initial_state[2])
        self.aircraft_initial_velocity = self.get_velocity(observation)[0]
        self.wall_i = 0
        self.reset_world()
        observation = self._get_full_obs(observation)

        self.steps = 0
        self.reward_state = None
        self.scroll_offset = 0
        self.prev_shaping = None

        
        self.termination = False
        self.success = False

        info = {}

        if self.render_mode == "human":
            self._render_frame()
        return (observation,info) if not return_info else observation
 
    def step(self, action):
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
        # print(observation)

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
        if abs(np.degrees(self.aircraft.angle)) > 360:
            print(f"pitch: {np.degrees(self.aircraft.angle)}")
            print(f"pitch 2 {self.pitch}")

        if self.use_lidar:
            self.update_lidar()

        observation = self._get_full_obs(observation)
        # print(f"observation: {observation}")

        reward = self.obstacle_avoid_box2d_shaped(observation)
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

        for i in range(self.n_lidar_rays):
            self.lidars[i].fraction = 1.0
            angle_offset = (0.5 * i / 10.0) - 5.0  # Adjust the starting angle of the lidar beams
            # angle_offset = (1 * i / N_LIDAR_RAYS) - 5.25
            # angle_offset = (ANGLE_INCREMENT * i) - (LIDAR_FOV / 2)
            start_angle = angle + angle_offset
            # end_angle = angle + 1.5 * i / 10.0 # why these magic numbers?

            self.lidars[i].p1 = lidar_start_point
            self.lidars[i].p2 = (
                pos[0] + math.sin(start_angle) * self.lidar_config["lidar_range"] * self.geom_scale,
                pos[1] - math.cos(start_angle) * self.lidar_config["lidar_range"] * self.geom_scale,
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

        self.lidars = [LidarCallback() for _ in range(self.n_lidar_rays)]
        self.update_lidar()


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
        if self.render_mode == "ansi" or self.render_mode == "human":
            elements = ", ".join([f"{v:.{4}f}" for v in self._get_obs()])
            return f"[{elements},{self.vehicle.airstate[0]},{self.vehicle.airstate[2]},{self.elevator},{self.throttle}]"
        
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

    def lowest_point(self):
        low_point = self.fuslelage_vertices[0]
        for vertex_group in [self.fuselage_vertices, self.main_wing_vertices, self.tail_plane_vertices]:
            for vertex in vertex_group:
                if vertex.y > low_point.y:
                    low_point = vertex

        return low_point
    
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

    def __init__(self, world=None, render_mode="human", initial_position=(0, 0),  obstacle_config=None, geom_scale=1, initial_alt=0):
        self.world = world
        self.initial_ac_pos = initial_position
        self._gap_position = 0
        self._wall_height = 0
        self._gap_width = 0
        self.geom_scale = geom_scale
        self.block_config = obstacle_config
        self.aircraft_initial_alt = initial_alt * geom_scale
        self.create_obstacles()
        
       
    def create_obstacles(self):
        # cumulative_wall_positions, first_wall_distance = self.define_placement()
        height = random.uniform(self.block_config['block_height'][0], self.block_config['block_height'][1])
        separation = random.uniform(self.block_config['block_separation'][0],self.block_config['block_separation'][1])
        for i in range(2):
            self.create_wall(i, height, separation)
    
    def create_wall(self, i, landing_block_height, block_separation):

        landing_block_width = self.block_config['block_width']
        # landing_block_height = self.block_config['block_height']
        
        landing_block_y = self.initial_ac_pos[1] - self.aircraft_initial_alt + landing_block_height / 2
        # block_separation = self.block_config['block_separation']
        
        initial_wall_offset = self.block_config['initial_wall_offset']

        block_body_def = b2.b2BodyDef()
        block_body_def.type = b2.b2_staticBody
        block_body_def.position = b2.b2Vec2(initial_wall_offset + (i * (block_separation + landing_block_width)), landing_block_y)
        
        block_body = self.world.CreateBody(block_body_def)
        block_shape = b2.b2PolygonShape()
        block_shape.SetAsBox(landing_block_width / 2, landing_block_height / 2)
        block_body.CreateFixture(shape=block_shape)

        left_block_x = initial_wall_offset
        right_block_x = initial_wall_offset + block_separation + landing_block_width
        gap_center_x = (left_block_x + right_block_x) / 2
        gap_center_y = landing_block_y

        self._gap_position = gap_center_x
        self._wall_height = landing_block_height
        self._gap_width = block_separation

        # print(gap_center_x, gap_center_y)

        # print(gap_height, wall_gap_width, gap_offset, gap_center_y)


    @property
    def last_wall_x(self):
        return self._last_wall_x
    
    @property
    def gap_position(self):
        return self._gap_position / self.geom_scale
    
    @property
    def wall_height(self):
        return self._wall_height / self.geom_scale
    
    @property
    def gap_width(self):
        return self._gap_width / self.geom_scale
    
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

class Wind:
    def __init__(self, x, y, world_to_screen, wind_size=50):
        self.x = x 
        self.y = y
        self.wind_size = wind_size
        self.world_to_screen = world_to_screen
        self.ellipses = []

        for _ in range(5):
            offset_x = random.randint(-wind_size // 2, wind_size // 2)
            offset_y = random.randint(-wind_size // 4, wind_size // 4)
            ellipse_width = random.randint(wind_size // 2, wind_size)
            ellipse_height = random.randint(wind_size // 4, wind_size // 2)
            self.ellipses.append((offset_x, offset_y, ellipse_width, ellipse_height))

    def draw(self, screen, scroll_x, scroll_y):
        color = (255, 255, 255)  # white color for clouds


        screen_x = self.x
        screen_y = self.y

        screen_x, screen_y = self.world_to_screen((screen_x, screen_y))

        for ellipse in self.ellipses:
            pygame.draw.ellipse(screen, color, (screen_x + ellipse[0], screen_y + ellipse[1], ellipse[2], ellipse[3]))
