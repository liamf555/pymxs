import random
import gym
from gym import spaces
import sys
from mxs_base import MxsEnv
import Box2D as b2
from earcut import earcut

import numpy as np
sys.path.insert(1, '/home/tu18537/dev/mxs/pymxs/models')

from pyaerso import AffectedBody, AeroBody, Body
from gym_mxs.model import Combined, calc_state, inertia, trim_points

# Constants for rendering
PPM = 5  # Pixels per meter (scaling factor)
WINDOW_WIDTH = 800
WINDOW_HEIGHT = 600
SCALE = 30.0 

# aircaft dimensions
AIRCRAFT_HEIGHT = 270 #mm
AIRCRAFT_LENGTH = 1000 #mm

# obstacle course params
MAX_WALL_HEIGHT = 75 # size of walls
N_OBSTACLE_WALLS = 5 # number of walls in obstacle course
MIN_GAP_HEIGHT = 10 # distance between wall halves
MAX_GAP_HEIGHT =15
MIN_WALL_DISTANCE = 30 # distance between walls
MAX_WALL_DISTANCE = 50 # distance between walls
MIN_GAP_WIDTH = 1 # width of walls 
MAX_GAP_WIDTH = 10
MIN_FIRST_WALL_DISTANCE = 20 # distance between plane and first wall
MAX_FIRST_WALL_DISTANCE = 30
MIN_WALL_SPACING = 5
MIN_GAP_OFFSET = -10.0
MAX_GAP_OFFSET = 10.0
INITIAL_WALL_OFFSET = 20
MIDDLE_Y = (WINDOW_HEIGHT / 2) / PPM


# LIDAR params
N_LIDAR_RAYS = 10
LIDAR_RANGE = 160/SCALE

# aircraft gemotry
PLANE_OUTLINE_PATH = "M -8.4344006,0.8833226 L -3.6174367,1.4545926 C -2.6957014,1.5861425 -1.2977255,1.7000225 -0.44895008,0.98453256 C 0.97534922,0.9358126 2.1554971,0.9295626 3.4694746,0.8473026 C 3.4694746,0.8473026 4.1040207,0.8167026 4.1204559,0.5018026 C 4.1306045,0.3072626 4.2764544,-1.2268074 1.7485665,-1.3031174 L 1.7604066,-1.0355474 L 1.3209316,-1.0233574 L 1.3822972,-1.7538274 C 1.9074643,-1.7412074 2.0141441,-2.5891474 1.4111688,-2.6446878 C 0.80819248,-2.7002378 0.8023354,-1.8387774 1.1839183,-1.7720774 L 1.0908357,-1.0522274 L -5.2189818,-0.91913738 L -12.198397,-0.80283738 C -12.198397,-0.80283738 -12.820582,-0.84082738 -12.643322,-0.31380735 C -12.466063,0.2132026 -11.622877,3.1026526 -11.622877,3.1026526 L -10.120232,3.1500026 C -10.120232,3.1500026 -9.8463164,3.1552526 -9.6753635,2.8748926 C -9.5044154,2.5944926 -8.4343678,0.8834126 -8.4343678,0.8834126 Z"
MAIN_WING_PATH="M 0.32346345,0.1815526 C 1.8962199,0.1638926 1.9691414,-0.33848735 0.34369001,-0.39724735 C -2.0368286,-0.46197735 -3.4920188,-0.15280735 -3.3975903,-0.13907735 C -1.5720135,0.1264326 -0.81500941,0.1943226 0.32346345,0.1815526 Z"
TAIL_PLANE_PATH="M -8.9838929,0.4470726 C -7.9395132,0.4475726 -7.8954225,0.0758826 -8.975461,0.01829265 C -10.557021,-0.05024735 -11.520801,0.1663226 -11.457966,0.1773326 C -10.24323,0.3898926 -9.739887,0.4467426 -8.9838897,0.4471126 Z"

class MxsEnvBox2D(MxsEnv):
    metadata = {
        "render_modes": ["ansi"],
        "render_fps": 4
    }

    def __init__(self, render_mode=None, reward_func=lambda obs: 0.5, timestep_limit=100, scenario=None, **kwargs):
   

        self.steps = 0

        self.reward_func = reward_func
        self.reward_state = None

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode
        self.scenario = scenario

        # create bx2d world 
        world = b2.b2World()

        # make pygame if render mode is not None
        if self.render_mode is not None:
            import pygame
            pygame.init()
            self.scroll = 0
            self.render_screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT), 0, 32)
            pygame.display.set_caption("Box2D World Rendering")
            self.surf = pygame.Surface(
            (WINDOW_HEIGHT + max(0.0, self.scroll) * SCALE, WINDOW_WIDTH)
            )
            pygame.transform.scale(self.surf, (SCALE, SCALE))
            self.clock = pygame.time.Clock()
            
        # make aircraft
        self.aircraft_body = MXS()


        # make obstacles (flappy plane for now)

        # lidars and collison stuff




    def _get_obs(self):
        return np.array(self.vehicle.statevector)

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
    
    def update_lidar(self):
        pos = self.aircraft_body.position
        angle = self.aircraft_body.angle
        offset_vector = b2.b2Vec2(4, 0)  # Define the offset vector for the front of the aircraft

        # Rotate the offset vector according to the body's angle
        rotated_offset_vector = b2.b2Vec2(
            offset_vector.x * math.cos(angle) - offset_vector.y * math.sin(angle),
            offset_vector.x * math.sin(angle) + offset_vector.y * math.cos(angle),
        )

        # Calculate the starting point of the lidar beams
        lidar_start_point = pos + rotated_offset_vector

        for i in range(10):
            self.lidars[i].fraction = 1.0
            angle_offset = (1.5 * i / 10.0) - 5.5  # Adjust the starting angle of the lidar beams
            start_angle = angle + angle_offset
            end_angle = angle + 1.5 * i / 10.0

            lidars[i].p1 = lidar_start_point
            lidars[i].p2 = (
                pos[0] + math.sin(start_angle) * LIDAR_RANGE,
                pos[1] - math.cos(start_angle) * LIDAR_RANGE,
            )
            world.RayCast(lidars[i], lidars[i].p1, lidars[i].p2)
    

    
    def reset_world(self):
        self.world = None

        world = b2.b2World()

        self.create_aircraft_and_obstacles()

        contact_listener = self.ContactListener()
        self.world.contactListener = contact_listener

        class LidarCallback(b2.b2RayCastCallback):
                def ReportFixture(self, fixture, point, normal, fraction):
                    if (fixture.filterData.categoryBits & 1) == 0:
                        return -1
                    self.p2 = point
                    self.fraction = fraction
                    return fraction

        self.lidars = [LidarCallback() for _ in range(10)]

    def create_aircraft_and_obstacles(self):
        self.aircraft_body = self.create_aircraft()
        self.create_obstacles()

    
    class ContactListener(b2.b2ContactListener):
        def __init__(self):
            super().__init__()
            self.collision_detected = False

        def BeginContact(self, contact):
            body_a, body_b = contact.fixtureA.body, contact.fixtureB.body
            if (body_a == self.aircraft_body and body_b.type == b2.b2_staticBody) or (body_a.type == b2.b2_staticBody and body_b == self.aircraft_body):
                self.collision_detected = True

    def render(self, mode):
        if mode == "ansi" or mode == "human":
            elements = ", ".join([f"{v:.{4}f}" for v in self._get_obs()])
            return f"[{elements},{self.vehicle.airstate[0]},{self.vehicle.airstate[2]},{self.elevator},{self.throttle}]"

    def close(self):
        pass


class Box2DGeometry:

    def __init__(self, world):
        self.world = world

    def _parse_path(path):
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
        return [b2.b2Vec2(x, y) for x, y in vertices]


class MXSBody(Box2DGeometry):
    """A class to store and process the geometry of the MXS aircraft"""

    PLANE_OUTLINE_PATH = "M -8.4344006,0.8833226 L -3.6174367,1.4545926 C -2.6957014,1.5861425 -1.2977255,1.7000225 -0.44895008,0.98453256 C 0.97534922,0.9358126 2.1554971,0.9295626 3.4694746,0.8473026 C 3.4694746,0.8473026 4.1040207,0.8167026 4.1204559,0.5018026 C 4.1306045,0.3072626 4.2764544,-1.2268074 1.7485665,-1.3031174 L 1.7604066,-1.0355474 L 1.3209316,-1.0233574 L 1.3822972,-1.7538274 C 1.9074643,-1.7412074 2.0141441,-2.5891474 1.4111688,-2.6446878 C 0.80819248,-2.7002378 0.8023354,-1.8387774 1.1839183,-1.7720774 L 1.0908357,-1.0522274 L -5.2189818,-0.91913738 L -12.198397,-0.80283738 C -12.198397,-0.80283738 -12.820582,-0.84082738 -12.643322,-0.31380735 C -12.466063,0.2132026 -11.622877,3.1026526 -11.622877,3.1026526 L -10.120232,3.1500026 C -10.120232,3.1500026 -9.8463164,3.1552526 -9.6753635,2.8748926 C -9.5044154,2.5944926 -8.4343678,0.8834126 -8.4343678,0.8834126 Z"
    MAIN_WING_PATH="M 0.32346345,0.1815526 C 1.8962199,0.1638926 1.9691414,-0.33848735 0.34369001,-0.39724735 C -2.0368286,-0.46197735 -3.4920188,-0.15280735 -3.3975903,-0.13907735 C -1.5720135,0.1264326 -0.81500941,0.1943226 0.32346345,0.1815526 Z"
    TAIL_PLANE_PATH="M -8.9838929,0.4470726 C -7.9395132,0.4475726 -7.8954225,0.0758826 -8.975461,0.01829265 C -10.557021,-0.05024735 -11.520801,0.1663226 -11.457966,0.1773326 C -10.24323,0.3898926 -9.739887,0.4467426 -8.9838897,0.4471126 Z"    

    def __init__(self, render_mode="human", initial_position=(0, 0), initial_angle=0):
        self.aircraft_body = self._create_body(initial_position, initial_angle)


    def _create_shape(self):
        fuselage_vertices = self.path_to_vertices(PLANE_OUTLINE_PATH)
        main_wing_vertices = self.path_to_vertices(MAIN_WING_PATH)
        tail_plane_vertices = self.path_to_vertices(TAIL_PLANE_PATH)
        aircraft_vertices = fuselage_vertices + main_wing_vertices + tail_plane_vertices
        aircraft_vertices_flat = [coord for vertex in aircraft_vertices for coord in (vertex.x, vertex.y)]
        triangles_indices = earcut.earcut(aircraft_vertices_flat)

        for i in range(0, len(triangles_indices), 3):
            triangle_shape = b2.b2PolygonShape()
            triangle_shape.vertices = [
                aircraft_vertices[triangles_indices[i]],
                aircraft_vertices[triangles_indices[i + 1]],
                aircraft_vertices[triangles_indices[i + 2]],
            ]

        return triangle_shape

    def _create_body(self, initial_position, initial_angle):
        triangle_shape = self._create_shape()
        aircraft_body_def = b2.b2BodyDef()
        aircraft_body_def.type = b2.b2_dynamicBody
        aircraft_body_def.position = b2.b2Vec2(0, MIDDLE_Y )
        aircraft_body_def.angle = initial_angle # rads
        aircraft_body = self.world.CreateBody(aircraft_body_def)
        aircraft_body.CreateFixture(shape=triangle_shape)

        return aircraft_body

    def update_velocities(self, linear_velocity, angular_velocity):
        self.aircraft_body.linearVelocity = linear_velocity
        self.aircraft_body.angularVelocity = angular_velocity

    def get_position(self):
        return self.aircraft_body.position

class ObstacleGeometry(Box2DGeometry):
    """A class to store and process the geometry of the obstacles"""

    def __init__(self, render_mode="human"):
        self.obstacle_body = self._create_body()

    def _create_shape(self):
        obstacle_shape = b2.b2PolygonShape()
        obstacle_shape.SetAsBox(0.5, 0.5)
        return obstacle_shape

    def _create_body(self):
        obstacle_shape = self._create_shape()
        obstacle_body_def = b2.b2BodyDef()
        obstacle_body_def.type = b2.b2_staticBody
        obstacle_body_def.position = b2.b2Vec2(0, 0)
        obstacle_body = self.world.CreateBody(obstacle_body_def)
        obstacle_body.CreateFixture(shape=obstacle_shape)

    def define_placement(self):
        first_wall_distance = random.uniform(MIN_FIRST_WALL_DISTANCE, MAX_FIRST_WALL_DISTANCE)
        wall_positions = [first_wall_distance] + [random.uniform(MIN_WALL_DISTANCE + MIN_WALL_SPACING, MAX_WALL_DISTANCE) for _ in range(N_OBSTACLE_WALLS - 1)]
        cumulative_wall_positions = [sum(wall_positions[:i+1]) for i in range(len(wall_positions))]

        return cumulative_wall_positions
    
    def create_obstacles(self):
        cumulative_wall_positions = self.define_placement()
        for i in range(N_OBSTACLE_WALLS):
            self.create_wall(cumulative_wall_positions, i)
    
    def create_wall(self, cumulative_wall_positions, i):
        gap_height = random.uniform(MIN_GAP_HEIGHT, MAX_GAP_HEIGHT)
        wall_gap_width = random.uniform(MIN_GAP_WIDTH, MAX_GAP_WIDTH)
        gap_offset = random.uniform(MIN_GAP_OFFSET, MAX_GAP_OFFSET)
        gap_center_y = self.aircraft_initial_y + gap_offset

        top_wall_y = gap_center_y + gap_height / 2
        wall_body_def = b2.b2BodyDef()
        wall_body_def.type = b2.b2_staticBody
        wall_body_def.position = b2.b2Vec2(INITIAL_WALL_OFFSET + cumulative_wall_positions[i], top_wall_y + (MAX_WALL_HEIGHT - gap_height) / 2)

        top_wall_body = self.world.CreateBody(wall_body_def)
        wall_shape = b2.b2PolygonShape()
        wall_shape.SetAsBox(wall_gap_width / 2, (MAX_WALL_HEIGHT - gap_height) / 2)
        top_wall_body.CreateFixture(shape=wall_shape)

        # Bottom part of the wall
        bottom_wall_y = gap_center_y - gap_height / 2
        wall_body_def = b2.b2BodyDef()
        wall_body_def.type = b2.b2_staticBody
        wall_body_def.position = b2.b2Vec2(INITIAL_WALL_OFFSET + cumulative_wall_positions[i], bottom_wall_y - (MAX_WALL_HEIGHT - gap_height) / 2)

        bottom_wall_body = self.world.CreateBody(wall_body_def)
        wall_shape = b2.b2PolygonShape()
        wall_shape.SetAsBox(wall_gap_width / 2, (MAX_WALL_HEIGHT - gap_height) / 2)
        bottom_wall_body.CreateFixture(shape=wall_shape)







 #  get state from model and convert for box2d

 # set linear and angular velocities for bod2d 

# get lidar data from box2d

# get collision stuff from box2d

# current position?? box2d vs MXS model (maybe some scaling playing has to happen here)

