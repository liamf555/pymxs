import math
import random
import Box2D as b2
import time 
import pygame
from matplotlib.path import Path
from earcut import earcut

# Constants for rendering
PPM = 5  # Pixels per meter (scaling factor)
WINDOW_WIDTH = 800
WINDOW_HEIGHT = 600

# LIDAR Constants


# Obstacle 
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


# first_wall_distance = random.uniform(MIN_FIRST_WALL_DISTANCE, MAX_FIRST_WALL_DISTANCE)

middle_y = (WINDOW_HEIGHT / 2) / PPM

scroll = 0
SCALE = 50
lidars=[]
lidar_render=0
LIDAR_RANGE = 1000/SCALE

PLANE_OUTLINE_PATH = "M -8.4344006,0.8833226 L -3.6174367,1.4545926 C -2.6957014,1.5861425 -1.2977255,1.7000225 -0.44895008,0.98453256 C 0.97534922,0.9358126 2.1554971,0.9295626 3.4694746,0.8473026 C 3.4694746,0.8473026 4.1040207,0.8167026 4.1204559,0.5018026 C 4.1306045,0.3072626 4.2764544,-1.2268074 1.7485665,-1.3031174 L 1.7604066,-1.0355474 L 1.3209316,-1.0233574 L 1.3822972,-1.7538274 C 1.9074643,-1.7412074 2.0141441,-2.5891474 1.4111688,-2.6446878 C 0.80819248,-2.7002378 0.8023354,-1.8387774 1.1839183,-1.7720774 L 1.0908357,-1.0522274 L -5.2189818,-0.91913738 L -12.198397,-0.80283738 C -12.198397,-0.80283738 -12.820582,-0.84082738 -12.643322,-0.31380735 C -12.466063,0.2132026 -11.622877,3.1026526 -11.622877,3.1026526 L -10.120232,3.1500026 C -10.120232,3.1500026 -9.8463164,3.1552526 -9.6753635,2.8748926 C -9.5044154,2.5944926 -8.4343678,0.8834126 -8.4343678,0.8834126 Z"
MAIN_WING_PATH="M 0.32346345,0.1815526 C 1.8962199,0.1638926 1.9691414,-0.33848735 0.34369001,-0.39724735 C -2.0368286,-0.46197735 -3.4920188,-0.15280735 -3.3975903,-0.13907735 C -1.5720135,0.1264326 -0.81500941,0.1943226 0.32346345,0.1815526 Z"
TAIL_PLANE_PATH="M -8.9838929,0.4470726 C -7.9395132,0.4475726 -7.8954225,0.0758826 -8.975461,0.01829265 C -10.557021,-0.05024735 -11.520801,0.1663226 -11.457966,0.1773326 C -10.24323,0.3898926 -9.739887,0.4467426 -8.9838897,0.4471126 Z"

def parse_path(path):
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


def draw_polygon(polygon, body, color):
    vertices = [(body.transform * v) * PPM for v in polygon.vertices]
    vertices = [(v[0], WINDOW_HEIGHT - v[1]) for v in vertices]
    pygame.draw.polygon(screen, color, vertices)

def draw_world(world):
    # Draw bodies
    for body in world.bodies:
        for fixture in body.fixtures:
            if type(fixture.shape) == b2.b2PolygonShape:
                color = (0, 0, 255)  # Default color: blue
                if body == aircraft_body:  # If the body is the aircraft, set the color to green
                    color = (0, 255, 0)
                draw_polygon(fixture.shape, body, color)

    # Draw lidar rays
    for lidar in lidars:
        if hasattr(lidar, "p1") and hasattr(lidar, "p2"):
            start_screen = world_to_screen(lidar.p1)
            end_screen = world_to_screen(lidar.p2)
            pygame.draw.line(screen, (255, 0, 0), start_screen, end_screen, 1)

    pygame.display.flip()

pygame.init()
screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT), 0, 32)
pygame.display.set_caption("Box2D World Rendering")
surf = pygame.Surface(
            (WINDOW_HEIGHT + max(0.0, scroll) * SCALE, WINDOW_WIDTH)
        )

pygame.transform.scale(surf, (SCALE, SCALE))
clock = pygame.time.Clock()
# Define the world
world = b2.b2World()

def path_to_vertices(path):
    vertices, _ = parse_path(path)
    return [b2.b2Vec2(x, y) for x, y in vertices]

def create_aircraft_and_obstacles():
    global aircraft_body, obstacle_body, aircraft_vertices

    fuselage_vertices = path_to_vertices(PLANE_OUTLINE_PATH)
    main_wing_vertices = path_to_vertices(MAIN_WING_PATH)
    tail_plane_vertices = path_to_vertices(TAIL_PLANE_PATH)
    aircraft_vertices = fuselage_vertices + main_wing_vertices + tail_plane_vertices
    aircraft_vertices_flat = [coord for vertex in aircraft_vertices for coord in (vertex.x, vertex.y)]
    

    aircraft_body_def = b2.b2BodyDef()
    aircraft_body_def.type = b2.b2_dynamicBody
    aircraft_body_def.position = b2.b2Vec2(0, middle_y )
    aircraft_body_def.angle = 45 * math.pi / 180

    # Create the aircraft body first
    aircraft_body = world.CreateBody(aircraft_body_def)

    triangles_indices = earcut.earcut(aircraft_vertices_flat)

    for i in range(0, len(triangles_indices), 3):
        triangle_shape = b2.b2PolygonShape()
        triangle_shape.vertices = [
                aircraft_vertices[triangles_indices[i]],
                aircraft_vertices[triangles_indices[i + 1]],
                aircraft_vertices[triangles_indices[i + 2]],
            ]
        

        # Create the fixture on the aircraft body inside the loop
        aircraft_body.CreateFixture(shape=triangle_shape)

    # Create the aircraft body definition

    aircraft_initial_y = aircraft_body.position.y
    
    first_wall_distance = random.uniform(MIN_FIRST_WALL_DISTANCE, MAX_FIRST_WALL_DISTANCE)
    wall_positions = [first_wall_distance] + [random.uniform(MIN_WALL_DISTANCE + MIN_WALL_SPACING, MAX_WALL_DISTANCE) for _ in range(N_OBSTACLE_WALLS - 1)]
    cumulative_wall_positions = [sum(wall_positions[:i+1]) for i in range(len(wall_positions))]

    for i in range(N_OBSTACLE_WALLS):
        gap_height = random.uniform(MIN_GAP_HEIGHT, MAX_GAP_HEIGHT)
        wall_gap_width = random.uniform(MIN_GAP_WIDTH, MAX_GAP_WIDTH)
        gap_offset = random.uniform(MIN_GAP_OFFSET, MAX_GAP_OFFSET)
        gap_center_y = aircraft_initial_y + gap_offset

        # Top part of the wall
        top_wall_y = gap_center_y + gap_height / 2
        wall_body_def = b2.b2BodyDef()
        wall_body_def.type = b2.b2_staticBody
        wall_body_def.position = b2.b2Vec2(INITIAL_WALL_OFFSET + cumulative_wall_positions[i], top_wall_y + (MAX_WALL_HEIGHT - gap_height) / 2)

        top_wall_body = world.CreateBody(wall_body_def)
        wall_shape = b2.b2PolygonShape()
        wall_shape.SetAsBox(wall_gap_width / 2, (MAX_WALL_HEIGHT - gap_height) / 2)
        top_wall_body.CreateFixture(shape=wall_shape)

        # Bottom part of the wall
        bottom_wall_y = gap_center_y - gap_height / 2
        wall_body_def = b2.b2BodyDef()
        wall_body_def.type = b2.b2_staticBody
        wall_body_def.position = b2.b2Vec2(INITIAL_WALL_OFFSET + cumulative_wall_positions[i], bottom_wall_y - (MAX_WALL_HEIGHT - gap_height) / 2)

        bottom_wall_body = world.CreateBody(wall_body_def)
        wall_shape = b2.b2PolygonShape()
        wall_shape.SetAsBox(wall_gap_width / 2, (MAX_WALL_HEIGHT - gap_height) / 2)
        bottom_wall_body.CreateFixture(shape=wall_shape)

def world_to_screen(world_point):
    screen_x = int(world_point[0] * PPM)
    screen_y = int(WINDOW_HEIGHT - world_point[1] * PPM)
    return (screen_x, screen_y)

def print_lidar_sensor_values():
    sensor_values = [lidar.fraction * LIDAR_RANGE for lidar in lidars]
    # print("Lidar sensor values:", sensor_values)


def update_lidar():
    pos = aircraft_body.position
    angle = aircraft_body.angle
    offset_vector = b2.b2Vec2(4, 0)  # Define the offset vector for the front of the aircraft

    # Rotate the offset vector according to the body's angle
    rotated_offset_vector = b2.b2Vec2(
        offset_vector.x * math.cos(angle) - offset_vector.y * math.sin(angle),
        offset_vector.x * math.sin(angle) + offset_vector.y * math.cos(angle),
    )

    # Calculate the starting point of the lidar beams
    lidar_start_point = pos + rotated_offset_vector

    for i in range(10):
        lidars[i].fraction = 1.0
        angle_offset = (1.5 * i / 10.0) - 5.5  # Adjust the starting angle of the lidar beams
        start_angle = angle + angle_offset
        end_angle = angle + 1.5 * i / 10.0

        lidars[i].p1 = lidar_start_point
        lidars[i].p2 = (
            pos[0] + math.sin(start_angle) * LIDAR_RANGE,
            pos[1] - math.cos(start_angle) * LIDAR_RANGE,
        )
        world.RayCast(lidars[i], lidars[i].p1, lidars[i].p2)

def reset_world():
    global world, contact_listener, lidars

    # Destroy the current world
    world = None

    # Create a new world
    world = b2.b2World()

    # Create the aircraft and obstacles
    create_aircraft_and_obstacles()

    # Set the contact listener for the new world
    contact_listener = ContactListener()
    world.contactListener = contact_listener

    class LidarCallback(b2.b2RayCastCallback):
            def ReportFixture(self, fixture, point, normal, fraction):
                if (fixture.filterData.categoryBits & 1) == 0:
                    return -1
                self.p2 = point
                self.fraction = fraction
                return fraction

    lidars = [LidarCallback() for _ in range(10)]

def draw_polygon(polygon, body, color):
    vertices = [(body.transform * v) * PPM for v in polygon.vertices]
    vertices = [(v[0], WINDOW_HEIGHT - v[1]) for v in vertices]
    pygame.draw.polygon(screen, color, vertices)

class ContactListener(b2.b2ContactListener):
    def __init__(self):
        super().__init__()
        self.collision_detected = False

    def BeginContact(self, contact):
        body_a, body_b = contact.fixtureA.body, contact.fixtureB.body
        if (body_a == aircraft_body and body_b.type == b2.b2_staticBody) or (body_a.type == b2.b2_staticBody and body_b == aircraft_body):
            self.collision_detected = True



# Add the contact listener to the world
contact_listener = ContactListener()

create_aircraft_and_obstacles()
world.contactListener = contact_listener
ray_start = aircraft_body.position
ray_end = ray_start + b2.b2Vec2(10, 0)  # Change the end point based on the LiDAR's range and direction

reset_world()
running = True
while running:
    # Handle Pygame events
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
            
    new_vx = 5.0
    new_vy = 0.0
    new_w = 0.5

    new_linear_velocity = b2.b2Vec2(new_vx, new_vy)  # Replace new_vx and new_vy with your calculated values
    new_angular_velocity = new_w  # Replace new_w with your calculated angular velocity

    # Update the aircraft's linear and angular velocities
    aircraft_body.linearVelocity = new_linear_velocity
    aircraft_body.angularVelocity = new_angular_velocity

    pos = aircraft_body.position

    scroll = pos.x - WINDOW_WIDTH / SCALE / 5

    world.Step(1.0 / 60, 8, 3)
    world.ClearForces()

    if contact_listener.collision_detected:
        print("Collision detected! Resetting the world...")
        reset_world()
        contact_listener.collision_detected = False

    ray_start = aircraft_body.position  # Update the ray_start variable inside the loop
    ray_end = ray_start + b2.b2Vec2(10, 0)  # Change the end point based on the LiDAR's range and direction

    # Perform the ray cast
    update_lidar()
    print_lidar_sensor_values()
    screen.fill((255, 255, 255))
    # Draw the world first
    draw_world(world)

    pygame.display.flip()

    clock.tick(60)  # Control the simulation frequency using Pygame clock

pygame.quit()
