import numpy as np
import matplotlib.pyplot as plt

AIRCRAFT_HEIGHT = 270 #mm
AIRCRAFT_LENGTH = 1000 #mm
MIN_OBSTACLE_X = 1000 #mm
MAX_OBSTACLE_X = 50000 #mm
N_OBSTACLES = 3
MIN_OBSTACLE_HEIGHT = 50  #mm
MAX_OBSTACLE_HEIGHT = 1000 #mm
MIN_OBSTACLE_WIDTH = 50 #mm
MAX_OBSTACLE_WIDTH = 1000 #mm
MAX_OBSTACLE_Y = 2000 #mm
GAP_Y_MAX = 10000

class Rectangle:
    """A class to represent a simple 2D rectangular obstacle."""
    def __init__(self, centre_x, centre_y, width, height, orientation=0):
        self.x = centre_x
        self.y = centre_y
        self.width = width
        self.height = height
        self.orientation = orientation
        self._coordinates = self.update_coordinates(self.x, self.y, self.orientation)

    @property
    def coordinates(self):
        return self._coordinates

    def update_coordinates(self, centre_x, centre_y, orientation=0):
        """Returns the a numpy array of the four corners of the obstacle, based on
        the centre point, width, height, and current orientation (pitch angle)."""
        x = self.x
        y = self.y
        w = self.width
        h = self.height
        o = self.orientation

        x1 = x + (w/2)*np.cos(o) - (h/2)*np.sin(o)
        y1 = y + (w/2)*np.sin(o) + (h/2)*np.cos(o)
        x2 = x + (w/2)*np.cos(o) + (h/2)*np.sin(o)
        y2 = y + (w/2)*np.sin(o) - (h/2)*np.cos(o)
        x3 = x - (w/2)*np.cos(o) + (h/2)*np.sin(o)
        y3 = y - (w/2)*np.sin(o) - (h/2)*np.cos(o)
        x4 = x - (w/2)*np.cos(o) - (h/2)*np.sin(o)
        y4 = y - (w/2)*np.sin(o) + (h/2)*np.cos(o)

        return np.array([[x1,y1],[x2,y2],[x3,y3],[x4,y4]])
    
    def plot(self):
        # A method to plot a fully closed rectangle
        plt.plot(self.coordinates[:,0], self.coordinates[:,1], 'k-')
        plt.plot([self.coordinates[0,0], self.coordinates[3,0]], [self.coordinates[0,1], self.coordinates[3,1]], 'k-')
        plt.axis('equal')
        
    
class SimpleObstacle(Rectangle):
    """A class to represent a simple 2D rectangular obstacle."""
    def __init__(self, centre_x, centre_y, width, height, orientation=0):
        super().__init__(centre_x, centre_y, width, height, orientation)

def make_random_obstacles(min_distance=1000):  # Added min_distance parameter
    obstacles = []
    for i in range(N_OBSTACLES):
        valid_obstacle = False
        while not valid_obstacle:
            x = np.random.randint(MIN_OBSTACLE_X, MAX_OBSTACLE_X)
            y = np.random.randint(-MAX_OBSTACLE_Y, MAX_OBSTACLE_Y)
            w = np.random.randint(MIN_OBSTACLE_WIDTH, MAX_OBSTACLE_WIDTH)
            h = np.random.randint(MIN_OBSTACLE_HEIGHT, MAX_OBSTACLE_HEIGHT)
            o = np.random.randint(0, 360)
            new_obstacle = SimpleObstacle(x, y, w, h, o)
            if min_distance_met(new_obstacle, obstacles, min_distance):
                valid_obstacle = True
                obstacles.append(new_obstacle)
    return obstacles

def make_single_inline_obstacle():
    """A function to create a single obstacle a minium distance from the fuselage,
    but at the same y coordinate."""
    x = np.random.randint(MIN_OBSTACLE_X, MAX_OBSTACLE_X)
    y = 0
    w = np.random.randint(MIN_OBSTACLE_WIDTH, MAX_OBSTACLE_WIDTH)
    h = np.random.randint(MIN_OBSTACLE_HEIGHT, MAX_OBSTACLE_HEIGHT)
    o = 0
    new_obstacle = SimpleObstacle(x, y, w, h, o)
    return [new_obstacle]

def make_gap_obstacle(min_gap_size=(AIRCRAFT_HEIGHT*1.5), max_gap_size=1000):
    """A function to create a gap obstacle, made up of a single wall-like structure
    with a gap in the middle, similar to Flappy Bird."""
    x = np.random.randint(MIN_OBSTACLE_X, MAX_OBSTACLE_X)
    w = np.random.randint(MIN_OBSTACLE_WIDTH, MAX_OBSTACLE_WIDTH)
    
    # Set the height to the maximum possible height
    h = GAP_Y_MAX

    o = 0

    # Calculate the y coordinates of the gap
    gap_size = np.random.randint(min_gap_size, max_gap_size)
    half_gap = gap_size / 2
    max_y = MAX_OBSTACLE_Y - half_gap
    min_y = -max_y

    gap_center = np.random.randint(min_y + half_gap, max_y - half_gap)

    y1 = gap_center - half_gap
    y2 = gap_center + half_gap

    obstacle1 = SimpleObstacle(x, y1 - h / 2, w, h - half_gap, o)
    obstacle2 = SimpleObstacle(x, y2 + h / 2, w, h - half_gap, o)

    return [obstacle1, obstacle2]


def make_random_gap_obstacles(n, min_distance=1000, min_start_distance=3000, max_start_distance=4000, max_distance=5000):
    """A function to create n gap obstacles with a minimum and maximum distance apart."""
    obstacles = []

    for i in range(n):
        valid_obstacle = False
        while not valid_obstacle:
            gap_obstacle = make_gap_obstacle()
            x = gap_obstacle[0].x

            if i == 0:
                # Check the minimum and maximum distance from the starting point
                if min_start_distance <= x <= max_start_distance:
                    valid_obstacle = True
            else:
                # Check the minimum and maximum distance between gap obstacles
                distance = x - obstacles[-2].x
                if min_distance <= distance <= max_distance:
                    valid_obstacle = True

            if valid_obstacle:
                obstacles.extend(gap_obstacle)

    return obstacles

def dot_product(v1, v2):
    return v1[0] * v2[0] + v1[1] * v2[1]

def edge_vector(p1, p2):
    return (p2[0] - p1[0], p2[1] - p1[1])

def orthogonal_vector(v):
    return (-v[1], v[0])

def project_polygon(axis, polygon):
    dots = [dot_product(vertex, axis) for vertex in polygon]
    return (min(dots), max(dots))

def is_overlap(interval1, interval2):
    return not (interval1[1] < interval2[0] or interval1[0] > interval2[1])

def is_collision(poly1, poly2):
    edges = [edge_vector(poly1[i], poly1[i - 1]) for i in range(len(poly1))] + [edge_vector(poly2[i], poly2[i - 1]) for i in range(len(poly2))]
    axes = [orthogonal_vector(edge) for edge in edges]

    for axis in axes:
        projection1 = project_polygon(axis, poly1)
        projection2 = project_polygon(axis, poly2)

        if not is_overlap(projection1, projection2):
            return False

    return True

aircraft = [(x1, y1), (x2, y2), (x3, y3), (x4, y4)]
obstacle1 = [(x1, y1), (x2, y2), (x3, y3), (x4, y4)]
obstacle2 = [(x1, y1), (x2, y2), (x3, y3), (x4, y4)]

obstacles = [obstacle1, obstacle2]

for obstacle in obstacles:
    if is_collision(aircraft, obstacle):
        print("Collision detected!")
        # Handle the collision here



def plot_all(fuselage, obstacles):
    """A method to plot all shapes on the same figure."""
    fuselage.plot()
    for obstacle in obstacles:
        obstacle.plot()
    plt.ylim(-3000, 3000)
    plt.show()
    
def euclidean_distance(x1, y1, x2, y2):
    return np.sqrt((x1 - x2)**2 + (y1 - y2)**2)

def min_distance_met(obstacle, obstacles, min_distance):
    for existing_obstacle in obstacles:
        distance = euclidean_distance(obstacle.x, obstacle.y, existing_obstacle.x, existing_obstacle.y)
        if distance < min_distance:
            return False
    return True

if __name__ == '__main__':
    fuselage = Rectangle(0,0,AIRCRAFT_LENGTH,AIRCRAFT_HEIGHT)
    # obstacles = []
    # inline_obstacle = make_single_inline_obstacle()
    # gap_obstacle = make_gap_obstacle()
    obstacles = make_random_gap_obstacles(3, min_distance=2000, min_start_distance=3000, max_distance=5000)

    # obstacles.extend(inline_obstacle)
    # obstacles.extend(gap_obstacle)
    plot_all(fuselage, obstacles)
    


    
