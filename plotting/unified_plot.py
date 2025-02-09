import os
import json
from matplotlib import patches

import pandas
import matplotlib.pyplot as plt
from matplotlib.path import Path
from matplotlib.patches import PathPatch

import numpy as np

from scipy.spatial.transform import Rotation

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

def draw_vehicle(ax, x, y, pitch, elev, scale = 0.1):
    transform = np.array([
        [np.cos(pitch), -np.sin(pitch)],
        [np.sin(pitch),  np.cos(pitch)]
    ])
    for path in [PLANE_OUTLINE_PATH, MAIN_WING_PATH, TAIL_PLANE_PATH]:
        vertices, codes = parse_path(path)
        vertices = scale * (transform @ np.array(vertices).T).T
        positioned = vertices + np.array([x,y])
        ax.add_patch(PathPatch(Path(positioned, codes=codes, closed=True), fill=False))

    # Plot the elevator
    vertices, codes = parse_path(TAIL_PLANE_PATH)
    elev_transform = np.array([
        [np.cos(-elev), -np.sin(-elev)],
        [np.sin(-elev),  np.cos(-elev)]
    ])
    elev_offset = np.array([-9.5,0.25])
    elev_vertices = (elev_transform @ (np.array(vertices) - elev_offset).T).T + elev_offset
    elev_vertices = scale * (transform @ np.array(elev_vertices).T).T
    elev_positioned = elev_vertices + np.array([x,y])
    ax.add_patch(PathPatch(Path(elev_positioned, codes=codes, closed=True), fill=False, color='g'))

def get_eulerized(data):
    if "qx" in data:
        rot = Rotation.from_quat(np.array([data.qx,data.qy,data.qz,data.qw]).T)
        rot_euler = rot.as_euler('zyx', degrees=True)
        euler_df = pandas.DataFrame(data=rot_euler, columns=['yaw', 'pitch', 'roll'])
        import copy
        before = copy.deepcopy(euler_df.pitch)
        # The conversion to Euler angles will convert inversions due to pitch into inversions due to roll
        # This mess makes it so only the pitch changes
        is_inverted = euler_df.yaw != 0
        pitch_sign = np.copysign(1,euler_df.pitch)
        euler_df.pitch = (1-is_inverted) * euler_df.pitch + \
            is_inverted * (pitch_sign * 90 + (pitch_sign * 90 - euler_df.pitch))
    else:
        euler_df = data
    return euler_df

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("run_name", help="Name of run")
    parser.add_argument("-d", "--directory", help="Directory for runs", default="./runs")
    parser.add_argument("--save", action="store_true", help="Save the output plots")
    parser.add_argument("--env", help="Show the output plots", default="obstacles")

    args = parser.parse_args()

    run_dir = os.path.join(args.directory, args.run_name)

    data = pandas.read_csv(os.path.join(run_dir, "output.csv"))

    if args.env == "land":
        # shift y values up by 25
        data.z = data.z - 25

    data_eul = get_eulerized(data)

    with open(f"{run_dir}/metadata.json") as f:
        metadata = json.load(f)

    plt.minorticks_on()
    plt.grid(True, 'both')

    # Get the unique episode numbers
    episodes = data.iloc[:, 0].unique()

    # create counter of successful episodes
    success_counter = 0

    # First, calculate success for all episodes
    episode_success = {}
    for i in range(len(episodes)-1):
        episode = episodes[i]
        next_episode = episodes[i+1]
        next_episode_data = data[data.iloc[:, 0] == next_episode]
        success = next_episode_data.iloc[0, -1]
        episode_success[episode] = success
        if success == 1:
            success_counter += 1

    # Now plot every 5th episode using the pre-calculated success values
    for i in range(0, len(episodes)-1, 1):
        episode = episodes[i]
        episode_data = data[data.iloc[:, 0] == episode]
        
        success = episode_success.get(episode, 0)  # Get pre-calculated success value
        color = 'g' if success else 'r'
        # line is solid if successful, dashed if not
        line_style = '-' if success else '--'

        # plt.plot(episode_data.x, -episode_data.z, color=color, linewidth=1.0, linestyle=line_style)
        episode_data_eul = get_eulerized(episode_data)

         # draw vehicle at every 10th point
        for i in range(len(episode_data.index)):
            if i % 10 != 0:
                continue
            draw_vehicle(plt.gca(), episode_data.x.iloc[i], -episode_data.z.iloc[i], np.radians(episode_data_eul.pitch.iloc[i]), episode_data.elevator.iloc[i])
            print(f"x = {episode_data.x.iloc[i]}, z = {episode_data.z.iloc[i]}, pitch = {episode_data_eul.pitch.iloc[i]}, elevator = {episode_data.elevator.iloc[i]}")
        plt.ylim([0, None])

    # Print success rate
    print(f"Success rate: {success_counter}")

    if args.env == "obstacles":
        plt.gcf().set_size_inches(12, 6)
        # Draw vertical lines
        start_x = data.x.min()  # Starting x position
        first_wall = 30
        start_x = first_wall + start_x
        line_distance = 30  # Distance between lines
        num_lines = 8  # Number of lines

        for i in range(num_lines):
            line_x = start_x + i * line_distance
            plt.axvline(x=line_x, color='b', linestyle='--')

            plt.xlabel('x-Position (m)', fontsize=14)  # Increase label font size
            plt.ylabel('z-Position (m)', fontsize=14)  # Increase label font size
            plt.xticks(fontsize=12)  # Increase tick numbers font size
            plt.yticks(fontsize=12)  # Increase tick numbers font size
            plt.axis('equal')

        ax = plt.gca()
        last_pos = [10_000, 10_000]
    else:
        plt.xlabel('x-Position (m)', fontsize=14)
        plt.ylabel('Height (m)', fontsize=14)
        plt.xlim([0, 80])
        plt.ylim([0, 30])

        gap_centre = 20
        gap_width = 30
        block_height = 12.5
        block_width = 25

        plt.axhline(y=0, color='k', linestyle='-')

        rect_l = patches.Rectangle((gap_centre - block_width/2, 0), block_width, block_height, linewidth=1, edgecolor='k', facecolor='k')
        rect_r = patches.Rectangle((gap_centre + gap_width + block_width/2, 0), block_width, block_height, linewidth=1, edgecolor='k', facecolor='k')
        
        ax = plt.gca()
        ax.add_patch(rect_l)
        ax.add_patch(rect_r)

    # save plot if desired
    if args.save:
        plt.savefig(f"{run_dir}/trajectory.eps", format="eps", dpi=1200)

    # Create state plots
    fig2, ax2 = plt.subplots(3, 2, sharex=True)

    ylabel_common_args = {
        "fontsize": 12
    }

    def plot_against_time(axis, ydata, label, xdata=data.time, color='k', linestyle='-'):
        row = axis // 2  # Integer division to get the row
        col = axis % 2  # Modulo operation to get the column
        ax2[row, col].plot(xdata, ydata, color=color, linestyle=linestyle)
        ax2[row, col].set_ylabel(label, **ylabel_common_args)
        ax2[row, col].grid(True, "both")

    # Plot states for every 5th episode using pre-calculated success values
    for i in range(0, len(episodes)-1, 1):
        episode = episodes[i]
        episode_data = data[data.iloc[:, 0] == episode]
        episode_data_eul = get_eulerized(episode_data)

        success = episode_success.get(episode, 0)  # Get pre-calculated success value
        color = 'g' if success else 'r'

        # line is solid if successful, dashed if not


        plot_against_time(0, -episode_data.z, "Height (m)", episode_data.time, color)
        plot_against_time(1, episode_data_eul.pitch, "Pitch Angle (deg)", episode_data.time, color)
        plot_against_time(2, np.degrees(episode_data.alpha), "Alpha (deg)", episode_data.time, color)
        plot_against_time(3, np.hypot(episode_data.u, episode_data.w), "Airspeed (m/s)", episode_data.time, color)
        plot_against_time(4, np.degrees(episode_data.elevator), "Elevator (deg)", episode_data.time, color)
        plot_against_time(5, episode_data.throttle, "Throttle (frac)", episode_data.time, color)

    # Set x-axis labels and limits
    for axis in ax2.flatten():
        axis.set_xlabel('Time (s)')

    if args.save:
        plt = fig2
        plt.tight_layout()
        plt.set_size_inches(10, 6)
        plt.savefig(f"{run_dir}/states.eps", format="eps", dpi=1200)

    # plt.show()