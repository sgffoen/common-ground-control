# LIBRARIES


from toolpath import random_toolpath_gen as rtg
from UR import compas_simple_comm as uc
from UR import compas_simple_ur_script as us
from UR import compas_utils as uu

import compas.geometry as cg
import math as m
import time
import numpy as np
import json


# HARD CODED VALUES


# fact sheet
with open('data/facts.json') as f:
    facts = json.load(f)

# for robot
BASE = cg.Point(facts['base']['x'], facts['base']['y'], facts['base']['z'])
COMPAS_FRAME = cg.Frame(BASE, cg.Vector.Xaxis(), cg.Vector.Yaxis())

# robot corner points
ORIGIN = cg.Point(facts['robot_corner_pts']['pt0'][0],
                  facts['robot_corner_pts']['pt0'][1],
                  facts['robot_corner_pts']['pt0'][2])
X_POINT = cg.Point(facts['robot_corner_pts']['ptx'][0],
                   facts['robot_corner_pts']['ptx'][1],
                   facts['robot_corner_pts']['ptx'][2])
Y_POINT = cg.Point(facts['robot_corner_pts']['pty'][0],
                   facts['robot_corner_pts']['pty'][1],
                   facts['robot_corner_pts']['pty'][2])

# for scanning
H_SCAN = facts['h_scan'] + facts['tcp_len']
W_KINECT = facts['w_kinect']  # offset from tcp to camera in X direction
delta_x = (X_POINT.x - ORIGIN.x) / 2
delta_y = (Y_POINT.y - ORIGIN.y) / 2 + 50
S_POINT = cg.Point(delta_x-W_KINECT, delta_y, H_SCAN)  # scan point in ur space

# for toolpath
figsize_x = facts['fig_size']['x']
figsize_y = facts['fig_size']['y']
figsize_z = facts['fig_size']['z']
box_x = abs(X_POINT.x - ORIGIN.x)
box_y = abs(Y_POINT.y - ORIGIN.y)
box_z = 0.

# croping
xSandbox_imgSize = facts['crop_idx']['xEnd'] - facts['crop_idx']['xStart']
ySandbox_imgSize = facts['crop_idx']['yStart'] - facts['crop_idx']['yEnd']


# FUNCTIONS


def move_robot_along_line(random_toolpath, robot_base, z_fig_center,
                          pure_trans=True,
                          velocity=0.15, acceleration=0.05, radius=0.01,
                          safety_dist=50):
    """
    Function that move robot to single or multiple points.

    Args:
        move_to : a list of compas.geometry Point.
        velocity : float. (m/s)
        acceleration : float. (m/s^2)
        radius : float. A blend radius (m)

    Returns:
        script : UR script
        way_pts : compas.geometry Point. Points to be visualized in viewer.
    """

    script = ""
    script += us.set_tcp_by_angles(0.0,               # X
                                   0.0,               # Y
                                   facts['tcp_len'],       # Z
                                   m.radians(0.0),    # RX
                                   m.radians(180.0),  # RY
                                   m.radians(90.0))   # RZ

    line_dir = random_toolpath.direction
    cross = line_dir.cross(-cg.Vector.Zaxis())

    move_pt = [random_toolpath.end, random_toolpath.start]
    move_pt.insert(0, move_pt[0].copy())
    move_pt.append(move_pt[-1].copy())

    frames = []
    for i, pt in enumerate(move_pt):
        if i == 0 or i == len(move_pt)-1:
            pt[2] += (facts['tcp_len'] + safety_dist)
        else:
            pt[2] += (facts['tcp_len'] - z_fig_center)
        frames.append(cg.Frame(pt, line_dir, cross))

    # add transform frame
    way_frames = [uu.compas_to_robot_space(geo,
                                           robot_base,
                                           pure_trans=pure_trans)
                  for geo in frames]

    for frame in way_frames:
        script += us.move_l_blend(frame, acceleration, velocity, radius)

    script = uc.concatenate_script(script)
    return script


def move_robot_to_points(move_to, robot_base, pure_trans=True,
                         velocity=0.30, acceleration=0.10, radius=0.0):
    """
    Function that move robot to single or multiple points.

    Args:
        move_to : a list of compas.geometry Point.
        velocity : float. (m/s)
        acceleration : float. (m/s^2)
        radius : float. A blend radius (m)

    Returns:
        script : UR script
        way_pts : compas.geometry Point. Points to be visualized in viewer.
    """

    script = ""
    script += us.set_tcp_by_angles(0.0,               # X
                                   0.0,               # Y
                                   facts['tcp_len'],       # Z
                                   m.radians(0.0),    # RX
                                   m.radians(180.0),  # RY
                                   m.radians(90.0))   # RZ

    # add transform frame
    way_pts = [uu.compas_to_robot_space(geo, robot_base, pure_trans=pure_trans)
               for geo in move_to]
    way_frames = [cg.Frame(pt, -cg.Vector.Xaxis(), -cg.Vector.Yaxis())
                  for pt in way_pts]

    for frame in way_frames:
        script += us.move_l_blend(frame, acceleration, velocity, radius)

    script = uc.concatenate_script(script)
    return script


def get_toolpath(count):
    # get toolpath
    random_line = rtg.random_line_gen(COMPAS_FRAME, figsize_x, figsize_y)
    # save toolpath as a image
    rtg.save_line_image(random_line, figsize_x, figsize_y, iteration=count)
    # provide toolpath
    random_toolpath, x_coord, y_coord = rtg.deploy_fig_to_box(random_line,
                                                          figsize_x, figsize_y, figsize_z,
                                                          box_x, box_y, box_z)
    # remap value into index
    x_ind = int(uu.remapValue(x_coord, int(figsize_x/2), int(box_x-figsize_x/2), 0, ySandbox_imgSize-1))
    y_ind = int(uu.remapValue(y_coord, int(figsize_y/2), int(box_y-figsize_y/2), 0, xSandbox_imgSize-1))
    return random_toolpath, x_ind, y_ind


def execute_toolpath(random_toolpath, z_fig_center, excavation_time=15):
    print('excavation START')
    robot_base = uu.set_robot_base(ORIGIN, X_POINT, Y_POINT)
    script_move = move_robot_along_line(random_toolpath,
                                        robot_base,
                                        z_fig_center)
    uc.send_script(facts['robot_ip'],
                   facts['ur_server_port'],
                   bytes(script_move, 'utf-8'))
    time.sleep(excavation_time)
    print('excavation DONE')


def scan_pose(scanning_time=7.5):
    print('scanning START')
    robot_base = uu.set_robot_base(ORIGIN, X_POINT, Y_POINT)
    script_scan = move_robot_to_points([S_POINT], robot_base)
    uc.send_script(facts['robot_ip'],
                   facts['ur_server_port'],
                   bytes(script_scan, 'utf-8'))
    time.sleep(scanning_time)


def get_z_fig(pcl, x_ind, y_ind):
    arr = np.asarray(pcl.points)
    zAve = np.mean(arr, axis=0)[2]

    arr_flip = np.flipud(arr)

    arr_re = np.reshape(arr_flip, (xSandbox_imgSize, ySandbox_imgSize, 3))

    z_fig = uu.remapValue(arr_re[y_ind][x_ind][2], zAve-50, zAve+50, 140, 120)
    return z_fig



if __name__ == "__main__":
    pass
