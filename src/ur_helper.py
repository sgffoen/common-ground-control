from toolpath import random_toolpath_gen as rtg
from UR import compas_simple_ur_script as us
from UR import compas_utils as uu
from UR import compas_simple_comm as uc

import compas.geometry as cg
import math as m
import time
import numpy as np


# HARD CODED VALUES


# for robot
ROBOT_IP = "192.168.10.10"
UR_SERVER_PORT = 30002
tool_height = 215

BASE = cg.Point(0, 0, 0)
COMPAS_FRAME = cg.Frame(BASE, cg.Vector.Xaxis(), cg.Vector.Yaxis())

ORIGIN = cg.Point(282.70, -398.30, -63.80)   # (left  upper  corner of sandbox)
X_POINT = cg.Point(785.00, -398.30, -63.80)  # (left  bottom corner of sandbox)
Y_POINT = cg.Point(282.70, 400.25, -63.80)   # (right bottom corner of sandbox)

# for scanning
H_SCAN = 385 + tool_height  # height of scan pos(215=original toolheight)
W_KINECT = 83.17  # offset from tcp to camera in X direction
delta_x = (X_POINT.x - ORIGIN.x) / 2
delta_y = (Y_POINT.y - ORIGIN.y) / 2 + 50
S_POINT = cg.Point(delta_x-W_KINECT, delta_y, H_SCAN)  # scan point in ur space

# for toolpath
fig_x = 256.
fig_y = 256.
fig_z = 0.
box_x = abs(X_POINT.x - ORIGIN.x)
box_y = abs(Y_POINT.y - ORIGIN.y)
box_z = 0.


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
                                   tool_height,       # Z
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
            pt[2] += (tool_height + safety_dist)
        else:
            pt[2] += (tool_height - z_fig_center)
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
                                   tool_height,       # Z
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
    random_line = rtg.random_line_gen(COMPAS_FRAME, fig_x, fig_y)
    # save toolpath as a image
    rtg.save_line_image(random_line, fig_x, fig_y, iteration=count)
    # provide toolpath
    random_toolpath, x_val, y_val = rtg.deploy_fig_to_box(random_line,
                                                          fig_x, fig_y, fig_z,
                                                          box_x, box_y, box_z)
    # remap value into index
    x_ind = int(uu.remapValue(x_val, int(fig_x/2), int(box_x-fig_x), 0, 300-1))
    y_ind = int(uu.remapValue(y_val, int(fig_y/2), int(box_y-fig_y), 0, 485-1))
    return random_toolpath, x_ind, y_ind


def execute_toolpath(random_toolpath, z_fig_center, excavation_time=15):
    # execute toolpath
    # 1) excavation
    print('excavation START')
    robot_base = uu.set_robot_base(ORIGIN, X_POINT, Y_POINT)
    script_move = move_robot_along_line(random_toolpath,
                                        robot_base,
                                        z_fig_center)
    uc.send_script(ROBOT_IP, UR_SERVER_PORT, bytes(script_move, 'utf-8'))
    time.sleep(excavation_time)
    print('excavation DONE')


def scan_pose(scanning_time=7.5):
    # 2) scanpose
    print('scanning START')
    robot_base = uu.set_robot_base(ORIGIN, X_POINT, Y_POINT)
    script_scan = move_robot_to_points([S_POINT], robot_base)
    uc.send_script(ROBOT_IP, UR_SERVER_PORT, bytes(script_scan, 'utf-8'))
    time.sleep(scanning_time)


def get_z_fig(pcl, x_ind, y_ind):
    arr = np.asarray(pcl.points)
    zAve = np.mean(arr, axis=0)[2]
    # zMin = np.amin(arr, axis=0)[2]

    arr_flip = np.flipud(arr)
    arr_re = np.reshape(arr_flip, (485, 300, 3))

    z_fig = uu.remapValue(arr_re[y_ind][x_ind][2], zAve-50, zAve+50, 140, 120)
    return z_fig
