"""
This module wraps test functions to move ur10 based on COMPAS frameworks.
Main change is that translation from Rhino.Geometry to compas.geometry.
"""


# LIBRARIES


import compas_simple_ur_script as us  # move_l etc...
import compas_utils as uu             # matrix, geometry, visualize func...
import compas_simple_comm as uc       # send and read etc...

import compas.geometry as cg
import math as m
import time


# HARD CODED VALUES


PATH = uu.get_path()

ROBOT_IP = "192.168.10.10"
UR_SERVER_PORT = 30002
tool_height = 215

BASE = cg.Point(0, 0, 0)
ORIGIN = cg.Point(248.70, -398.30, -63.80)   # (left  upper  corner of sandbox)
X_POINT = cg.Point(785.00, -398.30, -63.80)  # (left  bottom corner of sandbox)
Y_POINT = cg.Point(248.70, 400.25, -63.80)   # (right bottom corner of sandbox)

H_SCAN = 484  # height of scan pos
W_KINECT = 83.17  # offset from tcp to camera in X direction
delta_x = (X_POINT.x - ORIGIN.x) / 2
delta_y = (Y_POINT.y - ORIGIN.y) / 2 + 50
S_POINT = cg.Point(delta_x-W_KINECT, delta_y, H_SCAN)  # scan point in ur space


# FUNCTIONS


def move_robot_to_points(move_to, robot_base, pure_trans=True,
                         velocity=0.05, acceleration=0.02, radius=0.01):
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

    for pt in move_to:
        pt[2] += tool_height
    # add transform frame
    way_pts = [uu.compas_to_robot_space(geo, robot_base, pure_trans=pure_trans)
               for geo in move_to]
    way_frames = [cg.Frame(pt, -cg.Vector.Xaxis(), -cg.Vector.Yaxis())
                  for pt in way_pts]

    for frame in way_frames:
        script += us.move_l_blend(frame, acceleration, velocity, radius)

    script = uc.concatenate_script(script)
    return script, way_pts


# RUN CODE


# toggle
print_script = False
visualize = False
send = True


# point to move to
move_to = cg.Point(0, 0, 0)

# write ur script
robot_base = uu.set_robot_base(ORIGIN, X_POINT, Y_POINT)
script_scan, way_pts = move_robot_to_points([S_POINT], robot_base)
script_move, way_pts = move_robot_to_points([move_to], robot_base)

# print script
if print_script:
    # print(script_json)
    print(script_scan)

# visualize
if visualize:
    uu.run_viewer(way_pts)

# timers
iteration = 60
excavation = 5
scanning = 5

# send script

if send:
    count = 0
    # while True:
    count += 1
    print('iteration: ', count)
    start = time.time()
    print('start: ', start)

    # while True:
    # byte_script_move = bytes(script_move, 'utf-8')
    # uc.send_script(ROBOT_IP, UR_SERVER_PORT, byte_script_move)

    # time.sleep(excavation)
    # print('excavation done')

    byte_script_scan = bytes(script_scan, 'utf-8')
    uc.send_script(ROBOT_IP, UR_SERVER_PORT, byte_script_scan)

    # time.sleep(scanning)
    # print('scanning done')

    # end = time.time()
    # print('fab_time: ', end - start)
