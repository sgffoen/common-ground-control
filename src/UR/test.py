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


# HARD CODED VALUES


ROBOT_IP = "192.168.10.10"
UR_SERVER_PORT = 30002
tool_height = 215 + 175

BASE = cg.Point(0, 0, 0)
ORIGIN = cg.Point(248.70, -398.30, -63.80)   # (left  upper  corner of sandbox)
X_POINT = cg.Point(785.00, -398.30, -63.80)  # (left  bottom corner of sandbox)
Y_POINT = cg.Point(248.70, 400.25, -63.80)   # (right bottom corner of sandbox)


# FUNCTIONS


def move_robot_to_points(move_to, robot_base, pure_trans,
                         velocity=0.05, acceleration=0.02, radius=0.03):
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
pure_trans = False
print_script = False
visualize = False
send = False

# point to move to
# move_to = [cg.Point(502, -43, 0)]
move_to = [cg.Point(0, 0, 0)]

# write ur script
robot_base = uu.set_robot_base(ORIGIN, X_POINT, Y_POINT)
script, way_pts = move_robot_to_points(move_to, robot_base, pure_trans)

# print script
if print_script:
    print(script)

# visualize
if visualize:
    uu.run_viewer(way_pts)

# send script
if send is True:
    byte_script = bytes(script, 'utf-8')
    uc.send_script(ROBOT_IP, UR_SERVER_PORT, byte_script)
