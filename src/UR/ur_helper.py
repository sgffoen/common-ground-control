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
__BASE__ = cg.Point(facts['base']['x'], facts['base']['y'], facts['base']['z'])
__COMPAS_FRAME__ = cg.Frame(__BASE__, cg.Vector.Xaxis(), cg.Vector.Yaxis())

# robot corner points
__ORIGIN__ = cg.Point(facts['robot_corner_pts']['pt0'][0],
                  facts['robot_corner_pts']['pt0'][1],
                  facts['robot_corner_pts']['pt0'][2])
__X_POINT__ = cg.Point(facts['robot_corner_pts']['ptx'][0],
                   facts['robot_corner_pts']['ptx'][1],
                   facts['robot_corner_pts']['ptx'][2])
__Y_POINT__ = cg.Point(facts['robot_corner_pts']['pty'][0],
                   facts['robot_corner_pts']['pty'][1],
                   facts['robot_corner_pts']['pty'][2])

# for scanning
H_SCAN = facts['h_scan'] + facts['tcp_len']
W_KINECT = facts['w_kinect']  # offset from tcp to camera in X direction
delta_x = (__X_POINT__.x - __ORIGIN__.x) / 2
delta_y = (__Y_POINT__.y - __ORIGIN__.y) / 2 + 55
S_POINT = cg.Point(delta_x - W_KINECT, delta_y, H_SCAN)  # scan point in ur space

# for toolpath
figsize_x = facts['fig_size']['x']
figsize_y = facts['fig_size']['y']
figsize_z = facts['fig_size']['z']
box_x = abs(__X_POINT__.x - __ORIGIN__.x)
box_y = abs(__Y_POINT__.y - __ORIGIN__.y)
box_z = 0.

# croping
x_sandbox_img_size = facts['crop_idx']['xEnd'] - facts['crop_idx']['xStart']
y_sandbox_img_size = facts['crop_idx']['yStart'] - facts['crop_idx']['yEnd']


# FUNCTIONS

def get_toolpath(count):
    # get toolpath
    random_line = rtg.random_line_gen(__COMPAS_FRAME__, figsize_x, figsize_y)
    # save toolpath as a image
    rtg.save_line_image(random_line, figsize_x, figsize_y, iteration=count)
    # provide toolpath
    random_toolpath, x_coord, y_coord = rtg.deploy_fig_to_box(random_line,
                                                          figsize_x, figsize_y, figsize_z,
                                                          box_x, box_y, box_z)
    # remap value into index
    x_ind = int(uu.remapValue(x_coord, int(figsize_x/2), int(box_x-figsize_x/2), 0, y_sandbox_img_size-1))
    y_ind = int(uu.remapValue(y_coord, int(figsize_y/2), int(box_y-figsize_y/2), 0, x_sandbox_img_size-1))
    return random_toolpath, x_ind, y_ind


def get_z_fig(pcl, x_ind, y_ind):
    arr = np.asarray(pcl.points)
    zAve = np.mean(arr, axis=0)[2]

    arr_re = np.reshape(arr, (x_sandbox_img_size, y_sandbox_img_size, 3))
    z_toolpathbox2D = arr_re[y_ind][x_ind][2] + 289 - 35 # z_value from pendant - thickness of tcp

    return z_toolpathbox2D


def adapt_toolpath(random_toolpath, z_toolpathbox2D):
    random_toolpath_adapted = None
    return random_toolpath_adapted


def move_robot_along_line(random_toolpath, robot_base, z_toolpathbox2D,
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
            pt[2] += (facts['tcp_len'] + z_toolpathbox2D)
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


def move_robot_to_scan_pose(move_to, robot_base, pure_trans=True,
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

    RX = cg.Rotation.from_axis_and_angle(-cg.Vector.Xaxis(), m.radians(0.0))
    RY = cg.Rotation.from_axis_and_angle(-cg.Vector.Yaxis(), m.radians(-1.5))
    RZ = cg.Rotation.from_axis_and_angle(cg.Vector.Zaxis(), m.radians(1.0))

    T = RX * RY * RZ

    for frame in way_frames:
        frame = frame.transformed(T)
        script += us.move_l_blend(frame, acceleration, velocity, radius)

    script = uc.concatenate_script(script)
    return script


def move_robot_to_a_frame(move_to, robot_base, z_center_toolpathbox2D, pure_trans=True,
                         velocity=0.30, acceleration=0.10, radius=0.0):

    script = ""
    script += us.set_tcp_by_angles(0.0,               # X
                                   0.0,               # Y
                                   facts['tcp_len'],       # Z
                                   m.radians(0.0),    # RX
                                   m.radians(180.0),  # RY
                                   m.radians(90.0))   # RZ

    # add transform frame
    move_to.point.z += facts['tcp_len']
    way_frame = uu.compas_to_robot_space(move_to, robot_base, pure_trans=pure_trans)

    # adapt height
    way_frame_adapted = adapt_height_from_pcl(way_frame, z_center_toolpathbox2D)
    script += us.move_l_blend(way_frame_adapted, acceleration, velocity, radius)

    script = uc.concatenate_script(script)
    return script


def execute_toolpath(random_toolpath, z_toolpathbox2D, excavation_time=15):
    print('excavation START')
    robot_base = uu.set_robot_base(__ORIGIN__, __X_POINT__, __Y_POINT__)
    script_move = move_robot_along_line(random_toolpath,
                                        robot_base,
                                        z_toolpathbox2D)
    uc.send_script(facts['robot_ip'],
                   facts['ur_server_port'],
                   bytes(script_move, 'utf-8'))
    time.sleep(excavation_time)
    print('excavation DONE')


def scan_pose(scanning_time=7.5):
    print('scanning START')
    robot_base = uu.set_robot_base(__ORIGIN__, __X_POINT__, __Y_POINT__)
    script_scan = move_robot_to_scan_pose([S_POINT], robot_base)
    print(script_scan)
    uc.send_script(facts['robot_ip'],
                   facts['ur_server_port'],
                   bytes(script_scan, 'utf-8'))
    time.sleep(scanning_time)


def test_pose(move_to, z_center_toolpathbox2D):
    print('scanning START')
    robot_base = uu.set_robot_base(__ORIGIN__, __X_POINT__, __Y_POINT__)
    script_test = move_robot_to_a_frame(move_to, robot_base, z_center_toolpathbox2D)
    print(script_test)
    uc.send_script(facts['robot_ip'],
                   facts['ur_server_port'],
                   bytes(script_test, 'utf-8'))


def adapt_height_from_pcl(way_frame, z_center_toolpathbox2D):
    way_frame.point.z += (z_center_toolpathbox2D + 289)
    print(z_center_toolpathbox2D + 289)
    return way_frame


if __name__ == "__main__":
    scan_pose()

    # z_center_toolpathbox2D = -362
    # a_frame = cg.Frame(cg.Point(100, 389, -40), cg.Vector.Xaxis(), cg.Vector.Yaxis())
    # R = cg.Rotation.from_axis_and_angle(cg.Vector.Zaxis(), m.radians(0))
    # a_frame.transform(R)
    # test_pose(a_frame, z_center_toolpathbox2D)
    pass
