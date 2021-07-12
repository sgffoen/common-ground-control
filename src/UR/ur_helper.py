# LIBRARIES
from toolpath import random_toolpath_gen as rtg
from UR import compas_simple_comm as uc
from UR import compas_simple_ur_script as us
from UR import compas_utils as uu

# for testing.
# import compas_simple_comm as uc
# import compas_simple_ur_script as us
# import compas_utils as uu

import compas.geometry as cg
import math as m
import time
import numpy as np
import json

# fact sheet
with open('data/facts.json') as f:
    facts = json.load(f)


# FUNCTIONS
'''
move_robot_to_scan_pose
'''


def move_robot_to_scan_pose(scan_pt,
                            degree_x=0.0, degree_y=-1.5, degree_z=1.0,
                            vel=0.30, acc=0.10, rad=0.0):
    script = ""
    script += us.set_tcp_by_angles(0.0,               # X
                                   0.0,               # Y
                                   facts['tcp_len'],  # Z
                                   m.radians(0.0),    # RX
                                   m.radians(180.0),  # RY
                                   m.radians(90.0))   # RZ

    # transform pt form compas to robot space
    scan_pt_robot = uu.compas_to_robot_space(scan_pt)
    scan_frame_robot = cg.Frame(scan_pt_robot,
                                -cg.Vector.Xaxis(),
                                -cg.Vector.Yaxis())

    # rotate frame to get vertical camera pose
    way_frame = uu.rotate_scan_pose_frame(scan_frame_robot,
                                          degree_x,
                                          degree_y,
                                          degree_z)

    # generate ur script
    script += us.move_l_blend(way_frame, acc, vel, rad)
    script = us.concatenate_script(script)

    return script


def get_toolpath(count):
    # get toolpath
    figsize_x, figsize_y, figsize_z = uu.get_figsize()
    sandbox2d_size_x, sandbox2d_size_y = uu.get_sandbox2d_size
    compas_frame = cg.Frame.worldXY()
    random_line = rtg.random_line_gen(compas_frame, figsize_x, figsize_y)
    # save toolpath as a image
    rtg.save_line_image(random_line, figsize_x, figsize_y, iteration=count)
    # provide toolpath
    box_size_x, box_size_y, box_size_z = uu.get_sandbox_size()
    random_toolpath, x_coord, y_coord = rtg.deploy_fig_to_box(random_line,
                                                              figsize_x,
                                                              figsize_y,
                                                              figsize_z,
                                                              box_size_x,
                                                              box_size_y,
                                                              box_size_z)
    # remap value into index
    x_ind = int(uu.remapvalue(x_coord,
                              int(figsize_x/2), int(box_size_x-figsize_x/2),
                              0, sandbox2d_size_y-1))
    y_ind = int(uu.remapvalue(y_coord,
                              int(figsize_y/2), int(box_size_y-figsize_y/2),
                              0, sandbox2d_size_x-1))
    return random_toolpath, x_ind, y_ind


def get_z_fig(pcl, x_ind, y_ind):
    sandbox2d_size_x, sandbox2d_size_y = uu.get_sandbox2d_size
    arr = np.asarray(pcl.points)

    arr_re = np.reshape(arr, (sandbox2d_size_x, sandbox2d_size_y, 3))
    z_toolpathbox2D = arr_re[y_ind][x_ind][2] + 289 - 35
    # z_value from pendant - thickness of tcp

    return z_toolpathbox2D


def adapt_toolpath(random_toolpath, z_toolpathbox2D):
    random_toolpath_adapted = None
    return random_toolpath_adapted


def move_robot_along_line(random_toolpath, z_toolpathbox2D,
                          velocity=0.15, acceleration=0.05, radius=0.01,
                          safety_dist=50):
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
    way_frames = [uu.compas_to_robot_space(f) for f in frames]

    for frame in way_frames:
        script += us.move_l_blend(frame, acceleration, velocity, radius)

    script = us.concatenate_script(script)
    return script


def move_robot_to_a_frame(move_to, z_center_toolpathbox2D,
                          velocity=0.30, acceleration=0.10, radius=0.0):

    script = ""
    script += us.set_tcp_by_angles(0.0,               # X
                                   0.0,               # Y
                                   facts['tcp_len'],       # Z
                                   m.radians(0.0),    # RX
                                   m.radians(0.0),  # RY
                                   m.radians(0.0))   # RZ

    # add transform frame
    way_frame = uu.compas_to_robot_space(move_to)

    # adapt height
    way_frame_adapted = adapt_height_from_pcl(way_frame,
                                              z_center_toolpathbox2D)
    script += us.move_l_blend(way_frame_adapted,
                              acceleration,
                              velocity,
                              radius)

    script = us.concatenate_script(script)
    return script


def execute_toolpath(random_toolpath, z_toolpathbox2D, excavation_time=15):
    print('excavation START')
    script_move = move_robot_along_line(random_toolpath,
                                        z_toolpathbox2D)
    uc.send_script(facts['robot_ip'],
                   facts['ur_server_port'],
                   bytes(script_move, 'utf-8'))
    time.sleep(excavation_time)
    print('excavation DONE')


def scan_pose(scanning_time=7.5, z_center_toolpathbox2D=0):
    print('scanning START')
    scan_frame = uu.get_scan_frame()
    script_scan = move_robot_to_a_frame(scan_frame, z_center_toolpathbox2D)
    uc.send_script(facts['robot_ip'],
                   facts['ur_server_port'],
                   bytes(script_scan, 'utf-8'))
    time.sleep(scanning_time)


def test_pose(move_to, z_center_toolpathbox2D=0):
    script_test = move_robot_to_a_frame(move_to, z_center_toolpathbox2D)
    uc.send_script(facts['robot_ip'],
                   facts['ur_server_port'],
                   bytes(script_test, 'utf-8'))


def adapt_height_from_pcl(way_frame, z_center_toolpathbox2D):
    way_frame.point.z += (z_center_toolpathbox2D + 0)
    return way_frame


if __name__ == "__main__":
    scan_pose(scanning_time=0.5)

    # calibration_frame = uu.get_calibration_frame()
    # scan_frame = uu.get_scan_frame()
    # test_pose(scan_frame)
    pass
