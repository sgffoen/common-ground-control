# LIBRARIES
from UR import compas_simple_comm as uc
from UR import compas_simple_ur_script as us
from UR import compas_utils as uu

import math as m
import time
import json

# fact sheet
with open('data/facts.json') as f:
    facts = json.load(f)


# FUNCTIONS


def move_robot_to_a_frame(frame,
                          velocity=0.30,
                          acceleration=0.10,
                          radius=0.0):
    # initialize tcp
    script = ""
    script += us.set_tcp_by_angles(0.0,               # X
                                   0.0,               # Y
                                   facts['tcp_len'],  # Z
                                   m.radians(0.0),    # RX
                                   m.radians(0.0),    # RY
                                   m.radians(0.0))    # RZ
    # transform frame from compas to robot coord
    way_frame = uu.compas_to_robot_space(frame)
    # write a ur script
    script += us.move_l_blend(way_frame,
                              acceleration,
                              velocity,
                              radius)
    script = us.concatenate_script(script)
    return script


def move_robot_to_frames(frames,
                         velocity=0.30,
                         acceleration=0.10,
                         radius=0.01):
    # initialize tcp
    script = ""
    script += us.set_tcp_by_angles(0.0,               # X
                                   0.0,               # Y
                                   facts['tcp_len'],  # Z
                                   m.radians(0.0),    # RX
                                   m.radians(0.0),    # RY
                                   m.radians(0.0))    # RZ
    # transform frame from compas to robot
    way_frames = [uu.compas_to_robot_space(f) for f in frames]
    # write ur scripts
    for frame in way_frames:
        script += us.move_l_blend(frame,
                                  acceleration,
                                  velocity,
                                  radius)
    script = us.concatenate_script(script)
    return script


def scan_pose(scanning_time=7.5):
    print('scanning START')
    scan_frame = uu.get_scan_frame()
    script_scan = move_robot_to_a_frame(scan_frame)
    uc.send_script(facts['robot_ip'],
                   facts['ur_server_port'],
                   bytes(script_scan, 'utf-8'))
    time.sleep(scanning_time)


def test_pose(frame, z_center_toolpathbox2D=0):
    uu.adapt_height_from_pcl(frame, z_center_toolpathbox2D=0)
    script_test = move_robot_to_a_frame(frame)
    uc.send_script(facts['robot_ip'],
                   facts['ur_server_port'],
                   bytes(script_test, 'utf-8'))


def calibration_pose():
    print('scanning START')
    scan_frame = uu.get_calibration_frame()
    script_scan = move_robot_to_a_frame(scan_frame)
    uc.send_script(facts['robot_ip'],
                   facts['ur_server_port'],
                   bytes(script_scan, 'utf-8'))


def execute_toolpath(frames, z_center_toolpathbox2D=0):
    print('toolpath START')
    frames = uu.add_safety_frames(frames)
    uu.adapt_height_from_pcl(frames, z_center_toolpathbox2D=0)
    uu.reverse_z_value(frames)
    script_scan = move_robot_to_frames(frames)
    uc.send_script(facts['robot_ip'],
                   facts['ur_server_port'],
                   bytes(script_scan, 'utf-8'))


if __name__ == "__main__":
    pass
