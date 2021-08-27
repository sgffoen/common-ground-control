"""
This module contains communication functions:
    1) concatenate script
    2) send & read
"""


# LIBRARIES


import socket
import math
import traceback
from struct import unpack


# FUNCTIONS
'''
send_script
listen_to_robot
set_tcp_by_plane (wip)
set_tcp_by_angle
popup
sleep
'''


def send_script(ur_ip, UR_SERVER_PORT, script):
    """
    Function that send script to robot through socket(Ethernet cable).

    Args:
        ur_ip : local host IP. (check UR pendant)
        UR_SERVER_PORT : ur server port (check UR pendant)
        script : UR script to be sent

    Returns: NONE
    """
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.settimeout(2)
    try:
        s.connect((ur_ip, UR_SERVER_PORT))
    except BaseException:
        print("Cannot connect to ", ur_ip, UR_SERVER_PORT)

    s.settimeout(None)
    try:
        s.send(script)
        print("Script sent to %s on port %i" % (ur_ip, UR_SERVER_PORT))
    except BaseException:
        print("failed to send")

    s.close()


def listen_to_robot(robot_ip):
    PORT = 30003
    HOST = robot_ip
    # Create dictionary to store data
    chunks = {}
    chunks["target_joints"] = []
    chunks["actual_joints"] = []
    chunks["forces"] = []
    chunks["pose"] = []
    chunks["time"] = []

    data = read(HOST, PORT)
    get_messages(data, chunks)
    return chunks


def read(HOST, PORT):
    """
    Method that opens a TCP socket to the robot,
    receives data from the robot server and then closes socket

    Returns:
        data: Data broadcast by the robot. In bytes
    """

    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.settimeout(1)
    try:
        s.connect((HOST, PORT))
        # print("connected for reading")
    except BaseException:
        traceback.print_exc()
        print("Cannot connect to ", HOST, PORT)
    s.settimeout(None)
    data = s.recv(1024)
    s.close()
    return data


def get_messages(bytes, chunks_info):
    """
    Function parses data stream and selects the following information:
    1) q_target
    2) q_actual
    3) TCP force
    4) Tool Vector
    5) Time

    This data is formatted and the chunks dictionary is updated
    for more info see:
    http://wiki03.lynero.net/Technical/RealTimeClientInterface
    """
    # get messages
    q_target = bytes[12:60]
    q_actual = bytes[252:300]
    tcp_force = bytes[540:588]
    tool_vector = bytes[588:636]
    controller_time = bytes[740:748]

    # format type: int,
    fmt_double6 = "!dddddd"
    fmt_double1 = "!d"

    # Unpack selected data
    target_joints = unpack(fmt_double6, q_target)
    chunks_info["target_joints"] = (math.degrees(j) for j in target_joints)
    actual_joints = unpack(fmt_double6, q_actual)
    chunks_info["actual_joints"] = (math.degrees(j) for j in actual_joints)
    forces = unpack(fmt_double6, tcp_force)
    chunks_info["forces"] = forces
    pose = unpack(fmt_double6, tool_vector)
    chunks_info["pose"] = pose
    time = unpack(fmt_double1, controller_time)
    chunks_info["time"] = time
