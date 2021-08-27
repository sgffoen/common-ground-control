import compas.geometry as cg
import math

# FUNCTIONS
'''
-ur_script_functions
    move_l
    move_l_blend
    move_j
    set_tcp_by_angle
    popup
    sleep
    set_digital_out
    concatenate_script
-numerical_function
    matrix_to_angle
'''


def move_l(plane_to, accel, vel,
           max_acc=1.5, max_vel=2.0):
    """
    Function that returns UR script for linear movement in tool-space.

    Args:
        plane_to: compas.geometry Frame
        accel: tool accel in m/s^2
        vel: tool speed in m/s

    Returns:
        script: UR script
    """

    # Check acceleration and velocity are non-negative and below a set limit
    accel = max_acc if (abs(accel) > max_acc) else abs(accel)
    vel = max_vel if (abs(vel) > max_vel) else abs(vel)

    worldXY = cg.Frame.worldXY()
    _matrix = cg.Transformation.from_frame_to_frame(worldXY, plane_to)
    _axis_angle = matrix_to_axis_angle(_matrix)

    # Create pose data
    _pose = [plane_to.point[0]/1000,  # X
             plane_to.point[1]/1000,  # Y
             plane_to.point[2]/1000,  # Z
             _axis_angle[0],          # RX
             _axis_angle[1],          # RY
             _axis_angle[2]]          # RZ
    _pose_fmt = "p[" + ("%.4f,"*6)[:-1]+"]"
    _pose_fmt = _pose_fmt % tuple(_pose)

    # Format UR script
    script = "movel(%s, a = %.2f, v = %.2f)\n" % (_pose_fmt, accel, vel)
    return script


def move_l_blend(plane_to, accel, vel,
                 blend_radius=0, max_acc=1.5, max_vel=2.0):
    """
    Function that returns UR script for linear movement in tool-space.

    Args:
        plane_to: Rhino.Geometry Plane. A target plane for calculating pose.
        accel: tool accel in m/s^2
        vel: tool speed in m/s

    Returns:
        script: UR script
    """

    # Check acceleration and velocity are non-negative and below a set limit
    accel = max_acc if (abs(accel) > max_acc) else abs(accel)
    vel = max_vel if (abs(vel) > max_vel) else abs(vel)
    # Check blend radius is positive
    blend_radius = max(0, blend_radius)

    worldXY = cg.Frame.worldXY()
    _matrix = cg.Transformation.from_frame_to_frame(worldXY, plane_to)
    _axis_angle = matrix_to_axis_angle(_matrix)

    # Create pose data
    _pose = [plane_to.point[0]/1000,  # X
             plane_to.point[1]/1000,  # Y
             plane_to.point[2]/1000,  # Z
             _axis_angle[0],          # RX
             _axis_angle[1],          # RY
             _axis_angle[2]]          # RZ
    _pose_fmt = "p[" + ("%.4f,"*6)[:-1]+"]"
    _pose_fmt = _pose_fmt % tuple(_pose)

    # Format UR script
    script = "movel(%s, a = %.2f, v = %.2f, r = %.4f)\n" % (_pose_fmt,
                                                            accel,
                                                            vel,
                                                            blend_radius)

    return script


def move_j(joints, accel, vel):
    """
    Function that returns UR script for linear movement in joint space.

    Args:
        joints: A list of 6 joint angles (double).
        accel: tool accel in m/s^2
        accel: tool accel in m/s^2
        vel: tool speed in m/s

    Returns:
        script: UR script
    """
    # Check acceleration and velocity are non-negative and below a set limit
    _j_fmt = "[" + ("%.2f,"*6)[:-1]+"]"
    _j_fmt = _j_fmt % tuple(joints)
    script = "movej(%s, a = %.2f, v = %.2f)\n" % (_j_fmt, accel, vel)

    return script


def set_tcp_by_angles(x_offset, y_offset, z_offset,
                      x_rotate, y_rotate, z_rotate):
    """
    Function that returns UR script for setting tool center point

    Args:
        x_offset: float. tooltip offset in mm
        y_offset: float. tooltip offset in mm
        z_offset: float. tooltip offset in mm
        x_rotation: float. rotation around world x axis in radians
        y_rotation: float. rotation around world y axis in radians
        z_rotation: float. rotation around world z axis in radians

    Returns:
        script: UR script
    """

    # Create rotation matrix
    RX = cg.Rotation.from_axis_and_angle(cg.Vector.Xaxis(), x_rotate)
    RY = cg.Rotation.from_axis_and_angle(cg.Vector.Yaxis(), y_rotate)
    RZ = cg.Rotation.from_axis_and_angle(cg.Vector.Zaxis(), z_rotate)
    R = RZ * RY * RX

    # _axis_angle= R.euler_angles()
    _axis_angle = matrix_to_axis_angle(R)

    # Create pose data
    _pose = [x_offset/1000, y_offset/1000, z_offset/1000,
             _axis_angle[0], _axis_angle[1], _axis_angle[2]]
    _pose_fmt = "p[" + ("%.4f,"*6)[:-1]+"]"
    _pose_fmt = _pose_fmt % tuple(_pose)

    # Format UR script
    script = "set_tcp(%s)\n" % (_pose_fmt)

    return script


def popup(message, title):
    """
    Function that returns UR script for popup

    Args:
        message: float. tooltip offset in mm
        title: float. tooltip offset in mm

    Returns:
        script: UR script
    """

    script = 'popup("%s","%s") \n' % (message, title)

    return script


def sleep(time):
    """
    Function that returns UR script for sleep()

    Args:
        time: float.in s

    Returns:
        script: UR script
    """

    script = "sleep(%s) \n" % (time)

    return script


def set_digital_out(id, signal):
    """
    Function that returns UR script for setting digital out

    Args:
        id: int. Input id number
        signal: boolean. signal level - on or off

    Returns:
        script: UR script
    """

    # Format UR script
    script = "set_digital_out(%s,%s)\n" % (id, signal)

    return script


def concatenate_script(list_ur_commands):
    """
    Internal function that concatenates generated UR script
    into one large script file. Usually used to combine
    scripts generated by the GrasshopperPython components

    Args:
        list_ur_commands: A list of formatted UR Script strings

    Returns:
        ur_script: The concatenated script
    """

    ur_script = "\ndef my_script():\n"
    # ur_script += '\tpopup("running my_script")\n'

    combined_script = ""
    for ur_cmd in list_ur_commands:
        combined_script += ur_cmd

    # format combined script
    lines = combined_script.split("\n")
    for line in lines:
        ur_script += "\t" + line + "\n"

    ur_script += 'end\n'
    ur_script += '\nmy_script()\n'
    return ur_script


def matrix_to_axis_angle(m):
    """
    Function that transforms a 4x4 matrix to axis-angle format
    referenced from Martin Baker's www.euclideanspace.com

    Args:
        m: Rhino.Geometry Transform structure  - 4x4 matrix

    Returns:
        axis: Rhino.Geometry Vector3d object - axis-angle notation
    """

    epsilon = 0.01
    epsilon2 = 0.01

    if ((math.fabs(m[0, 1] - m[1, 0]) < epsilon) &
       (math.fabs(m[0, 2] - m[2, 0]) < epsilon) &
       (math.fabs(m[1, 2] - m[2, 1]) < epsilon)):
        # singularity found
        # first check for identity matrix which must have +1 for all terms
        # in leading diagonal and zero in other terms
        if ((math.fabs(m[0, 1] + m[1, 0]) < epsilon2) &
           (math.fabs(m[0, 2] + m[2, 0]) < epsilon2) &
           (math.fabs(m[1, 2] + m[2, 1]) < epsilon2) &
           (math.fabs(m[0, 0] + m[1, 1] + m[2, 2] - 3) < epsilon2)):
            # this singularity is identity matrix so angle = 0
            # make zero angle, arbitrary axis
            angle = 0
            x = 1
            y = z = 0
        else:
            # otherwise this singularity is angle = 180
            angle = math.pi
            xx = (m[0, 0] + 1) / 2
            yy = (m[1, 1] + 1) / 2
            zz = (m[2, 2] + 1) / 2
            xy = (m[0, 1] + m[1, 0]) / 4
            xz = (m[0, 2] + m[2, 0]) / 4
            yz = (m[1, 2] + m[2, 1]) / 4
            if ((xx > yy) & (xx > zz)):
                # m[0,0] is the largest diagonal term
                if (xx < epsilon):
                    x = 0
                    y = z = 0.7071
                else:
                    x = math.sqrt(xx)
                    y = xy / x
                    z = xz / x
            elif (yy > zz):
                # m[1,1] is the largest diagonal term
                if (yy < epsilon):
                    x = z = 0.7071
                    y = 0
                else:
                    y = math.sqrt(yy)
                    x = xy / y
                    z = yz / y
            else:
                # m[2,2] is the largest diagonal term so base result on this
                if (zz < epsilon):
                    x = y = 0.7071
                    z = 0
                else:
                    z = math.sqrt(zz)
                    x = xz / z
                    y = yz / z
    else:
        s = math.sqrt((m[2, 1] - m[1, 2]) * (m[2, 1] - m[1, 2]) +
                      (m[0, 2] - m[2, 0]) * (m[0, 2] - m[2, 0]) +
                      (m[1, 0] - m[0, 1]) * (m[1, 0] - m[0, 1]))
        if (math.fabs(s) < 0.001):
            # prevent divide by zero,
            # should not happen if matrix is orthogonal and should be
            s = 1
        angle = math.acos((m[0, 0] + m[1, 1] + m[2, 2] - 1) / 2)
        x = (m[2, 1] - m[1, 2]) / s
        y = (m[0, 2] - m[2, 0]) / s
        z = (m[1, 0] - m[0, 1]) / s
    angleRad = angle
    axis = cg.Vector(x, y, z)
    axis = axis*angleRad

    return axis
