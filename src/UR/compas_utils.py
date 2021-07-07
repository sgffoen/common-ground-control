"""
This module contains utility functions:
    1) Transformation functions
    2) Useful geometry functions e.g. Intersections
    3) visualization function
"""


# LIBRARIES


import compas.geometry as cg
from compas_view2.app import App
import math
import os


# FUNCTIONS
'''
1) run_viewer
2) set_robot_base
3) compas_to_robot_space
4) matrix_to_axis_angle
5) matrix_to_euler
6) dh_matrix (WIP)
7) concatenate_matrices
8) signed_angle (WIP)
9) cir_cir_intersection (WIP)
10) check_arguments
'''


def run_viewer(geo):
    """
    Function that visualize geometries on compas_viewer2.

    Args:
        geo : a list of compas.geometry. Geometries to be visualised

    Returns: NONE
    """

    viewer = App()
    for g in geo:
        viewer.add(g)
    viewer.run()


def set_robot_base(ORIGIN, X_POINT, Y_POINT):
    """
    Function that returns robot base as compas.geometry Frame.

    Args:
        ORIGIN : compas.geometry Point (left upper corner of sandbox)
        X_POINT: compas.geometry Point (left bottom corner of sandbox)
        Y_POINT: compas.geometry Point (right bottom corner of sandbox)

    Returns:
        robot_base : compas.geometry Frame as a robot base frame
    """

    robot_base = cg.Frame(ORIGIN, X_POINT-ORIGIN, Y_POINT-ORIGIN)
    return robot_base


def compas_to_robot_space(geo, robot_base, pure_trans=False):
    """
    Function that returns compas.geometry
    after Transformation from rhino to robot space.

    Args:
        geo : compas.geometry. A geometry to be transformed
        robot_base : compas.geometry Frame. A robot base.
        pure_trans : if True, from pure compas to robot.
                     if false, from physical robot to digital robot

    Returns:
        geo_trans : compas.geometry. A geometry after transformation.
    """
    R = cg.Rotation.from_axis_and_angle(cg.Vector.Zaxis(), math.radians(180.))

    if pure_trans:
        compas_origin_frame = cg.Frame.worldXY()
        M = cg.Transformation.from_frame_to_frame(compas_origin_frame,
                                                  robot_base)
        T = R.concatenated(M)
        geo_trans = geo.transformed(T)
    else:
        geo_trans = geo.transformed(R)

    return geo_trans


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


def matrix_to_euler(m):
    """
    Gets the Euler rotation angles from a transformation matrix
    from http://forums.codeguru.com/archive/index.php/t-329530.html

    Args:
        m = Transform object
    Returns:
        tuple of euler angles in radians
    """

    rotz = math.atan2(m[1, 0], m[0, 0])
    roty = -math.asin(m[2, 0])
    rotx = math.atan2(m[2, 1], m[2, 2])
    return (rotx, roty, rotz)


'''
def dh_matrix(d, theta, a, alpha):
    """
    This function creates the Denavit Hartenberg transformation matrix
    between adjacent frames

    Arguments:
        d: Joint distance. in mm
        theta: joint angle. in radians
        a: link length. in mm
        alpha: twist angle. in radians

    Returns:
        m: Denavit Hartenberg transformation matrix
    """

    _matrix = [(math.cos(theta),
               -math.sin(theta) * math.cos(alpha),
               math.sin(theta) * math.sin(alpha),
               a * math.cos(theta)),
               (math.sin(theta),
               math.cos(theta) * math.cos(alpha),
               -math.cos(theta) * math.sin(alpha),
               a * math.sin(theta)),
               (0, math.sin(alpha),
               math.cos(alpha), d),
               (0, 0, 0, 1)]

    m = rg.Transform()
    for i in range(4):
        for j in range(4):
            m[i, j] = _matrix[i][j]

    return m
'''


def concatenate_matrices(matrices):
    """
    This function creates a concatenated matrix from a list of matrices

    Arguments:
        matrices: A list of tranformation matrices

    Returns:
        _transform: Concatenated matrix
    """
    _transform = matrices[0]
    for i in range(1, len(matrices)):
        _transform *= matrices[i]
    return _transform


'''
def signed_angle(v1, v2, v_normal):
    """
    This function gets the angle between 2 vectors -pi < theta< pi

    Arguments:
        v1: Vector3d. First unitized vector
        v2: Vector3d. Second unitized vector
        v_normal: Vector3d. Normal to 2 vectors that determines
                  what is positive/negative

    Returns:
        theta: float. signed angle between -pi and pi
    """
    # from 0 to pi
    c = rg.Vector3d.Multiply(v1, v2)
    n = rg.Vector3d.CrossProduct(v1, v2)
    s = n.Length

    theta = math.atan2(s, c)

    if (rg.Vector3d.Multiply(n, v_normal) < 0):
        theta *= -1
    return theta
'''


'''
def cir_cir_intersection(cir1, cir2):
    """
    Funtion that returns the intersection points between two circles

    Arguments:
        1) cir1: First circle
        2) cir2: Second Circle

    Returns:
        xpts: list of 2 Point3d objectts

    Note that there is no error checking
    """
    r1 = cir1.Radius
    r2 = cir2.Radius
    d = cir1.Center.DistanceTo(cir2.Center)

    a = (r1 ** 2 - r2**2 + d**2)/(2*d)
    h = math.sqrt(r1 ** 2 - a ** 2)

    v_c1 = rg.Vector3d(cir1.Center)
    v_c2 = rg.Vector3d(cir2.Center)

    v_c1c2 = v_c2 - v_c1
    v_c1c2.Unitize()
    v_c1c2 *= a

    v_pt0 = v_c1 + v_c1c2

    v_pt0ptX = rg.Vector3d.CrossProduct(cir1.Normal, v_c1c2)
    v_pt0ptX.Unitize()
    v_pt0ptX *= h

    xpt1 = rg.Point3d(v_pt0 + v_pt0ptX)
    v_pt0ptX.Reverse()
    xpt2 = rg.Point3d(v_pt0 + v_pt0ptX)

    return [xpt1, xpt2]
'''


def check_arguments(function):
    def decorated(*args):
        if None in args:
            raise TypeError("Invalid Argument")
        return function(*args)
    return decorated


def get_path():
    HERE = os.path.dirname(__file__)
    DIR = os.path.dirname(HERE)
    FILE = "/data/facts.json"
    PATH = DIR + FILE
    return PATH


def remapValue(v, ori_Min, ori_Max, targetMin, targetMax):
    rv = ((v-ori_Min)/(ori_Max-ori_Min))*(targetMax-targetMin)+targetMin
    return rv


def compas_to_robot_frame():
    pass

