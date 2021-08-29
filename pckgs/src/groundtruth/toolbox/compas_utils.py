# LIBRARIES


import compas.geometry as cg
import compas.datastructures as cd
from compas_view2.app import App
import math
from .helper import Facts

# set global facts
__FACTS__ = Facts().facts


# FUNCTIONS
'''
-get_value_functions
    get_robot_corner_pts
    get_sandbox_size
    get_robot_frame
    get_scan_frame
    get_calibration_frame
    get_figsize
    get_sandbox2d_size
-transformations
    compas_to_robot_space
    robot_to_compas_space
    rotat_frame_for_scan
-numerical_functions
    remmap_value
    matrix_to_axis_angle
    matrix_to_euler
    concatenate_matrices
-extra_funcitons
    add_safety_frames
    reverse_z_value
    adapt_height_from_pcl
    check_arguments
    run_viewer
-gh_interface
    draw_feature
'''


def get_robot_corner_pts():
    robot_corner_pts = __FACTS__.robot_corner_pts
    pt_o = cg.Point(robot_corner_pts['pt0'][0],
                    robot_corner_pts['pt0'][1],
                    robot_corner_pts['pt0'][2])
    pt_x = cg.Point(robot_corner_pts['pt0'][0],
                    robot_corner_pts['ptx'][1],
                    robot_corner_pts['ptx'][2])
    pt_y = cg.Point(robot_corner_pts['pty'][0],
                    robot_corner_pts['pt0'][1],
                    robot_corner_pts['pty'][2])
    return pt_o, pt_x, pt_y


def get_sandbox_size():
    pt_o, pt_x, pt_y = get_robot_corner_pts()
    box_size_x = abs(pt_x.x - pt_o.x)
    box_size_y = abs(pt_y.y - pt_o.y)
    box_size_z = 0.
    return box_size_x, box_size_y, box_size_z


def get_robot_frame():
    pt_o, pt_x, pt_y = get_robot_corner_pts()
    robot_frame = cg.Frame(pt_o, pt_x-pt_o, pt_y-pt_o)
    return robot_frame


def get_scan_frame():
    # initialize frame
    scan_frame = cg.Frame.worldXY()
    # get scan_pt from facts
    scan_pt = __FACTS__.scan_pt
    # transformation
    RX = cg.Rotation.from_axis_and_angle(cg.Vector.Xaxis(),
                                         math.radians(scan_pt['rx']))
    RY = cg.Rotation.from_axis_and_angle(cg.Vector.Yaxis(),
                                         math.radians(scan_pt['ry']))
    RZ = cg.Rotation.from_axis_and_angle(cg.Vector.Zaxis(),
                                         math.radians(scan_pt['rz']))
    V = cg.Vector(scan_pt['x'],
                  scan_pt['y'],
                  scan_pt['z'])
    M = cg.Translation.from_vector(V)
    T = M * RX * RY * RZ
    # transform
    scan_frame.transform(T)
    return scan_frame


def get_calibration_frame():
    center = cg.Point(__FACTS__.calibration_pt['x'],
                      __FACTS__.calibration_pt['y'],
                      __FACTS__.calibration_pt['z'])
    calibration_frame = cg.Frame(center,
                                 cg.Vector.Xaxis(),
                                 cg.Vector.Yaxis())
    return calibration_frame


def get_figsize():
    figsize_x = __FACTS__.fig_size['x']
    figsize_y = __FACTS__.fig_size['y']
    figsize_z = __FACTS__.fig_size['z']

    return figsize_x, figsize_y, figsize_z


def get_sandbox2d_size():
    crop_ids = __FACTS__.crop_idx
    sandbox2d_size_x = crop_ids['xEnd'] - crop_ids['xStart']
    sandbox2d_size_y = crop_ids['yStart'] - crop_ids['yEnd']
    return sandbox2d_size_x, sandbox2d_size_y


def get_feature_bounds():
    feature_bounds = __FACTS__.feature_bounds
    min_bounds = feature_bounds['min_bound']
    max_bounds = feature_bounds['max_bound']
    return [min_bounds, max_bounds]


def compas_to_robot_space(geo):
    compas_origin_frame = cg.Frame.worldXY()
    robot_frame = get_robot_frame()
    M = cg.Transformation.from_frame_to_frame(compas_origin_frame,
                                              robot_frame)

    R = cg.Rotation.from_axis_and_angle(cg.Vector.Zaxis(), math.radians(180.))

    T = R.concatenated(M)

    geo_trans = geo.transformed(T)

    return geo_trans


def robot_to_compas_space(geo, robot_base, pure_trans=True):
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
    R = cg.Rotation.from_axis_and_angle(cg.Vector.Zaxis(), -math.radians(180.))

    if pure_trans:
        compas_origin_frame = cg.Frame.worldXY()
        M = cg.Transformation.from_frame_to_frame(robot_base,
                                                  compas_origin_frame)
        T = M.concatenated(R)
        geo_trans = geo.transformed(T)
    else:
        geo_trans = geo.transformed(R)

    return geo_trans


def rotate_scan_pose_frame(frame, degree_x, degree_y, degree_z):
    # rotate frame to get vertical camera pose
    RX = cg.Rotation.from_axis_and_angle(-cg.Vector.Xaxis(),
                                         math.radians(degree_x))
    RY = cg.Rotation.from_axis_and_angle(-cg.Vector.Yaxis(),
                                         math.radians(degree_y))
    RZ = cg.Rotation.from_axis_and_angle(cg.Vector.Zaxis(),
                                         math.radians(degree_z))
    T = RX * RY * RZ
    transformed_frame = frame.transformed(T)

    return transformed_frame


def remap_value(v, ori_Min, ori_Max, targetMin, targetMax):
    rv = ((v-ori_Min)/(ori_Max-ori_Min))*(targetMax-targetMin)+targetMin
    return rv


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


def add_safety_frames(frames, safety_dist=200):
    frame_s = frames[0].copy()
    frame_e = frames[-1].copy()
    frame_s.point.z += safety_dist
    frame_e.point.z += safety_dist
    frames.insert(0, frame_s)
    frames.append(frame_e)
    return frames


def reverse_z_value(frames):
    for f in frames:
        f.point.z *= (-1)


def adapt_height_from_pcl(frames, z_center_toolpathbox2D=0):
    for f in frames:
        f.point.z += z_center_toolpathbox2D


def check_arguments(function):
    def decorated(*args):
        if None in args:
            raise TypeError("Invalid Argument")
        return function(*args)
    return decorated


def run_viewer(geos):
    viewer = App()
    for g in geos:
        viewer.add(g)
    viewer.run()
