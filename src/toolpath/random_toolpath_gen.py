# LIBRARIES

from compas_view2.app import App
from collections import deque
import compas.geometry as cg
import compas.utilities as cu
import random as r
import math as m
import numpy as np
import cv2
import json
import datetime
import os


# set global facts
with open('data/facts.json') as f:
    facts = json.load(f)


# FUNCTIONS
'''
list
'''


def generate_ctrl_pts_tuple(num_ctrl_pts, level):
    ctrl_pts_list = []

    if level == '1.0':
        length = 255
        step = length / (num_ctrl_pts - 1)
        for i in range(num_ctrl_pts):
            x = i * step
            y = 0
            z = 255
            ctrl_pts_list.append((x, y, z))

    elif level == '1.1':
        length = r.randrange(30, 255)
        step = length / (num_ctrl_pts - 1)
        z = r.randrange(0, 255)
        for i in range(num_ctrl_pts):
            x = i * step
            y = 0
            ctrl_pts_list.append((x, y, z))

    elif level == '1.2':
        length = r.randrange(30, 255)
        step = length / (num_ctrl_pts - 1)
        for i in range(num_ctrl_pts):
            x = i * step
            y = 0
            z = r.randrange(0, 255)
            ctrl_pts_list.append((x, y, z))

    elif level == '2.0':
        length = r.randrange(30, 255)
        step = length / (num_ctrl_pts - 1)
        for i in range(num_ctrl_pts):
            x = i * step
            y = r.randrange(0, 100)
            z = 255
            ctrl_pts_list.append((x, y, z))

    elif level == '2.1':
        length = r.randrange(30, 255)
        step = length / (num_ctrl_pts - 1)
        z = r.randrange(0, 255)
        for i in range(num_ctrl_pts):
            x = i * step
            y = r.randrange(50, 200)
            ctrl_pts_list.append((x, y, z))

    elif level == '2.2':
        length = r.randrange(30, 255)
        step = length / (num_ctrl_pts - 1)
        for i in range(num_ctrl_pts):
            x = i * step
            y = r.randrange(50, 200)
            z = r.randrange(0, 255)
            ctrl_pts_list.append((x, y, z))

    return ctrl_pts_list


def draw_a_polyline(tuple_list):
    ctrl_pts = [cg.Point(tl[0], tl[1], tl[2]) for tl in tuple_list]
    polyline = cg.Polyline(ctrl_pts)
    return polyline


def draw_a_bezier(tuple_list):
    ctrl_pts = [cg.Point(tl[0], tl[1], tl[2]) for tl in tuple_list]
    curve = cg.Bezier(ctrl_pts)
    return curve


def tuple_to_compas_frame(tuple_list, curve_type, segments_num):
    compas_frames = []

    if curve_type == 'polyline':
        curve = draw_a_polyline(tuple_list)
        pts_on_curve = curve.divide_polyline(segments_num)

    elif curve_type == 'bezier':
        curve = draw_a_bezier(tuple_list)

        pts_on_curve = []
        step = 1 / (segments_num-1)
        for i in range(segments_num):
            t = step * i
            pt_on_curve = curve.point(t)
            pts_on_curve.append(pt_on_curve)

    for a, b in cu.pairwise(range(len(pts_on_curve))):
        # get first pt
        pta = cg.Point(pts_on_curve[a][0],
                       pts_on_curve[a][1],
                       pts_on_curve[a][2])
        # get end pt
        ptb = cg.Point(pts_on_curve[b][0],
                       pts_on_curve[b][1],
                       pts_on_curve[b][2])
        # calc axis on xy plane
        xaxis = cg.Vector.from_start_end(pta, ptb)
        yaxis = cg.Vector.Zaxis().cross(xaxis)
        # flatten vectors
        xaxis.z = 0.
        yaxis.z = 0.
        compas_frames.append(cg.Frame(pta, yaxis, -xaxis))

    return compas_frames


def rotate_ctrl_frames(frames):
    # set angle
    degree = r.randint(-180, 180)
    # get rotation center
    R = cg.Rotation.from_axis_and_angle(cg.Vector.Zaxis(), m.radians(degree))
    for f in frames:
        f.transform(R)
    return frames


def get_sandbox_size():
    # get corner pts
    pts = facts['robot_corner_pts']
    pt0 = cg.Point(pts['pt0'][0], pts['pt0'][1], pts['pt0'][2])
    ptx = cg.Point(pts['ptx'][0], pts['ptx'][1], pts['ptx'][2])
    pty = cg.Point(pts['pty'][0], pts['pty'][1], pts['pty'][2])
    # get size
    sandbox_size_x = cg.distance_point_point_xy(pt0, ptx)
    sandbox_size_y = cg.distance_point_point_xy(pt0, pty)

    return sandbox_size_x, sandbox_size_y


def get_toolpathbox2d_size():
    return (facts['fig_size']['x'],
            facts['fig_size']['y'],
            facts['fig_size']['z'])


def calc_offset_area_sandbox2d():
    # set coordinates' range
    sandbox_size_x, sandbox_size_y = get_sandbox_size()
    (toolpathbox2d_size_x,
     toolpathbox2d_size_y,
     toolpathbox2d_size_z) = get_toolpathbox2d_size()
    # calculate half of diagonal length of toolpathbox2d
    x_offset_dist = toolpathbox2d_size_x / m.sqrt(2)
    y_offset_dist = toolpathbox2d_size_y / m.sqrt(2)
    # set min/max of working area
    x_coord_min = x_offset_dist
    x_coord_max = sandbox_size_x - x_offset_dist
    y_coord_min = y_offset_dist
    y_coord_max = sandbox_size_y - y_offset_dist
    return x_coord_min, x_coord_max, y_coord_min, y_coord_max


def move_ctrl_frames_to_sandbox2d(ctrl_frames_rotation):
    # set coordinates range
    (x_coord_min, x_coord_max,
     y_coord_min, y_coord_max) = calc_offset_area_sandbox2d()
    # generate target frame to move to
    frame_to_x = r.randint(int(x_coord_min), int(x_coord_max))
    frame_to_y = r.randint(int(y_coord_min), int(y_coord_max))
    frame_to_center = cg.Point(frame_to_x, frame_to_y, 0)
    frame_to = cg.Frame(frame_to_center, cg.Vector.Xaxis(), cg.Vector.Yaxis())
    # generate frame at middle of the toolpath
    toolpath_line = cg.Line(ctrl_frames_rotation[0].point,
                            ctrl_frames_rotation[-1].point)
    toolpath_midpt = toolpath_line.midpoint
    toolpath_midpt.z = 0
    frame_from = cg.Frame(toolpath_midpt, cg.Vector.Xaxis(), cg.Vector.Yaxis())
    # create transformation
    T = cg.Transformation.from_frame_to_frame(frame_from, frame_to)
    for ctrl_frame in ctrl_frames_rotation:
        ctrl_frame.transform(T)

    return ctrl_frames_rotation


def run_viewer(geos):
    viewer = App()
    for g in geos:
        viewer.add(g)
    viewer.run()


def calc_contour(arr_sandbox2d_with_toolpath):
    img_gray = cv2.cvtColor(arr_sandbox2d_with_toolpath, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(src=img_gray,
                                thresh=127.5,
                                maxval=255,
                                type=cv2.THRESH_BINARY)
    contours, hierarchy = cv2.findContours(thresh, 1, 2)
    cnt = contours[0]
    return cnt


def calc_two_bounding_box(arr_sandbox2d_with_toolpath, cnt, text, show):
    # draw orthogonal bbox
    corner_pts_ortho = cv2.boundingRect(cnt)
    x, y, w, h = corner_pts_ortho
    rect = cv2.minAreaRect(cnt)
    corner_pts_orient_float = cv2.boxPoints(rect)

    if show:
        cv2.rectangle(arr_sandbox2d_with_toolpath,
                      (x, y),
                      (x+w, y+h),
                      (0, 255, 0),
                      1)
        # draw oriented bbox
        corner_pts_orient_int = np.int0(corner_pts_orient_float)
        cv2.drawContours(arr_sandbox2d_with_toolpath,
                         [corner_pts_orient_int],
                         0,
                         (0, 0, 255),
                         2)
        # put text at the corner
        if text:
            for i, pt in enumerate(corner_pts_orient_int):
                cv2.putText(arr_sandbox2d_with_toolpath,
                            str(i),
                            (pt[0], pt[1]),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1.0,
                            (0, 0, 0),
                            1,
                            cv2.LINE_AA)
        cv2.imshow('calc_two_bounding_box', arr_sandbox2d_with_toolpath)
        cv2.waitKey(0)

    return rect, corner_pts_orient_float


def calc_oriented_corner_pts_in_figsize(oriented_bbox):
    centroid, ratio, angle = oriented_bbox
    fig_size_x, fig_size_y, fig_size_z = get_toolpathbox2d_size()
    fig_size_diagonal = (fig_size_x * m.sqrt(2)) / 2

    if angle < 45:
        theta = m.radians(45-angle)
        pt0 = [centroid[0] - fig_size_diagonal*m.cos(theta),
               centroid[1] + fig_size_diagonal*m.sin(theta)]
        pt1 = [centroid[0] - fig_size_diagonal*m.sin(theta),
               centroid[1] - fig_size_diagonal*m.cos(theta)]
        pt2 = [centroid[0] + fig_size_diagonal*m.cos(theta),
               centroid[1] - fig_size_diagonal*m.sin(theta)]
        pt3 = [centroid[0] + fig_size_diagonal*m.sin(theta),
               centroid[1] + fig_size_diagonal*m.cos(theta)]

    else:
        theta = m.radians(angle-45)
        pt0 = [centroid[0] - fig_size_diagonal*m.cos(theta),
               centroid[1] - fig_size_diagonal*m.sin(theta)]
        pt1 = [centroid[0] + fig_size_diagonal*m.sin(theta),
               centroid[1] - fig_size_diagonal*m.cos(theta)]
        pt2 = [centroid[0] + fig_size_diagonal*m.cos(theta),
               centroid[1] + fig_size_diagonal*m.sin(theta)]
        pt3 = [centroid[0] - fig_size_diagonal*m.sin(theta),
               centroid[1] + fig_size_diagonal*m.cos(theta)]

    return [pt0, pt1, pt2, pt3]


def calc_toolpath_dir(ctrl_frames_in_sandbox2d):
    # calc direction of toolpath
    toolpath_start = ctrl_frames_in_sandbox2d[0].point
    toolpath_end = ctrl_frames_in_sandbox2d[-1].point
    toolpath_dir = cg.Vector.from_start_end(toolpath_start, toolpath_end)
    return toolpath_dir


def calc_toolpath_box_dir(corner_pts_orient_float):
    # calc direction of oriented_rect
    oriented_rect_pt0 = cg.Point(corner_pts_orient_float[0][0],
                                 corner_pts_orient_float[0][1],
                                 0)
    oriented_rect_pt1 = cg.Point(corner_pts_orient_float[1][0],
                                 corner_pts_orient_float[1][1],
                                 0)
    oriented_rect_pt2 = cg.Point(corner_pts_orient_float[2][0],
                                 corner_pts_orient_float[2][1],
                                 0)
    oriented_rect_pt3 = cg.Point(corner_pts_orient_float[3][0],
                                 corner_pts_orient_float[3][1],
                                 0)
    vec_toolpathbox_pt0_to_pt2 = cg.Vector.from_start_end(oriented_rect_pt0,
                                                          oriented_rect_pt2)
    vec_toolpathbox_pt1_to_pt3 = cg.Vector.from_start_end(oriented_rect_pt1,
                                                          oriented_rect_pt3)
    return vec_toolpathbox_pt0_to_pt2, vec_toolpathbox_pt1_to_pt3


def calc_dot_toolpath_and_bbox(toolpath_dir, edge1_dir, edge2_dir):
    # check orientation of toolpath and crop_area
    toolpath_dir.unitize()
    edge1_dir.unitize()
    edge2_dir.unitize()
    dot1 = cg.dot_vectors_xy(toolpath_dir, edge1_dir)
    dot2 = cg.dot_vectors_xy(toolpath_dir, edge2_dir)
    return dot1, dot2


def calc_shift_number_bbox_corner(dot1, dot2):
    shift_num = 0
    if (0 < dot1) and (0 < dot2):
        shift_num = 0
    elif (dot1 < 0) and (dot2 < 0):
        # shift 2
        shift_num = 2
    elif (0 < dot1) and (dot2 < 0):
        # shift 1
        shift_num = 1
    elif (dot1 < 0) and (0 < dot2):
        # shift -1
        shift_num = -1
    return shift_num


def shift_list(list_to_shift, shift_num):
    collection_to_shift = deque(list_to_shift)
    collection_to_shift.rotate(shift_num)
    return list(collection_to_shift)


def align_bbox_with_toolpath_dir(
        ctrl_frames_in_sandbox2d,
        corner_pts_orient_float,
        oriented_bbox_corner_pts_in_figsize):
    toolpath_dir = calc_toolpath_dir(ctrl_frames_in_sandbox2d)
    (toolpathbox_pt0_to_pt2,
     toolpathbox_pt1_to_pt3) = calc_toolpath_box_dir(corner_pts_orient_float)
    dot1, dot2 = calc_dot_toolpath_and_bbox(toolpath_dir,
                                            toolpathbox_pt0_to_pt2,
                                            toolpathbox_pt1_to_pt3)
    shift_num = calc_shift_number_bbox_corner(dot1, dot2)
    oriented_bbox_corner_pts_in_figsize = shift_list(
        oriented_bbox_corner_pts_in_figsize,
        shift_num)
    return oriented_bbox_corner_pts_in_figsize


def crop_toolpathbox2d_oriented(
        arr_sandbox2d_with_toolpath,
        oriented_bbox_corner_pts_in_figsize,
        show_cropped_img):
    pts_from = np.float32(oriented_bbox_corner_pts_in_figsize)
    (toolpathbox2d_size_x,
     toolpathbox2d_size_y,
     toolpathbox2d_size_z) = get_toolpathbox2d_size()
    pts_to = np.float32([[0, 0],
                         [toolpathbox2d_size_x, 0],
                         [toolpathbox2d_size_x, toolpathbox2d_size_y],
                         [0, toolpathbox2d_size_y]])
    M = cv2.getPerspectiveTransform(pts_from, pts_to)
    warped = cv2.warpPerspective(arr_sandbox2d_with_toolpath,
                                 M,
                                 (int(toolpathbox2d_size_x),
                                  int(toolpathbox2d_size_y)))
    if show_cropped_img:
        cv2.imshow('crop_toolpathbox2d_oriented', warped)
        cv2.waitKey(0)

    return warped


def create_scan_identifier(iteration_num):
    id_num = str(iteration_num).zfill(5)
    return str(datetime.date.today()) + '_' + str(id_num)


def make_iteration_dirs(dir, id):
    # create folder for toolpath collection
    new_folder_path = dir + '/toolpath_collection_{}'.format(id)
    try:
        os.makedirs(new_folder_path)
    except FileExistsError:
        print("Directory ", new_folder_path, " already exists")

    # create folder for toolpath json
    new_folder_path_toolpath_frames = (new_folder_path
                                       + '/'
                                       + 'toolpath_frames')
    try:
        os.makedirs(new_folder_path_toolpath_frames)
    except FileExistsError:
        print("Directory ", new_folder_path_toolpath_frames, " already exists")

    # create folder for cropped toolpath img
    new_folder_path_cropped_toolpath = (new_folder_path
                                        + '/'
                                        + 'cropped_toolpath')

    # create folder for sandbox img
    new_folder_path_sdandbox = new_folder_path + '/' + 'sandbox'
    try:
        os.makedirs(new_folder_path_sdandbox)
    except FileExistsError:
        print("Directory ", new_folder_path_sdandbox, " already exists")

    # create folder for cropped toolpath img
    new_folder_path_cropped_toolpath = (new_folder_path
                                        + '/'
                                        + 'cropped_toolpath')
    try:
        os.makedirs(new_folder_path_cropped_toolpath)
    except FileExistsError:
        print("Directory ",
              new_folder_path_cropped_toolpath,
              " already exists")

    return (new_folder_path,
            new_folder_path_toolpath_frames,
            new_folder_path_sdandbox,
            new_folder_path_cropped_toolpath)


def export_an_img(filename, img_to_save):
    cv2.imwrite(filename, img_to_save)


def create_json_file(
        folder_path,
        filename):
    data = {}
    data['toolpath_ctrl_frames'] = {}

    filepath = folder_path + '/' + '{}.json'.format(filename)
    with open(filepath, 'w') as o:
        json.dump(data, o, indent=4)


def export_ctrl_frames_json(
        ctrl_frames_in_sandbox2d,
        folder_path,
        filename,
        excavation_iter=1):

    # load json
    filepath = folder_path + '/' + filename + '.json'
    with open(filepath, 'r') as f:
        data = json.load(f)

    # add a key
    iter_num = str(excavation_iter).zfill(5)
    iter_key = 'iteration_{}'.format(iter_num)
    data['toolpath_ctrl_frames'][iter_key] = {}

    # store frames
    for j, f in enumerate(ctrl_frames_in_sandbox2d):
        frame_num = str(j).zfill(5)
        frame_key = 'frame_{}'.format(frame_num)
        data['toolpath_ctrl_frames'][iter_key][frame_key] = f.to_jsonstring()

    # export and overwrite json
    with open(filepath, 'w') as o:
        json.dump(data, o, indent=4)


def export_sandbox_as_img(
        sandbox,
        folder_path,
        excavation_iter):
    iter_num = str(excavation_iter).zfill(5)
    filename = folder_path + '/' + 'sandbox_' + iter_num + '.png'
    export_an_img(filename, sandbox)


def export_cropped_toolpath_as_img(
        cropped_toolpath,
        folder_path,
        excavation_iter):
    iter_num = str(excavation_iter).zfill(5)
    filename = folder_path + '/' + 'cropped_toolpath_' + iter_num + '.png'
    export_an_img(filename, cropped_toolpath)


def generate_toolpath_frames_in_sandbox2d(
        new_folder_path,
        filename_id,
        excavation_iter,
        num_ctrl_pts,
        level,
        curve_type,
        segments_num,
        viewer,
        save_json):

    # for robot frame
    ctrl_pts_tuple_list = generate_ctrl_pts_tuple(num_ctrl_pts, level)
    ctrl_frames = tuple_to_compas_frame(ctrl_pts_tuple_list,
                                        curve_type,
                                        segments_num)
    ctrl_frames_rotation = rotate_ctrl_frames(ctrl_frames)
    ctrl_frames_in_sandbox2d = move_ctrl_frames_to_sandbox2d(
                                   ctrl_frames_rotation)

    if viewer:
        run_viewer(ctrl_frames_in_sandbox2d)

    if save_json:
        export_ctrl_frames_json(
            ctrl_frames_in_sandbox2d,
            new_folder_path,
            filename_id,
            excavation_iter)

    return ctrl_frames_in_sandbox2d


def draw_polyline_in_sandbox2d(
        frames,
        new_folder_path_sdandbox,
        excavation_iter,
        thickness,
        show_toolpath_in_sandbox,
        save_img):

    sandbox_dimensions = facts['sandbox_dimensions']
    blank = 255 * np.ones(shape=[sandbox_dimensions['w'],
                                 sandbox_dimensions['l'],
                                 3],
                          dtype=np.uint8)

    for a, b in cu.pairwise(range(len(frames))):
        pt_s = frames[a].point
        pt_e = frames[b].point
        cv2.line(blank,
                 (int(pt_s[0]), int(pt_s[1])),
                 (int(pt_e[0]), int(pt_e[1])),
                 color=(0, 0, pt_s[2]),  # red channel for toolpath height
                 thickness=thickness)

    if show_toolpath_in_sandbox:
        cv2.imshow('draw_a_line_in_sandbox2d', blank)
        cv2.waitKey(0)

    if save_img:
        export_sandbox_as_img(
            blank,
            new_folder_path_sdandbox,
            excavation_iter)

    return blank


def crop_toolpath_in_sandbox2d(
        arr_sandbox2d_with_toolpath,
        ctrl_frames_in_sandbox2d,
        new_folder_path_cropped_toolpath,
        excavation_iter,
        show_cropped_img,
        show_bbox,
        show_text,
        save_img):

    # calc contour
    cnt = calc_contour(
        arr_sandbox2d_with_toolpath)
    # calc minAreaRect
    oriented_bbox, corner_pts_orient_float = calc_two_bounding_box(
        arr_sandbox2d_with_toolpath,
        cnt,
        show_text,
        show_bbox)
    oriented_bbox_corner_pts_in_figsize = calc_oriented_corner_pts_in_figsize(
        oriented_bbox)
    oriented_bbox_corner_pts_in_figsize_shifted = align_bbox_with_toolpath_dir(
        ctrl_frames_in_sandbox2d,
        corner_pts_orient_float,
        oriented_bbox_corner_pts_in_figsize)
    cropped_toolpath = crop_toolpathbox2d_oriented(
        arr_sandbox2d_with_toolpath,
        oriented_bbox_corner_pts_in_figsize_shifted,
        show_cropped_img)
    if save_img:
        export_cropped_toolpath_as_img(
            cropped_toolpath,
            new_folder_path_cropped_toolpath,
            excavation_iter)

    return cropped_toolpath


def get_and_store_toolpath(level, curvetype):
    # input before data collection
    data_collection_iter = 0
    excavation_iter = 1

    folder_path = "G:/Shared drives/2021_MAS/T3/Common Ground Control/01_data/00_data_collection/00_test/02_toolpath"
    filename_id = create_scan_identifier(data_collection_iter)

    (new_folder_path,
     new_folder_path_toolpath_frames,
     new_folder_path_sdandbox,
     new_folder_path_cropped_toolpath) = make_iteration_dirs(
        folder_path,
        filename_id)
    create_json_file(new_folder_path_toolpath_frames, filename_id)

    ctrl_frames_in_sandbox2d = generate_toolpath_frames_in_sandbox2d(
        new_folder_path_toolpath_frames,
        filename_id,
        excavation_iter=excavation_iter,
        num_ctrl_pts=5,
        level=level,
        curve_type=curvetype,  # polyline or bezier
        segments_num=50,
        viewer=False,
        save_json=True)

    arr_sandbox2d_with_toolpath = draw_polyline_in_sandbox2d(
        ctrl_frames_in_sandbox2d,
        new_folder_path_sdandbox,
        excavation_iter=excavation_iter,
        thickness=2,
        show_toolpath_in_sandbox=False,
        save_img=True)

    cropped_toolpath = crop_toolpath_in_sandbox2d(
        arr_sandbox2d_with_toolpath,
        ctrl_frames_in_sandbox2d,
        new_folder_path_cropped_toolpath,
        excavation_iter=excavation_iter,
        show_cropped_img=False,
        show_bbox=False,
        show_text=False,
        save_img=True)

    return ctrl_frames_in_sandbox2d


if __name__ == "__main__":

    level = '1.2'
    curvetype = 'polyline'
    ctrl_frames_in_sandbox2d = get_and_store_toolpath(level, curvetype)

# test in a loop
'''
    data_collection_iter = 0
    excavation_iter = 1

    for i in range(excavation_iter):

        # create placeholders
        if i == 0:
            folder_path = "G:/Shared drives/2021_MAS/T3/Common Ground Control/01_data/00_data_collection/00_test/02_toolpath"
            filename_id = create_scan_identifier(data_collection_iter)

            (new_folder_path,
             new_folder_path_toolpath_frames,
             new_folder_path_sdandbox,
             new_folder_path_cropped_toolpath) = make_iteration_dirs(
                folder_path,
                filename_id)
            create_json_file(new_folder_path_toolpath_frames, filename_id)

        ctrl_frames_in_sandbox2d = generate_toolpath_frames_in_sandbox2d(
            new_folder_path_toolpath_frames,
            filename_id,
            excavation_iter=i,
            num_ctrl_pts=5,
            level='1.0',
            curve_type='bezier',  # polyline or bezier
            segments_num=100,
            viewer=False,
            save_json=True)

        arr_sandbox2d_with_toolpath = draw_polyline_in_sandbox2d(
            ctrl_frames_in_sandbox2d,
            new_folder_path_sdandbox,
            excavation_iter=i,
            thickness=2,
            show_toolpath_in_sandbox=False,
            save_img=True)

        cropped_toolpath = crop_toolpath_in_sandbox2d(
            arr_sandbox2d_with_toolpath,
            ctrl_frames_in_sandbox2d,
            new_folder_path_cropped_toolpath,
            excavation_iter=i,
            show_cropped_img=False,
            show_bbox=False,
            show_text=False,
            save_img=True)
'''
