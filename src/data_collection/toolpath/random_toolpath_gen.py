# LIBRARIES

# from compas_view2.app import App
from collections import deque
import compas.geometry as cg
import compas.utilities as cu
import random as r
import math as m
import numpy as np
import cv2
import json


# set global facts
with open('data/facts.json') as f:
    facts = json.load(f)


# FUNCTIONS
'''
list
'''


class Toolpath():
    def __init__(self,
                 level,
                 curve_type,
                 num_ctrl_pts,
                 segments_num,
                 thickness,
                 parent_folder,
                 id,
                 d,
                 hm_feature=None,
                 adaptive_depth=0):
        self.level = level
        self.curve_type = curve_type
        self.num_ctrl_pts = num_ctrl_pts
        self.segments_num = segments_num
        self.thickness = thickness
        self.id = id
        self.parent_folder = parent_folder
        self.d = d
        self.hm_feature = hm_feature
        self.adaptive_depth = adaptive_depth

        self.create_json_file()
        self.move_ctrl_frames_to_feature()
        self.shift_list()

    def generate_ctrl_pts_tuple(self):
        self.y_min = 30
        self.y_max = 180
        self.z_min = 50
        self.z_max = 100
        self.l_min = 30
        self.l_max = 180
        ctrl_pts_list = []

        if self.level == '1-0':
            step = self.l_max / (self.num_ctrl_pts - 1)
            for i in range(self.num_ctrl_pts):
                x = i * step
                y = 0
                z = 75.0
                ctrl_pts_list.append((x, y, z))

        elif self.level == '1-1':
            length = r.randrange(self.l_min, self.l_max)
            step = length / (self.num_ctrl_pts - 1)
            z = r.randrange(self.z_min, self.z_max)
            for i in range(self.num_ctrl_pts):
                x = i * step
                y = 0
                ctrl_pts_list.append((x, y, z))

        elif self.level == '1-2':
            length = r.randrange(self.l_min, self.l_max)
            step = length / (self.num_ctrl_pts - 1)
            for i in range(self.num_ctrl_pts):
                x = i * step
                y = 0
                z = r.randrange(self.z_min, self.z_max)
                ctrl_pts_list.append((x, y, z))

        elif self.level == '1-3':
            length = r.randrange(self.l_min, self.l_max)
            step = length / (self.num_ctrl_pts - 1)
            for i in range(self.num_ctrl_pts):
                x = i * step
                y = 0
                z = r.randrange(self.z_min, self.z_max)
                ctrl_pts_list.append((x, y, z))

        elif self.level == '2-0':
            length = r.randrange(self.l_min, self.l_max)
            step = length / (self.num_ctrl_pts - 1)
            for i in range(self.num_ctrl_pts):
                x = i * step
                y = r.randrange(self.y_min, self.y_max)
                z = 75.0
                ctrl_pts_list.append((x, y, z))

        elif self.level == '2-1':
            length = r.randrange(self.l_min, self.l_max)
            step = length / (self.num_ctrl_pts - 1)
            z = r.randrange(self.z_min, self.z_max)
            for i in range(self.num_ctrl_pts):
                x = i * step
                y = r.randrange(self.y_min, self.y_max)
                ctrl_pts_list.append((x, y, z))

        elif self.level == '2-2':
            length = r.randrange(self.l_min, self.l_max)
            step = length / (self.num_ctrl_pts - 1)
            for i in range(self.num_ctrl_pts):
                x = i * step
                y = r.randrange(self.y_min, self.y_max)
                z = r.randrange(self.z_min, self.z_max)
                ctrl_pts_list.append((x, y, z))

        elif self.level == 'center':
            length = 180
            step = length / (self.num_ctrl_pts-1)
            for i in range(self.num_ctrl_pts):
                x = i * step
                y = 0.0
                z = 75.0
                ctrl_pts_list.append((x, y, z))

        elif self.level == 'adaptive':
            length = 120
            step = length / (self.num_ctrl_pts - 1)
            for i in range(self.num_ctrl_pts):
                x = i * step
                y = 0.0
                z = 0.0
                ctrl_pts_list.append((x, y, z))

        return ctrl_pts_list

    def tuple_to_compas_frame(self):
        ctrl_frames = []
        ctrl_pts_tuple = self.generate_ctrl_pts_tuple()
        ctrl_pts = [cg.Point(tl[0], tl[1], tl[2]) for tl in ctrl_pts_tuple]

        if self.curve_type == 'polyline':
            polyline = cg.Polyline(ctrl_pts)
            pts_on_curve = polyline.divide_polyline(self.segments_num)

        elif self.curve_type == 'bezier':
            curve = cg.Bezier(ctrl_pts)
            pts_on_curve = []
            step = 1 / (self.segments_num-1)
            for i in range(self.segments_num):
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
            ctrl_frames.append(cg.Frame(pta, yaxis, -xaxis))
        return ctrl_frames

    def rotate_ctrl_frames(self):
        # get control frames
        ctrl_frames = self.tuple_to_compas_frame()
        # set angle
        degree = r.randint(-180, 0)

        if self.level == 'center':
            degree = -90
        # get rotation center
        R = cg.Rotation.from_axis_and_angle(cg.Vector.Zaxis(),
                                            m.radians(degree))
        for f in ctrl_frames:
            f.transform(R)
        toggle = r.randint(0, 1)
        if toggle:
            ctrl_frames.reverse()
        return ctrl_frames

    def move_ctrl_frames_to_sandbox2d(self):
        # get rotated ctrl_frames
        ctrl_frames = self.rotate_ctrl_frames()
        # generate target frame to move to
        frame_to_x = r.randint(int(self.d.offset_x_min), int(self.d.offset_x_max))
        frame_to_y = r.randint(int(self.d.offset_y_min), int(self.d.offset_y_max))
        if self.level == 'center':
            frame_to_x = self.d.sandbox_xsize / 2
            frame_to_y = self.d.sandbox_ysize / 2
        frame_to_center = cg.Point(frame_to_x,
                                   frame_to_y,
                                   0)
        frame_to = cg.Frame(frame_to_center,
                            cg.Vector.Xaxis(),
                            cg.Vector.Yaxis())
        # generate frame at middle of the toolpath
        toolpath_line = cg.Line(ctrl_frames[0].point,
                                ctrl_frames[-1].point)
        toolpath_midpt = toolpath_line.midpoint
        toolpath_midpt.z = 0
        frame_from = cg.Frame(toolpath_midpt,
                              cg.Vector.Xaxis(),
                              cg.Vector.Yaxis())
        # create transformation
        T = cg.Transformation.from_frame_to_frame(frame_from, frame_to)
        for ctrl_frame in ctrl_frames:
            ctrl_frame.transform(T)
        return ctrl_frames

    def move_ctrl_frames_to_feature(self):
        ctrl_frames = self.move_ctrl_frames_to_sandbox2d()
        framefrom = cg.Frame(cg.Point(0, 0, 0),
                             cg.Vector.Xaxis(),
                             cg.Vector.Yaxis())
        frameto = cg.Frame(cg.Point(self.d.feature_origin_x,
                                    self.d.feature_origin_y,
                                    self.d.feature_origin_z),
                           cg.Vector.Xaxis(),
                           cg.Vector.Yaxis())
        T = cg.Transformation.from_frame_to_frame(frameto, framefrom)
        self.ctrlframes_feature = []
        self.ctrlframes = []
        for ctrl_frame in ctrl_frames:
            f = ctrl_frame.transformed(T)
            f_robot = ctrl_frame.copy()
            if self.level == 'adaptive':
                h_sand = self.get_scanned_height(int(f.point[0]), int(f.point[1]))
                f.point[2] = h_sand
                f_robot.point.z = h_sand
            self.ctrlframes_feature.append(f)
            self.ctrlframes.append(f_robot)

    def get_scanned_height(self, x, y):
        height_pix = self.hm_feature[y, x, 0]
        height_mm = self.remapValue(height_pix, 0, 255, 0, 150)
        height = (130 - height_mm) + self.adaptive_depth

        # safety net
        if height > 100:
            height = 100
        return height

    # image processing from here

    def remapValue(self, v, ori_Min, ori_Max, targetMin, targetMax):
        rv = ((v-ori_Min)/(ori_Max-ori_Min))*(targetMax-targetMin)+targetMin
        return rv

    def draw_polyline_in_sandbox2d(self):
        ctrl_frames = self.ctrlframes_feature
        img = 255 * np.ones(shape=[m.floor(self.d.feature_ysize),
                                   m.floor(self.d.feature_xsize),
                                   3], dtype=np.uint8)

        for a, b in cu.pairwise(range(len(ctrl_frames))):
            pt_s = ctrl_frames[a].point
            pt_e = ctrl_frames[b].point
            z = self.remapValue(pt_s[2], self.z_min, self.z_max, 0, 255)
            cv2.line(img,
                     (int(pt_s[0]), int(pt_s[1])),
                     (int(pt_e[0]), int(pt_e[1])),
                     color=(0, 0, z),  # red channel for toolpath height
                     thickness=self.thickness)
        self.img = img
        return img, ctrl_frames

    def calc_contour(self):
        img = self.draw_polyline_in_sandbox2d()[0]
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, thresh = cv2.threshold(src=img_gray,
                                    thresh=127.5,
                                    maxval=255,
                                    type=cv2.THRESH_BINARY)
        contours, hierarchy = cv2.findContours(thresh, 1, 2)
        cnt = contours[0]
        return cnt

    def calc_two_bounding_box(self):
        cnt = self.calc_contour()
        # draw orthogonal bbox
        corner_pts_ortho = cv2.boundingRect(cnt)
        x, y, w, h = corner_pts_ortho
        rect = cv2.minAreaRect(cnt)
        corner_pts_orient_float = cv2.boxPoints(rect)
        return rect, corner_pts_orient_float

    def calc_crop_idx(self):
        centroid, ratio, angle = self.calc_two_bounding_box()[0]
        fig_size_diagonal = (self.d.frame_size_x * m.sqrt(2)) / 2

        if angle < 45:
            theta = m.radians(45-angle)
            pt0 = [int(centroid[0] - fig_size_diagonal*m.cos(theta)),
                   int(centroid[1] + fig_size_diagonal*m.sin(theta))]
            pt1 = [int(centroid[0] - fig_size_diagonal*m.sin(theta)),
                   int(centroid[1] - fig_size_diagonal*m.cos(theta))]
            pt2 = [int(centroid[0] + fig_size_diagonal*m.cos(theta)),
                   int(centroid[1] - fig_size_diagonal*m.sin(theta))]
            pt3 = [int(centroid[0] + fig_size_diagonal*m.sin(theta)),
                   int(centroid[1] + fig_size_diagonal*m.cos(theta))]

        else:
            theta = m.radians(angle-45)
            pt0 = [int(centroid[0] - fig_size_diagonal*m.cos(theta)),
                   int(centroid[1] - fig_size_diagonal*m.sin(theta))]
            pt1 = [int(centroid[0] + fig_size_diagonal*m.sin(theta)),
                   int(centroid[1] - fig_size_diagonal*m.cos(theta))]
            pt2 = [int(centroid[0] + fig_size_diagonal*m.cos(theta)),
                   int(centroid[1] + fig_size_diagonal*m.sin(theta))]
            pt3 = [int(centroid[0] - fig_size_diagonal*m.sin(theta)),
                   int(centroid[1] + fig_size_diagonal*m.cos(theta))]
        crop_idx = [pt0, pt1, pt2, pt3]
        return crop_idx

    def calc_toolpath_dir(self):
        ctrl_frames = self.draw_polyline_in_sandbox2d()[1]
        # calc direction of toolpath
        toolpath_start = ctrl_frames[0].point
        toolpath_end = ctrl_frames[-1].point
        toolpath_dir = cg.Vector.from_start_end(toolpath_start,
                                                toolpath_end)
        return toolpath_dir

    def calc_toolpath_box_dir(self):
        corner_pts_orient_float = self.calc_two_bounding_box()[1]
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
        edge1_dir = cg.Vector.from_start_end(oriented_rect_pt0,
                                             oriented_rect_pt2)
        edge2_dir = cg.Vector.from_start_end(oriented_rect_pt1,
                                             oriented_rect_pt3)
        return edge1_dir, edge2_dir

    def calc_dot_toolpath_and_bbox(self):
        toolpath_dir = self.calc_toolpath_dir()
        edge1_dir, edge2_dir = self.calc_toolpath_box_dir()
        # check orientation of toolpath and crop_area
        toolpath_dir.unitize()
        # self.edge1_dir.unitize()
        # self.edge2_dir.unitize()
        dot1 = cg.dot_vectors_xy(toolpath_dir, edge1_dir)
        dot2 = cg.dot_vectors_xy(toolpath_dir, edge2_dir)
        return dot1, dot2

    def calc_shift_number_bbox_corner(self):
        dot1, dot2 = self.calc_dot_toolpath_and_bbox()
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

    def shift_list(self):
        list_to_shift = self.calc_crop_idx()
        shift_num = self.calc_shift_number_bbox_corner()
        collection_to_shift = deque(list_to_shift)
        collection_to_shift.rotate(shift_num)
        self.crop_idx = list(collection_to_shift)

    def crop_toolpathbox2d_oriented(self, img):
        pts_from = np.float32(self.crop_idx)
        pts_to = np.float32([[0, 0],
                            [self.d.frame_size_x, 0],
                            [self.d.frame_size_x, self.d.frame_size_y],
                            [0, self.d.frame_size_y]])
        M = cv2.getPerspectiveTransform(pts_from, pts_to)
        img_cropped = cv2.warpPerspective(img,
                                          M,
                                          (int(self.d.frame_size_x),
                                           int(self.d.frame_size_y)))
        return img_cropped

    def show_img(self, img_to_show):
        cv2.imshow('show_img', img_to_show)
        cv2.waitKey(0)

    def export_an_img(self, img_to_save):
        filename = self.parent_folder + '/' + self.id + '.png'
        cv2.imwrite(filename, img_to_save)

    def create_json_file(self):
        data = {}
        data['frame_corner_pts'] = {}
        data['ctrl_frames'] = {}
        data['lvl'] = {}
        data['num_ctrl_pts'] = {}
        data['curve_type'] = {}
        data['thickness'] = {}
        data['adaptive_depth'] = {}
        filepath = self.parent_folder + '/' + '{}_toolpath.json'.format(self.id)
        with open(filepath, 'w') as o:
            json.dump(data, o, indent=4)

    def export_json(self):
        # load json
        filepath = self.parent_folder + '/' + self.id + '_toolpath.json'
        with open(filepath, 'r') as f:
            data = json.load(f)
        # store crop_idx
        for i, ci in enumerate(self.crop_idx):
            data['frame_corner_pts'][i] = ci
        # store frames
        ctrl_frames = self.ctrlframes
        for j, f in enumerate(ctrl_frames):
            frame_num = str(j).zfill(3)
            frame_key = 'f_{}'.format(frame_num)
            data['ctrl_frames'][frame_key] = f.to_jsonstring()
        # store curve info
        data['lvl'] = self.level
        data['num_ctrl_pts'] = self.num_ctrl_pts
        data['thickness'] = self.thickness
        data['curve_type'] = self.curve_type
        data['adaptive_depth'] = self.adaptive_depth
        # export and overwrite json
        with open(filepath, 'w') as o:
            json.dump(data, o, indent=4)


class Dimension():
    def __init__(self):
        self.get_feature_frame_size()
        self.get_feature_bounds()
        self.get_sandbox_size()
        self.get_feature_origin()
        self.get_feature_frame_size()
        self.calc_offset_area_sandbox2d()

    def get_feature_bounds(self):
        (self.f_bounds_xmin,
            self.f_bounds_ymin,
            self.f_bounds_zmin) = facts['feature_bounds']['min_bound']
        (self.f_bounds_xmax,
            self.f_bounds_ymax,
            self.f_bounds_zmax) = facts['feature_bounds']['max_bound']
        self.feature_xsize = int(abs(self.f_bounds_ymax
                                        - self.f_bounds_ymin))
        self.feature_ysize = int(abs(self.f_bounds_xmax
                                        - self.f_bounds_xmin))

    def get_sandbox_size(self):
        # get corner pts
        pts = facts['robot_corner_pts']
        self.pt0 = cg.Point(pts['pt0'][0], pts['pt0'][1], pts['pt0'][2])
        ptx = cg.Point(pts['ptx'][0], pts['ptx'][1], pts['ptx'][2])
        pty = cg.Point(pts['pty'][0], pts['pty'][1], pts['pty'][2])
        # get size
        self.sandbox_xsize = cg.distance_point_point_xy(self.pt0, ptx)  # 1135
        self.sandbox_ysize = cg.distance_point_point_xy(self.pt0, pty)  # 737

    def get_feature_origin(self):
        self.feature_origin_x = (self.pt0.y - self.f_bounds_ymax)  # 587 - 570 = 17
        self.feature_origin_y = (self.pt0.x - self.f_bounds_xmax)   # (-338) - (-360) = 22
        self.feature_origin_z = 0

    def get_feature_frame_size(self):
        self.frame_size_x = facts['fig_size']['x']
        self.frame_size_y = facts['fig_size']['y']
        self.frame_size_z = facts['fig_size']['z']

    def calc_offset_area_sandbox2d(self):
        # calculate half of diagonal length of toolpathbox2d
        x_offset_dist = (self.frame_size_x * m.sqrt(2)) / 2
        y_offset_dist = (self.frame_size_y * m.sqrt(2)) / 2
        # set min/max of working area
        self.offset_x_min = x_offset_dist
        self.offset_x_max = self.feature_xsize - x_offset_dist
        self.offset_y_min = y_offset_dist
        self.offset_y_max = self.feature_ysize - y_offset_dist


if __name__ == "__main__":
    pass
