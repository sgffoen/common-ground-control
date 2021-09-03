from argparser import parse_args
from learning import LearningData
import compas.geometry as cg
import compas.utilities as cu
import numpy as np
import math as m
import datetime
import time
import json
import cv2
import os

__FOLDER__ = "G:/Shared drives/2021_MAS/T3/Common Ground Control/01_data/"


# fact sheet
path = os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..'))
dir = os.path.join(path, 'data_collection', 'data', 'facts.json')
print(dir)
with open(dir) as f:
    facts = json.load(f)


def remapValue(v, ori_Min, ori_Max, targetMin, targetMax):
    rv = ((v-ori_Min)/(ori_Max-ori_Min))*(targetMax-targetMin)+targetMin
    return rv


def get_iteration_dirs():
    iteration_dirs = os.listdir('G:/Shared drives/2021_MAS/T3/Common Ground Control/01_data/00_data_collection/01_production')
    iteration_dirs.remove('meta.json')
    iteration_dirs.sort()
    return iteration_dirs


def aliasing_thickening(env, thickness):
    start_time = time.time()
    print("starting aliasing mode")
    print("environment: {} \n".format(env))

    iteration_dirs = get_iteration_dirs()
    iteration_dir = iteration_dirs[0]
    d = Dimension()

    for i, id in enumerate(iteration_dirs):
    # for i in range(1):
        # initiate lap
        start_lap = time.time()
        # 1. accessing data_collection path/img
        data = LearningData(i, id, env)
        data.create_iter_dirs_mydrive()
        data.get_iter_dirs()

        # load json
        dir = data.path_name_raw
        fname = data.id + '_toolpath.json'
        fpath = os.path.join(dir, fname)
        with open(fpath, 'r') as o:
            facts = json.load(o)

        # get control frames from json
        ctrl_frames = []
        fkeys = facts['ctrl_frames'].keys()
        # for j, c in enumerate(facts['ctrl_frames']):
        # set a white canvas
        img = np.zeros(shape=[d.feature_ysize,
                                d.feature_xsize,
                                3],
                                dtype=np.uint8)
        aliasing_img = np.copy(img)
        thickening_img = np.copy(img)
        for a, b in cu.pairwise(fkeys):
            # move to origin
            feature_origin = cg.Frame(cg.Point(d.feature_origin_x, d.feature_origin_y, 0),
                                        cg.Vector.Xaxis(),
                                        cg.Vector.Yaxis())
            sandbox_origin = cg.Frame(cg.Point(0,0,0),
                                        cg.Vector.Xaxis(),
                                        cg.Vector.Yaxis())
            # T = cg.Transformation.from_frame_to_frame(feature_origin, sandbox_origin)
            # get start and end
            frame = cg.Frame.from_jsonstring(facts['ctrl_frames'][a])
            start = frame.point#.transformed(T)
            frame = cg.Frame.from_jsonstring(facts['ctrl_frames'][b])
            end = frame.point#.transformed(T)
            # remap
            z_flip = remapValue(start.z, 130, 0, 0, 130)
            z_pixel = remapValue(z_flip, 0, 150, 0, 255)
            cv2.line(aliasing_img,
                    (int(start[0]), int(start[1])),
                    (int(end[0]), int(end[1])),
                    color=(0, 0, z_pixel),  # red channel for toolpath height
                    thickness=2,
                    lineType=cv2.FILLED)
            cv2.line(thickening_img,
                    (int(start[0]), int(start[1])),
                    (int(end[0]), int(end[1])),
                    color=(0, 0, z_pixel),  # red channel for toolpath height
                    thickness=50,
                    lineType=cv2.FILLED)

        # crop_toolpathbox2d_oriented(self, img, d):
        crop_idx = []
        for k in range(4):
            crop_idx.append(facts['frame_corner_pts'][str(k)])
        pts_from = np.float32(crop_idx)
        pts_to = np.float32([[0, 0],
                            [256, 0],
                            [256, 256],
                            [0, 256]])
        M = cv2.getPerspectiveTransform(pts_from, pts_to)
        aliasing_img_cropped = cv2.warpPerspective(aliasing_img,
                                                   M,
                                                   (int(256),int(256)),
                                                   flags=cv2.WARP_FILL_OUTLIERS,
                                                   borderMode=cv2.BORDER_TRANSPARENT)
        thickening_img_cropped = cv2.warpPerspective(thickening_img,
                                                     M,
                                                     (int(256),int(256)),
                                                     flags=cv2.WARP_FILL_OUTLIERS,
                                                     borderMode=cv2.BORDER_TRANSPARENT)

        # export images
        filename = data.path_name_processed_mydrive + '/' + data.id + '_toolpath_feature_fix_2.png'
        cv2.imwrite(filename, aliasing_img)
        filename = data.path_name_processed_mydrive + '/' + data.id + '_toolpath_featureframe_fix_2.png'
        cv2.imwrite(filename, aliasing_img_cropped)
        # export images
        filename = data.path_name_processed_mydrive + '/' + data.id + '_toolpath_feature_thickness{}_2.png'.format(thickness)
        cv2.imwrite(filename, thickening_img)
        filename = data.path_name_processed_mydrive + '/' + data.id + '_toolpath_featureframe_thickness{}_2.png'.format(thickness)
        cv2.imwrite(filename, thickening_img_cropped)

        if i % 100 == 0:
            lap = (time.time()-start_lap)/60
            print('\nfixing id: {} / {}\nLAP-TIME: {} min\n'.format(i, len(iteration_dirs), lap))

    print('\nTotal fixing time: ', int((time.time()-start_time)/60), ' min\n\n')


def meta_data(env):
    filepath = os.path.join(__FOLDER__,
                            "00_data_collection",
                            ("00_test" if env == 'test' else "01_production"),
                            "meta.json")
    with open(filepath, 'r') as f:
        data = json.load(f)

    max_scan_id = data['max_scan_id']

    return max_scan_id


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
    aliasing_thickening(env='production', thickness=50)
