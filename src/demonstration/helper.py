import os
import sys
import json
import cv2 as cv
import numpy as np

sys.path.insert(0, 'C:/Users/trtku/OneDrive/Data/03_MAS/17_common_ground_control/01_git/common-ground-control/src/data_collection')
import UR as ur
from data import TrainingData
from scanning import ScanData, HeightMap, PointCloud


class Helper(object):
    def __init__(self):
        self.facts = self.call_fact()
        self.feature_center = self.get_feature_center()
        self.fframe_bounds = self.get_fframe_bounds()
        self.crop_idx = self.get_corp_idx()

    def call_fact(self):
        dir = os.getcwd()
        fname = "data_collection/data/facts.json"
        path = os.path.join(dir, fname)
        with open(path, 'r') as f:
            facts = json.load(f)
        return facts

    def get_feature_center(self):
        (f_bounds_xmin,
        f_bounds_ymin,
        f_bounds_zmin) = self.facts['feature_bounds']['min_bound']
        (f_bounds_xmax,
        f_bounds_ymax,
        f_bounds_zmax) = self.facts['feature_bounds']['max_bound']

        x = (f_bounds_xmax - f_bounds_xmin)/2
        y = (f_bounds_ymax - f_bounds_ymin)/2
        z = (f_bounds_zmax - f_bounds_zmin)/2
        return [y, x, z]

    def get_fframe_bounds(self):
        xsize = self.facts['fig_size']['x']
        ysize = self.facts['fig_size']['y']

        xmin = self.feature_center[0] - (xsize/2)
        xmax = self.feature_center[0] + (xsize/2)
        ymin = self.feature_center[1] - (ysize/2)
        ymax = self.feature_center[1] + (ysize/2)

        x_range = [xmin, xmax]
        y_range = [ymin, ymax]
        return [x_range, y_range]

    def get_fframe_corner(self):
        topleft = [self.fframe_bounds[0][0], self.fframe_bounds[1][0]]
        topright = [self.fframe_bounds[0][1], self.fframe_bounds[1][0]]
        bottomleft = [self.fframe_bounds[0][0], self.fframe_bounds[1][1]]
        bottomright = [self.fframe_bounds[0][1], self.fframe_bounds[1][1]]
        return [topleft, topright, bottomright, bottomleft]

    def get_corp_idx(self):
        idx = []
        corners = self.get_fframe_corner()
        for c in corners:
            coord = []
            for i in c:
                coord.append(int(i))
            idx.append(coord)
        return idx

    def crop_feature(self, img):
        pts_from = np.float32(self.crop_idx)
        pts_to = np.float32([[0, 0],
                            [self.facts['fig_size']['x'], 0],
                            [self.facts['fig_size']['x'], self.facts['fig_size']['y']],
                            [0, self.facts['fig_size']['y']]])
        M = cv.getPerspectiveTransform(pts_from, pts_to)
        img_cropped = cv.warpPerspective(img,
                                         M,
                                         (int(self.facts['fig_size']['x']),
                                          int(self.facts['fig_size']['x'])),
                                         flags=cv.WARP_FILL_OUTLIERS,
                                         borderMode=cv.BORDER_TRANSPARENT)
        return img_cropped


if __name__ == '__main__':
    save_dir = "G:/Shared drives/2021_MAS/T3/Common Ground Control/01_data/02_demo/00_test"
