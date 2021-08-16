import os
import sys
import cv2 as cv
import json
import datetime
import math as m
import numpy as np
import random as r
import compas.geometry as cg
import compas.utilities as cu
import tensorflow as tf
import matplotlib.pyplot as plt
from tkinter.filedialog import askdirectory


class Artist(object):
    def __init__(self, num_ctrl_pts, fframe_bounds, height_fframe, model, target_img):
        self.num_ctrl_pts = num_ctrl_pts
        self.fframe_bounds = fframe_bounds
        self.height_fframe = height_fframe

        self.facts = self.call_fact()
        self.toolpath_feature = self.draw_polyline_in_sandbox2d()
        self.toolpath_fframe = self.crop_feature(self.toolpath_feature)
        self.input_img = self.custom_img_addition(self.height_fframe, self.toolpath_fframe)

        self.brain = self.BRAIN(model, self.input_img, target_img)
        self.dna = self.DNA(self.ctrl_pts)

    def random_ctrl_pts(self):
        arr = np.zeros((self.num_ctrl_pts, 3))
        for i in range(arr.shape[0]):
            arr[i][0] = r.randint(int(self.fframe_bounds[0][0]), int(self.fframe_bounds[0][1]))
            arr[i][1] = r.randint(int(self.fframe_bounds[1][0]), int(self.fframe_bounds[1][1]))
            arr[i][2] = r.randint(50, 100)
        self.ctrl_pts = arr

    def tuple_to_compas_frame(self, curve_type='bezier', segments_num=50):
        ctrl_frames = []
        ctrl_pts = self.ctrl_pts
        ctrl_pts = [cg.Point(tl[0], tl[1], tl[2]) for tl in ctrl_pts]

        if curve_type == 'polyline':
            polyline = cg.Polyline(ctrl_pts)
            pts_on_curve = polyline.divide_polyline(segments_num)

        elif curve_type == 'bezier':
            curve = cg.Bezier(ctrl_pts)
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
            ctrl_frames.append(cg.Frame(pta, yaxis, -xaxis))
        return ctrl_frames

    def remapValue(self, v, ori_Min, ori_Max, targetMin, targetMax):
        rv = ((v-ori_Min)/(ori_Max-ori_Min))*(targetMax-targetMin)+targetMin
        return rv

    def draw_polyline_in_sandbox2d(self):
        feature_xsize = int(abs(self.facts['feature_bounds']['max_bound'][1]
                                - self.facts['feature_bounds']['min_bound'][1]))
        feature_ysize = int(abs(self.facts['feature_bounds']['max_bound'][0]
                                - self.facts['feature_bounds']['min_bound'][0]))

        img = np.zeros(shape=[m.floor(feature_ysize),
                              m.floor(feature_xsize),
                              3], dtype=np.uint8)

        ctrl_frames = self.tuple_to_compas_frame()
        for a, b in cu.pairwise(range(len(ctrl_frames))):
            pt_s = ctrl_frames[a].point
            pt_e = ctrl_frames[b].point
            z = self.remapValue(pt_s[2], 0, 150, 0, 255)
            cv.line(img,
                    (int(pt_s[0]), int(pt_s[1])),
                    (int(pt_e[0]), int(pt_e[1])),
                    color=(0, 0, z),  # red channel for toolpath height
                    thickness=2,
                    lineType=cv.FILLED)
        return img

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

        feature_center = self.get_feature_center()

        xmin = feature_center[0] - (xsize/2)
        xmax = feature_center[0] + (xsize/2)
        ymin = feature_center[1] - (ysize/2)
        ymax = feature_center[1] + (ysize/2)

        x_range = [xmin, xmax]
        y_range = [ymin, ymax]
        return [x_range, y_range]

    def get_fframe_corner(self):
        fframe_bounds = self.get_fframe_bounds()

        topleft = [fframe_bounds[0][0], fframe_bounds[1][0]]
        topright = [fframe_bounds[0][1], fframe_bounds[1][0]]
        bottomleft = [fframe_bounds[0][0], fframe_bounds[1][1]]
        bottomright = [fframe_bounds[0][1], fframe_bounds[1][1]]
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
        crop_idx = self.get_corp_idx()
        pts_from = np.float32(crop_idx)
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
                                         borderMode=cv.BORDER_CONSTANT)
        return img_cropped

    def custom_img_addition(self, height_img, toolpath_img):
        arr = np.zeros([256, 256, 3], dtype=np.uint8)
        for i in range(256):
            for j in range(256):
                if toolpath_img[i][j][2] > 0:
                    arr[i][j] = toolpath_img[i][j]
                else:
                    arr[i][j] = height_img[i][j]
        return arr

    class BRAIN():
        def __init__(self, model, input_img, target_img):
            self.model = model
            self.input_tensor, self.target_tensor = self.normalize(input_img, target_img)
            self.generated_img = self.generate_img(model, self.input_tensor, self.target_tensor)

        def decode(self, img_to_decode1, img_to_decode2):
            decoded_img1 = tf.convert_to_tensor(img_to_decode1)
            decoded_img1 = tf.cast(decoded_img1, tf.float32)
            decoded_img2 = tf.convert_to_tensor(img_to_decode2)
            decoded_img2 = tf.cast(decoded_img2, tf.float32)
            return decoded_img1, decoded_img2

        def normalize(self, img_to_normalize1, img_to_normalize2):
            img_to_normalize1, img_to_normalize2 = self.decode(img_to_normalize1, img_to_normalize2)
            img_to_normalize1 = (img_to_normalize1 / 127.5) - 1
            img_to_normalize2 = (img_to_normalize2 / 127.5) - 1
            return img_to_normalize1, img_to_normalize2

        def generate_img(self, model, test_input, target_input):
            test_input = np.reshape(test_input, [1, 256, 256, 3])
            prediction = model(test_input, training=True)
            return prediction[0].numpy()

    class DNA():  # GENOTYPE
        def __init__(self, ctrl_pts, generated_img):
            self.genotype = ctrl_pts
            self.phenotype = generated_img

        def cal_zdiff(self, arr1, arr2):
            arr_diff = np.subtract(arr1, arr2)
            arr_abs_diff = np.absolute(arr_diff)
            return arr_abs_diff

        def get_minmax(self, arr):
            min = np.amin(arr)
            max = np.amax(arr)
            return min, max

        def fit(self, target_arr):
            # evaluation
            zdiff = self.cal_zdiff(self.genes, target_arr)
            zdiff_min, zdiff_max = self.get_minmax(zdiff)
            mean_zdiff = np.mean(zdiff)
            raw_fitness = m.exp(-(mean_zdiff/10))
            if raw_fitness < 0.01:
                raw_fitness += 0.01
            self.fitness = raw_fitness

        def crossover(self, child, parent):
            coin_x = int(r.randint(0, 2))
            coin_y = int(r.randint(0, 2))

            if coin_x and coin_y:
                # half
                child.dna.genes = (self.genes + parent.dna.genes) / 2
            elif coin_x and not coin_y:
                # true false
                child.dna.genes = self.genes * (2/3) + parent.dna.genes * (1/3)
            elif not coin_x and coin_y:
                # false true
                child.dna.genes = self.genes * (1/3) + parent.dna.genes * (2/3)
            elif not coin_x and not coin_y:
                # false false
                child.dna.genes = (self.genes + parent.dna.genes) / 2
            return child

        def mutation(self, metation_rate):
            for i in range(self.genes.shape[0]):
                for j in range(self.genes.shape[1]):
                    if r.random() < metation_rate:
                        self.genes[i][j] = r.randint(int(np.amin(self.genes)), int(np.amax(self.genes)))

        def save_fig(self, fname):
            cv.imwrite(fname, self.genes)

if __name__ == '__main__':
    pass
