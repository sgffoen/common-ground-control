import os
import sys
import json
import datetime
import cv2 as cv
import math as m
import numpy as np
import random as r
import tensorflow as tf
import compas.geometry as cg
import compas.utilities as cu
import matplotlib.pyplot as plt
from tkinter.filedialog import askdirectory


class Artist(object):
    def __init__(self, num_ctrl_pts, height_fframe, target_img, model):
        self.facts = self.call_fact()
        self.num_ctrl_pts = num_ctrl_pts
        self.fframe_bounds = self.get_fframe_bounds()
        self.model = model
        self.target_img = target_img.astype(np.uint8)
        self.height_img = height_fframe

        self.genotype = self.pts_on_curve()
        self.phenotype = None

    def random_ctrl_pts(self):
        arr = np.zeros((self.num_ctrl_pts, 3))
        for i in range(arr.shape[0]):
            arr[i][0] = r.randint(int(self.fframe_bounds[0][0]), int(self.fframe_bounds[0][1]))
            arr[i][1] = r.randint(int(self.fframe_bounds[1][0]), int(self.fframe_bounds[1][1]))
            arr[i][2] = r.randint(50, 100)
        return arr

    def pts_on_curve(self):
        ctrl_pts = self.random_ctrl_pts()
        ctrl_frames = []
        ctrl_pts = [cg.Point(tl[0], tl[1], tl[2]) for tl in ctrl_pts]

        curve = cg.Bezier(ctrl_pts)
        segments_num = 10
        arr = np.zeros((segments_num, 3))
        step = 1 / (segments_num-1)
        for i in range(segments_num):
            t = step * i
            pt_on_curve = curve.point(t)
            arr[i][0] = pt_on_curve.x
            arr[i][1] = pt_on_curve.y
            arr[i][2] = pt_on_curve.z
        return arr

    def tuple_to_compas_frame(self, ctrl_pts, curve_type='bezier', segments_num=50):
        ctrl_frames = []
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
                              3],
                       dtype=np.uint8)

        ctrl_frames = self.tuple_to_compas_frame(self.genotype)
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

    def crop_feature(self):
        img = self.draw_polyline_in_sandbox2d()
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

    def custom_img_addition(self):
        self.toolpath_img = self.crop_feature()
        arr = np.zeros([256, 256, 3], dtype=np.uint8)
        for i in range(256):
            for j in range(256):
                if self.toolpath_img[i][j][2] > 0:
                    arr[i][j] = self.toolpath_img[i][j]
                else:
                    arr[i][j] = self.height_img[i][j]
        self.input_img = arr

    def load_img(self, img_path):
        # Read and decode an image file to a uint8 tensor
        image = tf.io.read_file(img_path)
        image = tf.image.decode_png(image)
        # Convert an image to float32 tensors
        image = tf.cast(image, tf.float32)
        # Normalizing the images to [-1, 1]
        image = (image / 127.5) - 1
        # reshape to add batch at the index of zero
        image = np.reshape(image, [1, 256, 256, 3])
        return image

    def encode_img(self, image):
        # translate tensor into numpy
        image = image.numpy()
        # de-normalize image from [-1,1] to [0,255]
        image = np.add(image, 1)
        image = np.multiply(image, 127.5)
        # change dtype from tf.float32 to np.uint8
        image = image.astype(np.uint8)
        return image

    def generate_img(self):
        # generate_input image

        # save and load to convert image into tensor
        fname = "G:/Shared drives/2021_MAS/T3/Common Ground Control/01_data/02_demo/00_test/input.png"
        cv.imwrite(fname, self.input_img)
        input_tensor = self.load_img(fname)
        # generate_img
        prediction = self.model(input_tensor, training=True)
        # denormalized
        self.phenotype = self.encode_img(prediction[0])

    def cal_zdiff(self, arr1, arr2):
        arr1 = arr1.astype(np.float32)
        arr2 = arr2.astype(np.float32)
        # remap
        bounds = 85  # np.amax([np.ptp(arr1), np.ptp(arr2)])
        arr1 = np.interp(arr1, [np.amin(arr1), np.amin(arr1)+bounds], [0, bounds])
        arr2 = np.interp(arr2, [np.amin(arr2), np.amin(arr2)+bounds], [0, bounds])
        # subtraction
        arr_diff = np.subtract(arr1, arr2)
        # absolute
        arr_diff = np.absolute(arr_diff)
        # exponential
        powers = 1
        arr_diff = arr_diff ** powers
        # sum & percentage
        arr_diff = np.sum(arr_diff)
        total_error = (bounds ** powers) * 256 * 256
        fitness = arr_diff / total_error
        fitness = 1 - fitness
        return fitness

    def get_minmax(self, arr):
        min = np.amin(arr)
        max = np.amax(arr)
        return min, max

    def get_fitness(self, img1, img2):
        self.zdiff = self.cal_zdiff(img1, img2)
        return self.zdiff

    def fit(self):
        # generate phenotype
        self.generate_img()
        # fitness function
        raw_fitness = self.get_fitness(self.target_img, self.phenotype)
        # put at least one item into pool
        if raw_fitness < 0.01:
            raw_fitness += 0.01
        self.fitness = raw_fitness

    def crossover(self, parent_a, parent_b):

        midpoint = int(r.randint(0, self.genotype.shape[0]-1))
        for i in range(self.genotype.shape[0]):
            if i < midpoint:
                self.genotype[i] = parent_a.genotype[i]
            else:
                self.genotype[i] = parent_b.genotype[i]

    def mutation(self, metation_rate):
        for i in range(self.genotype.shape[0]):
            if r.random() < metation_rate:
                self.genotype[i][0] = r.randint(int(self.fframe_bounds[0][0]), int(self.fframe_bounds[0][1]))
                self.genotype[i][1] = r.randint(int(self.fframe_bounds[1][0]), int(self.fframe_bounds[1][1]))
                self.genotype[i][2] = r.randint(50, 100)

    def save_fig(self, tp_fname, ex_fname):
        cv.imwrite(tp_fname, self.input_img)
        cv.imwrite(ex_fname, self.phenotype)


if __name__ == '__main__':
    pass
