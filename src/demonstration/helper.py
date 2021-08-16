import os
import sys
import json
import cv2 as cv
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tkinter.filedialog import askdirectory

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

    def load_model(self):
        dir_name = askdirectory()
        try:
            loaded_model = tf.keras.models.load_model(dir_name)
            print('model is loaded from {}\n'.format(dir_name))
            return loaded_model
        except FileNotFoundError:
            print('model, {}, does not exist\n'.format(dir_name))

    def decode(self, img_to_decode1, img_to_decode2):
        decoded_img1 = tf.image.decode_png(img_to_decode1)
        decoded_img1 = tf.cast(decoded_img1, tf.float32)
        decoded_img2 = tf.image.decode_png(img_to_decode2)
        decoded_img2= tf.cast(decoded_img2, tf.float32)
        return decoded_img1, decoded_img2

    def normalize(self, img_to_normalize1, img_to_normalize2):
        img_to_normalize1 = (img_to_normalize1 / 127.5) - 1
        img_to_normalize2 = (img_to_normalize2 / 127.5) - 1
        return img_to_normalize1, img_to_normalize2


    def generate_images(self, model, test_input, target_input, plot_dir, step=0):
        test_input = np.reshape(test_input, [1, 256, 256, 3])
        target_input = np.reshape(target_input, [1, 256, 256, 3])

        prediction = model(test_input, training=True)
        plt.figure(figsize=(15, 15))

        display_list = [test_input[0], target_input[0], prediction[0]]
        title = ['Input Image', 'Target Image', 'Predicted Image']

        for i in range(3):
            plt.subplot(1, 3, i+1)
            plt.title(title[i])
            # Getting the pixel values in the [0, 1] range to plot.
            plt.imshow(display_list[i] * 0.5 + 0.5)
            plt.axis('off')
        # plt.show()
        fname = os.path.join(plot_dir,
                            'predicted_img_{}.png'.format(step))
        plt.savefig(fname)
        return display_list[0]

    def cal_zdiff(self, arr1, arr2):
        arr_diff = np.subtract(arr1, arr2)
        arr_abs_diff = np.absolute(arr_diff)
        return arr_abs_diff

    def get_minmax(self, arr):
        min = np.amin(arr)
        max = np.amax(arr)
        return min, max



if __name__ == '__main__':
    save_dir = "G:/Shared drives/2021_MAS/T3/Common Ground Control/01_data/02_demo/00_test"
