import os
import sys
import json
import cv2 as cv
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tkinter.filedialog import askdirectory, askopenfilename


class Helper(object):
    def __init__(self):
        self.facts = self.call_fact()

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

    def load_img(self, img_path):
        # Read and decode an image file to a uint8 tensor
        image = tf.io.read_file(img_path)
        image = tf.image.decode_png(image)
        # Convert an image to float32 tensors
        image = tf.cast(image, tf.float32)
        # Normalizing the images to [-1, 1]
        image = (image / 127.5) - 1
        return image

    def load_model(self):
        dir_name = askdirectory()
        try:
            loaded_model = tf.keras.models.load_model(dir_name)
            print('model is loaded from {}\n'.format(dir_name))
            return loaded_model
        except FileNotFoundError:
            print('model, {}, does not exist\n'.format(dir_name))

    def custom_img_addition(self, height_img):
        fname = "G:/Shared drives/2021_MAS/T3/Common Ground Control/01_data/00_data_collection/01_production/00000_2021-08-05/01_processed/00000_2021-08-05_toolpath_featureframe_fix.png"
        self.toolpath_img = cv.imread(fname)
        arr = np.zeros([256, 256, 3], dtype=np.uint8)
        for i in range(256):
            for j in range(256):
                if self.toolpath_img[i][j][2] > 0:
                    arr[i][j] = self.toolpath_img[i][j]
                else:
                    arr[i][j] = height_img[i][j]
        return arr

    def decode(self, img_to_decode):
        decoded_img = tf.convert_to_tensor(img_to_decode)
        decoded_img = tf.cast(decoded_img, tf.float32)
        return decoded_img

    def normalize(self, img_to_normalize):
        normalized_img = (img_to_normalize / 127.5) - 1
        return normalized_img

    def denormalize(self, img_to_denormalize):
        img_to_denormalize = np.add(img_to_denormalize, 1)
        img_to_denormalize = np.multiply(img_to_denormalize, 127.5)
        return img_to_denormalize

    def encode(self, img_to_encode):
        img_to_encode = img_to_encode.astype(np.uint8)
        return img_to_encode

    def generate_img(self, model, input_img):
        input_tensor = np.reshape(input_img, [1, 256, 256, 3])

        prediction = model(input_tensor, training=True)

        output_img = prediction[0].numpy()
        img_denormalized = self.denormalize(output_img)
        img_encoded = self.encode(img_denormalized)
        self.phenotype = img_encoded


if __name__ == '__main__':
    save_dir = "G:/Shared drives/2021_MAS/T3/Common Ground Control/01_data/02_demo/00_test"
