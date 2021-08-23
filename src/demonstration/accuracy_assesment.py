# built-in
import sys
import time
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import cv2 as cv
import random as r
from tkinter.filedialog import askopenfilename, askdirectory
import os
import cv2

# package
from artist import Artist
from helper import Helper
# toolbox
from toolbox import TrainingData
from toolbox import ScanData, HeightMap, PointCloud
import toolbox.ur_helper as ur


__TEST__ = "G:/Shared drives/2021_MAS/T3/Common Ground Control/01_data/01_gan/00_dataset/dataset_lvl_all/h2h/test"
__ACC__ = "G:/Shared drives/2021_MAS/T3/Common Ground Control/01_data/01_gan/01_models/00000_2021-08-17/accuracy_assessment/h2h"

__ERRORS__ = []
__ERRORS_MAX__ = []


def get_iteration_dirs(path):
    iteration_dirs = os.listdir(path)
    try:
        iteration_dirs.remove('meta.json')
    except:
        pass
    iteration_dirs.sort()
    return iteration_dirs


def load_img(img_path):
    # Read and decode an image file to a uint8 tensor
    image = tf.io.read_file(img_path)
    image = tf.image.decode_png(image)

    # Split each image tensor into two tensors:
    # - one with a real building facade image
    # - one with an architecture label image
    w = tf.shape(image)[1]
    w = w // 2
    input_image = image[:, :w, :]
    real_image = image[:, w:, :]
    # Convert an image to float32 tensors
    input_image = tf.cast(input_image, tf.float32)
    #ground_truth = tf.cast(real_image, tf.float32)
    # Normalizing the images to [-1, 1]
    input_image = (input_image / 127.5) - 1
    # ground truth
    ground_truth = cv.imread(img_path)
    ground_truth = ground_truth[:, w:, :]

    # toolpath
    toolpath = cv.imread(img_path)
    toolpath = toolpath[:, :w, :]

    return input_image, ground_truth, toolpath

def predictions(image_paths):
    h = Helper()
    loaded_model = h.load_model()
    for path in image_paths:
        input_img, ground_truth, toolpath = load_img(path)
        h.generate_img(loaded_model, input_img)
        # cv.imshow('single_prediction', h.phenotype)
        # cv.waitKey(0)
        error, avg_error = calcalate_error(pred=h.phenotype, ground_truth=ground_truth)
        error_map = error_to_bgr(error)
        # save images
        id = os.path.basename(os.path.normpath(path))
        id = str(avg_error).zfill(3) + "_mm_" + id
        save_assessment(to_grayscale(h.phenotype[:,:,1]), img_type='prediction', id=id)
        save_assessment(to_grayscale(ground_truth[:,:,1]), img_type='groundtruth', id=id)
        save_assessment(error_map, img_type='error', id=id)
        combined = np.hstack((toolpath, to_grayscale(ground_truth[:,:,1]), to_grayscale(h.phenotype[:,:,1])))
        save_assessment(combined, img_type='result', id=id)

def sample():
    image_names = get_iteration_dirs(__TEST__)
    test_samples = r.sample(image_names, 500)
    image_paths = [ os.path.join(__TEST__, fname) for fname in test_samples ]
    predictions(image_paths)

def save_assessment(img, img_type, id):
    # create folder
    path_name = os.path.join(__ACC__, id)
    try:
        os.makedirs(path_name)
    except FileExistsError:
        print("Directory " , path_name,  " already exists")
    file_id = id[8:13]
    #save
    file_name = os.path.join(path_name, file_id + "_" + img_type + ".png")
    cv2.imwrite(file_name, img)


def calcalate_error(pred, ground_truth):
    h_pred = pred[:,:,1]
    h_gr = ground_truth[:,:,1]

    error = np.absolute(h_pred.astype(np.float32) - h_gr.astype(np.float32))
    error_int = np.around(error, decimals=0).astype(np.uint8)
    error_mm = error_to_mm(error)
    avg_error = round(np.mean(error_mm), 0)
    max_error = np.max(error_mm)

    __ERRORS_MAX__.append(max_error)
    __ERRORS__.append(avg_error)

    return error_int, int(avg_error)

def error_to_bgr(error):
    # hsv = np.zeros((256,256,3))
    # hsv[:,:,2] = np.ones((256, 256)) * 100
    # hsv[:,:,1] = error * 10
    max = np.ones((256, 256)) * 255
    error = max - error
    gray = to_grayscale(error)
    gray = gray.astype(np.uint8)
    im_colormap = cv2.applyColorMap(gray, cv2.COLORMAP_OCEAN)

    # bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    # print(bgr)
    return im_colormap

def error_to_mm(err):
    pix_v = 150. / 256.
    error_mm = err * pix_v
    return error_mm

def to_grayscale(array1d):
    gray = np.zeros((256,256,3))
    gray[:,:,0], gray[:,:,1], gray[:,:,2] = array1d, array1d, array1d
    return gray



if __name__ == '__main__':
    sample()
    print('AVG ERROR: ', sum(__ERRORS__)/len(__ERRORS__))
    print('MAX ERROR: ', max(__ERRORS_MAX__))
    print('AVG MAX ERROR: ', sum(__ERRORS_MAX__)/len(__ERRORS_MAX__))


