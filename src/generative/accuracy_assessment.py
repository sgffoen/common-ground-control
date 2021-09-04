# built-in
import sys
import time
import numpy as np
from numpy.lib.function_base import average
import tensorflow as tf
import matplotlib.pyplot as plt
import cv2
import random as r
from tkinter.filedialog import askopenfilename, askdirectory
import os
from compas.utilities import remap_values, i_to_white

# packag
import groundtruth.toolbox.generative_utils as gu



__TEST__ = "C:/Users/simon/Documents/MAS DFAB/04_MAS_THESIS/05_data/training set/h2h/test"
__ACC__ = "C:/Users/simon/Documents/MAS DFAB/04_MAS_THESIS/05_data/training set/accuracy_assessment"

__ERRORS__ = []
__ERRORS_MAX__ = []

__DIFFS__ = []


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
    ground_truth = cv2.imread(img_path)
    ground_truth = ground_truth[:, w:, :]
    # toolpath
    toolpath = cv2.imread(img_path)
    toolpath = toolpath[:, :w, :]

    return input_image, ground_truth, toolpath

def predictions(image_paths):
    loaded_model = gu.load_model()
    for path in image_paths:
        input_img, ground_truth, toolpath = load_img(path)
        prediction = gu.generate_img(model=loaded_model, input_tensor=np.reshape(input_img, [1, 256, 256, 3]))

        error, avg_abs_error = calcalate_error(pred=prediction, ground_truth=ground_truth)
        error_map = error_to_bgr(error)
        # save images
        id = os.path.basename(os.path.normpath(path))
        id = str(avg_abs_error).zfill(3) + "_mm_" + id
        save_assessment(to_grayscale(prediction[:,:,1]), img_type='prediction', id=id)
        save_assessment(to_grayscale(ground_truth[:,:,1]), img_type='groundtruth', id=id)
        save_assessment(error_map, img_type='error', id=id)
        combined = np.hstack((toolpath, to_grayscale(ground_truth[:,:,1]), to_grayscale(prediction[:,:,1])))
        save_assessment(combined, img_type='result', id=id)

def sample():
    image_names = get_iteration_dirs(__TEST__)
    test_samples = r.sample(image_names, 2000)
    image_paths = [ os.path.join(__TEST__, fname) for fname in test_samples ]
    return image_paths

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

    error = h_pred.astype(np.float32) - h_gr.astype(np.float32)
    error = error_to_mm(error)
    error = error + 150.
    error_bounds = 20. #mm
    error = remap_values(error, target_min=0.0, target_max=255.0, original_min=150. - error_bounds, original_max=150. + error_bounds)
    error = np.reshape(error, (256,256))




    abs_error = np.absolute(h_pred.astype(np.float32) - h_gr.astype(np.float32))
    abs_error_int = np.around(abs_error, decimals=0).astype(np.uint8)
    abs_error_mm = error_to_mm(abs_error)
    avg_abs_error = round(np.mean(abs_error_mm), 0)
    max_abs_error = np.max(abs_error_mm)

    __ERRORS_MAX__.append(max_abs_error)
    __ERRORS__.append(avg_abs_error)

    return error, avg_abs_error

def error_to_bgr(error):
    gray = to_grayscale(error)
    gray = gray.astype(np.uint8)
    im_colormap = cv2.applyColorMap(gray, cv2.COLORMAP_RAINBOW)

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

def max_depth():
    samples = sample()
    for path in samples:
        input_img, ground_truth, toolpath = load_img(path)
        min_ff = np.min(toolpath[np.nonzero(toolpath)])
        min_gt = np.min(ground_truth[:,:,1])
        diff = min_ff.astype('float') - min_gt.astype('float')
        __DIFFS__.append(diff)
    print("MIN: ", min(__DIFFS__))
    print("MAX: ",max(__DIFFS__))
    print("AVG: ",sum(__DIFFS__)/len(__DIFFS__))



if __name__ == '__main__':
    samples = sample()
    predictions(samples)
    print('AVG ERROR: ', sum(__ERRORS__)/len(__ERRORS__))
    print('MAX ERROR: ', max(__ERRORS_MAX__))
    print('AVG MAX ERROR: ', sum(__ERRORS_MAX__)/len(__ERRORS_MAX__))

    #max_depth()
