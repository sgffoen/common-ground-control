import tensorflow as tf
from tkinter.filedialog import askopenfilename, askdirectory
import cv2 as cv
import json
import numpy as np
import random as r
import compas.geometry as cg
import compas.utilities as cu
import math as m
from matplotlib import pyplot as plt
import os
import sys
sys.path.insert(0, 'C:/Users/trtku/OneDrive/Data/03_MAS/17_common_ground_control/01_git/common-ground-control/src/data_collection')
import UR as ur
from data import TrainingData
from scanning import ScanData, HeightMap, PointCloud
from ga_sample import DNA


def browse_file():
    filename = askopenfilename()
    return filename


def browse_dir():
    dir_name = askdirectory()
    return dir_name


def load_image():
    filename = browse_file()
    loaded_img = cv.imread(filename)
    return loaded_img


def save_fig(img_to_save, path):
    cv.imwrite(fname, img_to_save)


def cal_zdiff(arr1, arr2):
    arr_diff = np.subtract(arr1, arr2)
    arr_abs_diff = np.absolute(arr_diff)
    return arr_abs_diff


def get_minmax(arr):
    min = np.amin(arr)
    max = np.amax(arr)
    return min, max


def cal_MSE(img1, img2):
    # mean square error
    mse = (np.square(img1.astype(int)-img2.astype(int))).mean(axis=None)
    return mse


def crop_image(img, path):
    # Cropping an image
    left_image = img[0:256, 0:256]
    right_image = img[0:256, 256:512]

    # Save the cropped image
    fname = path + '/' + 'left_img.png'
    cv.imwrite(fname, left_image)
    fname = path + '/' + 'right_img.png'
    cv.imwrite(fname, right_image)


def load_model():
    dir_name = browse_dir()
    try:
        loaded_model = tf.keras.models.load_model(dir_name)
        print('model is loaded from {}\n'.format(dir_name))
        return loaded_model
    except FileNotFoundError:
        print('model, {}, does not exist\n'.format(dir_name))


def load(img_path, save=False):
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

    # Convert both images to float32 tensors
    input_image = tf.cast(input_image, tf.float32)
    real_image = tf.cast(real_image, tf.float32)

    return input_image, real_image


# Normalizing the images to [-1, 1]
def normalize(input_image, real_image):
    input_image = (input_image / 127.5) - 1
    real_image = (real_image / 127.5) - 1

    return input_image, real_image


def generate_images(model, test_input, tar, plot_dir, step=0):
    test_input = np.reshape(test_input, [1, 256, 256, 3])
    # tar = np.reshape(tar, [1, 256, 256, 3])

    prediction = model(test_input, training=True)
    plt.figure(figsize=(15, 15))

    display_list = [test_input[0], tar[0], prediction[0]]
    title = ['Input Image', 'Ground Truth', 'Predicted Image']

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


def call_fact():
    dir = os.getcwd()
    fname = "data_collection/data/facts.json"
    path = os.path.join(dir, fname)
    with open(path) as f:
        facts = json.load(f)
    return facts


def get_feature_center(facts):
    (f_bounds_xmin,
    f_bounds_ymin,
    f_bounds_zmin) = facts['feature_bounds']['min_bound']
    (f_bounds_xmax,
    f_bounds_ymax,
    f_bounds_zmax) = facts['feature_bounds']['max_bound']

    x = (f_bounds_xmax - f_bounds_xmin)/2
    y = (f_bounds_ymax - f_bounds_ymin)/2
    z = (f_bounds_zmax - f_bounds_zmin)/2
    return [y, x, z]


def get_fframe_bounds(facts, feature_center):
    xsize = facts['fig_size']['x']
    ysize = facts['fig_size']['y']

    xmin = feature_center[0] - (xsize/2)
    xmax = feature_center[0] + (xsize/2)
    ymin = feature_center[1] - (ysize/2)
    ymax = feature_center[1] + (ysize/2)

    x_range = [xmin, xmax]
    y_range = [ymin, ymax]

    return [x_range, y_range]


def get_fframe_corner(fframe_bounds):
    topleft = [fframe_bounds[0][0], fframe_bounds[1][0]]
    topright = [fframe_bounds[0][1], fframe_bounds[1][0]]
    bottomleft = [fframe_bounds[0][0], fframe_bounds[1][1]]
    bottomright = [fframe_bounds[0][1], fframe_bounds[1][1]]

    return [topleft, topright, bottomright, bottomleft]


def get_corp_idx(corners):
    idx = []
    for c in corners:
        coord = []
        for i in c:
            coord.append(int(i))
        idx.append(coord)
    return idx


def crop_feature(crop_idx, img):
    pts_from = np.float32(crop_idx)
    pts_to = np.float32([[0, 0],
                        [facts['fig_size']['x'], 0],
                        [facts['fig_size']['x'], facts['fig_size']['y']],
                        [0, facts['fig_size']['y']]])
    M = cv.getPerspectiveTransform(pts_from, pts_to)
    img_cropped = cv.warpPerspective(img,
                                        M,
                                        (int(facts['fig_size']['x']),
                                        int(facts['fig_size']['x'])))
    return img_cropped


def generate_ctrl_pts(fframe_bounds, num):
    ctrl_pts_list = []
    num_ctrl_pts = num
    for i in range(num_ctrl_pts):
        x = r.randrange(int(fframe_bounds[0][0]), int(fframe_bounds[0][1]))
        y = r.randrange(int(fframe_bounds[1][0]), int(fframe_bounds[1][1]))
        z = r.randrange(50, 100)
        ctrl_pts_list.append((x, y, z))
    return ctrl_pts_list


def tuple_to_compas_frame(ctrl_pts_list, curve_type='bezier', segments_num=50):
    ctrl_frames = []
    ctrl_pts = [cg.Point(tl[0], tl[1], tl[2]) for tl in ctrl_pts_list]

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


def remapValue(v, ori_Min, ori_Max, targetMin, targetMax):
    rv = ((v-ori_Min)/(ori_Max-ori_Min))*(targetMax-targetMin)+targetMin
    return rv


def draw_polyline_in_sandbox2d(ctrl_frames, facts):
    feature_xsize = int(abs(facts['feature_bounds']['max_bound'][1]
                            - facts['feature_bounds']['min_bound'][1]))
    feature_ysize = int(abs(facts['feature_bounds']['max_bound'][0]
                            - facts['feature_bounds']['min_bound'][0]))


    img = 255 * np.ones(shape=[m.floor(feature_ysize),
                                m.floor(feature_xsize),
                                3], dtype=np.uint8)

    for a, b in cu.pairwise(range(len(ctrl_frames))):
        pt_s = ctrl_frames[a].point
        pt_e = ctrl_frames[b].point
        z = remapValue(pt_s[2], 50, 100, 0, 255)
        cv.line(img,
                    (int(pt_s[0]), int(pt_s[1])),
                    (int(pt_e[0]), int(pt_e[1])),
                    color=(0, 0, z),  # red channel for toolpath height
                    thickness=2)
    return img


if __name__ == '__main__':
    save_dir = "G:/Shared drives/2021_MAS/T3/Common Ground Control/01_data/02_demo/00_test"

    # scan and initialize
    ur.ur_helper.scan_pose(scanning_time=5)
    data = TrainingData(iteration=0, environment='test')
    scan = ScanData()
    pcl_obj = PointCloud(scan)
    heightmap = HeightMap(scan)

    # store data
    data.scan_data = scan
    data.pointcloud = pcl_obj
    data.heightmap = heightmap
    data.store_data()
    print('{}: data is collected and stored'.format(data.identifier))
    feature = load_image()

    # crop feature into feature_frame
    facts = call_fact()
    feature_center = get_feature_center(facts)
    fframe_bounds = get_fframe_bounds(facts, feature_center)
    corners = get_fframe_corner(fframe_bounds)
    crop_idx = get_corp_idx(corners)
    fframe = crop_feature(crop_idx, feature)
    fname = save_dir + '/test_fframe.png'
    save_fig(fframe, fname)

    # draw toolpath inside the fframe
    ctrl_pts = generate_ctrl_pts(fframe_bounds, num=5)
    ctrl_frames = tuple_to_compas_frame(ctrl_pts, curve_type='bezier', segments_num=50)
    feature = draw_polyline_in_sandbox2d(ctrl_frames, facts)
    toolpath_fframe = crop_feature(crop_idx, feature)
    fname = save_dir + '/test_toolpath_fframe.png'
    save_fig(toolpath_fframe, fname)

    # overlay image

    # loaded_model = load_model()
    # image_path = "G:/Shared drives/2021_MAS/T3/Common Ground Control/01_data/02_demo/00_test/00000_2021-08-05_h2h_training.png"
    # input_img, target_img = load(image_path)
    # input_img, target_img = normalize(input_img, target_img)
    # # crop_image(test_input, save_dir)
    # generate_images(loaded_model, input_img, target_img, save_dir)

    # load 2 image to compare
    # img1 = load_image()
    # img2 = load_image()
    # print(img1.shape, img2.shape)

    # calc z value difference
    # arr_diff = cal_zdiff(img1, img2)
    # print(arr_diff.shape)

    # differenr type of difference
    # mse = cal_MSE(img1, img2)
    # print(mse)

    # get min & max difference
    # min, max = get_minmax(arr_diff)
    # print(min, max)

    # save difference image
    # save_path = browse_dir()
    # fname = save_path + '/test_diff.png'
    # save_fig(arr_diff, fname)
