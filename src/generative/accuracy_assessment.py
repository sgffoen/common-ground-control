# built-in
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import cv2
import random as r
import os
from compas.utilities import remap_values
from math import sqrt
import json
import compas.geometry as cg

# packag
import groundtruth.toolbox.generative_utils as gu
from groundtruth.toolbox.features import Feature
import groundtruth.interactive.gh_toolpath as ght


__TEST__ = "G:/Shared drives/Ko-Simon MAS thesis (temporary)/01_data/01_gan/00_dataset/dataset_lvl_all/tgd2tgd_thick/test"
#__ACC__ = "G:/Shared drives/Ko-Simon MAS thesis (temporary)/01_data/02_gan/01_models/00010_2021-09-06/accuracy/00_data"
__ACC__ = "C:/Users/simon/Documents/MAS DFAB/04_MAS_THESIS/03_documentation/01_images/02_final/errors"


__ERRORS__ = []
__ERRORS_MAX__ = []
__ERRORS_MIN__ = []
__RMSE__ = []
__MEAN_ERRORS__ = []
__TOT_ERROR__ = []

__DIFFS__ = []


def get_iteration_dirs(path):
    iteration_dirs = os.listdir(path)
    try:
        iteration_dirs.remove('meta.json')
    except:
        pass
    iteration_dirs.sort()
    return iteration_dirs


def load_model(dir_name="G:/Shared drives/Ko-Simon MAS thesis (temporary)/01_data/01_gan/01_models/00010_2021-09-06/model"):
    try:
        loaded_model = tf.keras.models.load_model(dir_name)
        print('model is loaded from {}\n'.format(dir_name))
        return loaded_model
    except FileNotFoundError:
        print('model, {}, does not exist\n'.format(dir_name))


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
    input_image = np.reshape(input_image, [1, 256, 256, 3])
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
    loaded_model = load_model(dir_name="G:/Shared drives/Ko-Simon MAS thesis (temporary)/01_data/01_gan/01_models/00010_2021-09-06/model/1")
    for path in image_paths:
        input_img, ground_truth, toolpath = load_img(path)
        prediction = gu.prediction_from_model(model=loaded_model, input_tensor=input_img)
        prediction = gu.generate_img(prediction=prediction)

        error, avg_abs_error = calcalate_error(pred=prediction, ground_truth=ground_truth)
        if error is not None:
            error_map = error_to_bgr(error)
            # save images
            id = os.path.basename(os.path.normpath(path))
            id = str(avg_abs_error).zfill(3) + "_mm_" + id
            # save_assessment(to_grayscale(prediction[:,:,1]), img_type='prediction', id=id)
            # save_assessment(to_grayscale(ground_truth[:,:,1]), img_type='groundtruth', id=id)
            # save_assessment(error_map, img_type='error', id=id)
            # 4 combination: input/ground truth/prediction/accuracy
            # combined = np.hstack((toolpath, to_grayscale(ground_truth[:,:,1]), to_grayscale(prediction[:,:,1]), error_map))
            # save_assessment(combined, img_type='result', id=id)
            #save_demonstration_error(error_map, id=id)


def sample(n):
    image_names = get_iteration_dirs(__TEST__)
    test_samples = r.sample(image_names, n)
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


def save_demonstration_error(img, id):
    path_name = __ACC__
    file_id = id[8:13]
    #save
    file_name = os.path.join(path_name, file_id + ".png")
    cv2.imwrite(file_name, img)


def calcalate_error(pred, ground_truth, error_bounds=30.):
    h_pred = pred#[:,:,1]
    h_gr = ground_truth#[:,:,1]

    error = h_pred.astype(np.float32) - h_gr.astype(np.float32)
    error_mm = error_to_mm(error)
    # make range positive (0 -- 300mm), 150 is 0 point -> so 0 is -150mm
    error_mm = error_mm + 150.
    error_pix = remap_values(error_mm, target_min=0.0, target_max=255.0, original_min=150. - error_bounds, original_max=150. + error_bounds)
    error_pix = np.reshape(error_pix, h_pred.shape)




    abs_error = np.absolute(h_pred.astype(np.float32) - h_gr.astype(np.float32))
    abs_error_int = np.around(abs_error, decimals=0).astype(np.uint8)
    abs_error_mm = error_to_mm(abs_error)
    avg_abs_error = round(np.mean(abs_error_mm), 0)
    max_abs_error = np.max(abs_error_mm)
    min_abs_error = np.min(abs_error_mm)


    if max_abs_error < 35.:

        __RMSE__.append(calc_RMSE(pred, ground_truth))

        __ERRORS_MIN__.append(min_abs_error)
        __ERRORS_MAX__.append(max_abs_error)
        __ERRORS__.append(avg_abs_error)
        __MEAN_ERRORS__.append(np.mean(abs_error_mm))
        __TOT_ERROR__.append(np.sum(abs_error_mm))
        return error_pix, avg_abs_error
    else:
        return None, None

    #return error_pix, avg_abs_error


def calc_RMSE(predictions, observations):
    h_pred = predictions#[:,:,1]
    h_gr = observations#[:,:,1]

    abs_error_sqr = (h_pred.astype(np.float32) - h_gr.astype(np.float32))**2
    mean_sqr = np.mean(abs_error_sqr)

    rmse = sqrt(mean_sqr)

    return rmse


def plot_prob_dens_func():
    from scipy.stats.kde import gaussian_kde
    # create fake data
    data = __RMSE__
    # this create the kernel, given an array it will estimate the probability over that values
    kde = gaussian_kde( data )
    # these are the values over wich your kernel will be evaluated
    dist_space = np.linspace( min(data), max(data), len(data) )
    # plot the results
    plt.plot( dist_space, kde(dist_space),color="black" )
    plt.grid()
    plt.xlabel('RMSE (mm)', fontsize=11)
    plt.ylabel('Probability density', fontsize=11)
    plt.show()


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
    shape0, shape1 = array1d.shape[0], array1d.shape[1]
    gray = np.zeros((shape0,shape1,3))
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


def max_bounds():
    samples = sample()
    for path in samples:
        input_img, ground_truth, toolpath = load_img(path)
        min = np.min(ground_truth[:,:,1])
        max = np.max(ground_truth[:,:,1])
        diff = max.astype('float') - min.astype('float')
        if diff > 180.0:
            print(path)
        __DIFFS__.append(diff)
    print("MIN: ", np.min(__DIFFS__))
    print("MAX: ",np.max(__DIFFS__))
    print("AVG: ",sum(__DIFFS__)/len(__DIFFS__))


def get_toolpath_crnr_pts(path):
    with open(path) as f:
        data = json.load(f)
    read = data["frame_corner_pts"]
    return read


def get_toolpath_compas_frames(path):
    with open(path) as f:
        data = json.load(f)
    # get control frames from json
    ctrl_frames = []
    for j, c in enumerate(data['ctrl_frames']):
        id_num = str(j).zfill(3)
        key = 'f_{}'.format(id_num)
        jsonstring = data['ctrl_frames'][key]
        frame = cg.Frame.from_jsonstring(jsonstring)
        ctrl_frames.append(frame)

    return ctrl_frames


def get_height(arr):
    arr1 =  gu.split_channel(arr)[1].astype(np.float32)
    cropped_arr = create_designspace(arr1)
    return cropped_arr


def create_designspace(arr):
    offset = 50  # 100mm
    return arr[offset:729-offset, offset:1135-offset]


def plot_rmse(errors):
    errors = errors[1:]
    nsteps = list(range(len(errors)))
    # plot the results
    plt.plot( nsteps, errors, color="black" )
    plt.grid()
    plt.xlabel('Number of toolpaths', fontsize=11)
    plt.ylabel('RMSE (mm)', fontsize=11)
    plt.show()


def plot_mean_errors(errors):
    errors = errors[1:]
    nsteps = list(range(len(errors)))
    # plot the results
    plt.plot( nsteps, errors, color="black" )
    plt.grid()
    plt.xlabel('Number of toolpaths', fontsize=11)
    plt.ylabel('Mean error (mm)', fontsize=11)
    plt.show()


def plot_max_errors(errors):
    errors = errors[1:]
    nsteps = list(range(len(errors)))
    # plot the results
    plt.plot( nsteps, errors, color="black" )
    plt.grid()
    plt.xlabel('Number of toolpaths', fontsize=11)
    plt.ylabel('Max error (mm)', fontsize=11)
    plt.show()


def plot_total_error(errors):
    errors = errors[1:]
    nsteps = list(range(len(errors)))
    # plot the results
    plt.plot( nsteps, errors, color="black" )
    plt.grid()
    plt.xlabel('Number of toolpaths', fontsize=11)
    plt.ylabel('Total error (mm)', fontsize=11)
    plt.show()


def plot_error_change(errors):
    errors = errors[1:]
    d_err = []
    for i in range(len(errors)-1):
        d = errors[i+1] - errors[i]
        d_err.append(d)
    nsteps = list(range(len(errors)-1))
    # plot the results
    plt.plot( nsteps, d_err, color="black" )
    plt.grid()
    plt.xlabel('Number of toolpaths', fontsize=11)
    plt.ylabel('Error delta (mm)', fontsize=11)
    plt.show()


def accrued_error():
    # SETUP DATA

    DATA_COLLECTION = 'G:/Shared drives/Ko-Simon MAS thesis (temporary)/01_data/00_data_collection'
    TEST_DATA = "G:/Shared drives/Ko-Simon MAS thesis (temporary)/01_data/01_gan/00_dataset/dataset_1500-1600"
    INITIAL_HEIGHTMAP = "G:/Shared drives/Ko-Simon MAS thesis (temporary)/01_data/00_data_collection/01501_2021-08-05/01_processed/01501_2021-08-05_height_feature.png"

    data_collection_dirs = get_iteration_dirs(DATA_COLLECTION)
    test_data_dirs = get_iteration_dirs(TEST_DATA)

    initial_feat = Feature()
    initial_height_feat = initial_feat.feature_from_file(path=INITIAL_HEIGHTMAP)

    START_INDEX = 1501
    data_date = '2021-08-05'

    #model = load_model(dir_name="G:/Shared drives/Ko-Simon MAS thesis (temporary)/01_data/01_gan/01_models/00010_2021-09-06/model/1")

    __OFFSET_DIST__ = 5
    ########################

    for i in range(len(data_collection_dirs[0:-1])):
        index = START_INDEX + i
        identifier = '0' + str(index) + '_' + data_date
        identifier_after = '0' + str(index+1) + '_' + data_date
        # load heigt map
        hm_feature_before = initial_height_feat

        # toolpath
        tp_json = os.path.join(DATA_COLLECTION, data_collection_dirs[i], '00_RAW', identifier + '_toolpath.json')
        #fframe_pts = get_toolpath_crnr_pts(path=tp_json)
        ctrl_frames_orig = get_toolpath_compas_frames(path=tp_json)
        ctrl_frames = []
        for c in ctrl_frames_orig:
            c.point.z = 130.0 - c.point.z
            ctrl_frames.append(c)
        t = ght.toolpath(ctrl_frames, os.path.join(DATA_COLLECTION,data_collection_dirs[i]), hm_feature_before, depth=0, adaptive=False)
        offset_crop_idx = gu.offset_crop_area(t.crop_idx, __OFFSET_DIST__)

        # generate input image
        input_img, M, M_offset, bbound, tbound = ght.generate_input_img(t, hm_feature_before, os.path.join(DATA_COLLECTION,data_collection_dirs[i]), offset_crop_idx, offset_dist=__OFFSET_DIST__)

        # create tensor and predict
        input_tensor = gu.load_input_tensor(input_img)
        # api prediction
        #predicted_tensor = gu.prediction_from_model(model=model, input_tensor=input_tensor)
        predicted_tensor = gu.prediction_from_api(input_tensor=input_tensor)
        # prediction from loaded model
        ##predicted_img = prediction_from_model(model, input_tensor)
        predicted_img = gu.generate_img(predicted_tensor)
        # remap back into original range
        remapped_predicted_img = gu.inverse_remap_img(predicted_img, bbound, tbound)
        # offset image
        offset_crop_fframe = gu.offset_fframe(remapped_predicted_img, __OFFSET_DIST__)
        # patch predicted image into original height map
        inversed_img = ght.inverse_fframe(M_offset, hm_feature_before, offset_crop_fframe)
        # blend edges prediction and height feature
        blend = ght.blend_edges(prediction=inversed_img, crop=offset_crop_idx, blur_ksize=39, mask_thickness=9)
        # update current state of sand
        hm_feature_prediction = Feature(blend)
        hm_feature_prediction.save(identifier, "G:/Shared drives/Ko-Simon MAS thesis (temporary)/01_data/01_gan/01_models/00010_2021-09-06/accuracy/01_accrued/00_pred")

        hm_feature_gt = Feature().feature_from_file(path=os.path.join(DATA_COLLECTION, data_collection_dirs[i+1], '01_processed', identifier_after + '_height_feature.png') )
        hm_feature_gt.save(identifier, "G:/Shared drives/Ko-Simon MAS thesis (temporary)/01_data/01_gan/01_models/00010_2021-09-06/accuracy/01_accrued/01_groundtruth")

        h_pred = get_height(hm_feature_prediction.feature)
        h_gt = get_height(hm_feature_gt.feature)

        error_pix, avg_abs_error = calcalate_error(pred=h_pred, ground_truth=h_gt)

        initial_height_feat = hm_feature_prediction

        error_map = error_to_bgr(error_pix)
        # save images
        filename=os.path.join("G:/Shared drives/Ko-Simon MAS thesis (temporary)/01_data/01_gan/01_models/00010_2021-09-06/accuracy/01_accrued/02_errormap", str(index) + '.png')
        cv2.imwrite(filename, img=error_map)

    plot_prob_dens_func()
    plot_rmse(errors=__RMSE__)
    plot_mean_errors(errors=__MEAN_ERRORS__)
    plot_max_errors(errors=__ERRORS_MAX__)
    #plot_total_error(errors=__TOT_ERROR__)
    plot_error_change(errors=__MEAN_ERRORS__)




if __name__ == '__main__':
    # samples = sample(n=1000)
    # predictions(samples)
    # print('AVG ERROR: ', sum(__ERRORS__)/len(__ERRORS__))
    # print('MAX ERROR: ', max(__ERRORS_MAX__))
    # print('AVG MAX ERROR: ', sum(__ERRORS_MAX__)/len(__ERRORS_MAX__))
    # #print('MAX ERRORS: ', __ERRORS_MAX__)
    # print('AVG MIN ERROR: ', sum(__ERRORS_MIN__)/len(__ERRORS_MIN__))
    # print('AVG RMSE ERROR: ', sum(__RMSE__)/len(__RMSE__))
    # plot_prob_dens_func()


    accrued_error()
    print('AVG ERROR: ', sum(__ERRORS__)/len(__ERRORS__))
    print('MAX ERROR: ', max(__ERRORS_MAX__))
    print('AVG MAX ERROR: ', sum(__ERRORS_MAX__)/len(__ERRORS_MAX__))
    #print('MAX ERRORS: ', __ERRORS_MAX__)
    print('AVG MIN ERROR: ', sum(__ERRORS_MIN__)/len(__ERRORS_MIN__))
    print('AVG RMSE ERROR: ', sum(__RMSE__)/len(__RMSE__))




    #max_depth()
    # max_bounds()
    # test_f = ["C:/Users/simon/Documents/MAS DFAB/04_MAS_THESIS/05_data/training set/h2h/test",
    #           "G:/Shared drives/Ko-Simon MAS thesis (temporary)/01_data/02_gan/00_dataset/dataset_lvl_all/tgd2tgd_thick/test"  ]
    # __ACC2__ = "C:/Users/simon/Documents/MAS DFAB/04_MAS_THESIS/05_data/training set/accuracy_assessment/07-09-2021"
    # __TEST2__ = "C:/Users/simon/Documents/MAS DFAB/04_MAS_THESIS/05_data/training set/h2h/test"
