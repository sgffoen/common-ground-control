from groundtruth.interactive.gh_scanning import *
from groundtruth.interactive.gh_toolpath import *
from groundtruth.toolbox.features import Feature
from groundtruth.toolbox.generative_utils import *
from groundtruth.toolbox.raster_utils import g2height
from groundtruth.toolbox.genetic import GA
import time
import numpy as np
import cv2
import os


__HERE__ = os.path.dirname(__file__)
# shared dbt drive
# __GH_DATA__ = 'G:/Shared drives/2021_MAS/T3/Common Ground Control/01_data/02_demo/01_interactive-gh/01_data'
# __GH_EXPORT__ = "G:/Shared drives/2021_MAS/T3/Common Ground Control/01_data/02_demo/01_interactive-gh/00_designs"
# shared temporary drive
__GH_EXPORT__ = "G:/Shared drives/Ko-Simon MAS thesis (temporary)/01_data/02_demo/01_interactive-gh/00_designs"
__GH_DATA__ = 'G:/Shared drives/Ko-Simon MAS thesis (temporary)/01_data/02_demo/01_interactive-gh/01_data'


__OFFSET_DIST__ = 5


def scan_sandbox(cellsize=1):
    s = scan()
    hm_feature = get_heightmap(s)
    hm_feature.save(fname='scan_height_feature', path=__GH_DATA__)

    path = get_height_grid(cellsize=int(cellsize), hm_feature=hm_feature, fname='scan_grid.asc')
    return path


def get_height_grid(cellsize, hm_feature, fname):
    height_g = hm_feature.channel_split()[1]
    height = g2height(height_g)
    path = height2ascii(height, path=__GH_DATA__, fname=fname, cellsize=cellsize)
    return path


def prediction(toolpaths, adaptive, cellsize=1, depth=0):
    # load heigt map
    f = Feature()
    hm_feature = f.feature_from_file(path=os.path.join(__GH_DATA__, 'scan_height_feature.png'))
    # load model
    #model = load_model()
    # flip y to match feature
    toolpaths = flip_y_value_frame(toolpaths)
    adapted_toolpaths = []
    for tp in toolpaths:
        # draw toolpath in feature
        t = toolpath(tp, __GH_DATA__, hm_feature, depth, adaptive=adaptive)
        # offset crop index
        offset_crop_idx = offset_crop_area(t.crop_idx, __OFFSET_DIST__)
        # generate input image
        input_img, M, M_offset, bbound, tbound = generate_input_img(t, hm_feature, __GH_DATA__, offset_crop_idx, offset_dist=__OFFSET_DIST__)
        # create tensor and predict
        input_tensor = load_input_tensor(input_img)
        # api prediction
        predicted_tensor = prediction_from_api(input_tensor)
        # prediction from loaded model
        ##predicted_img = prediction_from_model(model, input_tensor)
        predicted_img = generate_img(predicted_tensor)
        # remap back into original range
        remapped_predicted_img = inverse_remap_img(predicted_img, bbound, tbound)
        # offset image
        offset_crop_fframe = offset_fframe(remapped_predicted_img, __OFFSET_DIST__)
        # patch predicted image into original height map
        inversed_img = inverse_fframe(M_offset, hm_feature, offset_crop_fframe)
        # blend edges prediction and height feature
        blend = blend_edges(prediction=inversed_img, crop=offset_crop_idx, blur_ksize=39, mask_thickness=9)
        # update current state of sand
        hm_feature = Feature(blend)
        # store control points
        adapted_toolpaths.append(t.ctrlframes_feature)
    # save image for checking
    # save_img(predicted_img, __GH_DATA__, 'predicted_fframe.png')
    save_img(hm_feature.feature, __GH_DATA__, 'predicted_feature.png')
    # noise filtering
    hm_feature.remove_noise()
    # get path to ascii
    path = get_height_grid(cellsize=int(cellsize), hm_feature=hm_feature, fname='prediction_grid.asc')
    return path, adapted_toolpaths


def prediction_ga(toolpaths, target_path, searchspace, denoise=False):
    # load heigt map
    f = Feature()
    hm_feature = f.feature_from_file(path=os.path.join(__GH_DATA__, 'scan_height_feature.png'))
    # flip y to match feature
    toolpaths = flip_y_value_frame(toolpaths)
    for tp in toolpaths:
        # draw toolpath in feature
        t = toolpath(tp, __GH_DATA__, hm_feature, depth=0, adaptive=False)
        # offset crop index
        offset_crop_idx = offset_crop_area(t.crop_idx, __OFFSET_DIST__)
        # generate input image
        input_img, M, M_offset, bbound, tbound = generate_input_img(t, hm_feature, __GH_DATA__, offset_crop_idx, offset_dist=__OFFSET_DIST__)
        # create tensor and predict
        input_tensor = load_input_tensor(input_img)
        # api prediction
        predicted_tensor = prediction_from_api(input_tensor)
        # prediction from loaded model
        ##predicted_img = prediction_from_model(model, input_tensor)
        predicted_img = generate_img(predicted_tensor)
        # remap back into original range
        remapped_predicted_img = inverse_remap_img(predicted_img, bbound, tbound)
        # offset image
        offset_crop_fframe = offset_fframe(remapped_predicted_img, __OFFSET_DIST__)
        # patch predicted image into original height map
        inversed_img = inverse_fframe(M_offset, hm_feature, offset_crop_fframe)
        # blend edges prediction and height feature
        blend = blend_edges(prediction=inversed_img, crop=offset_crop_idx, blur_ksize=39, mask_thickness=9)
        # update current state of sand
        hm_feature = Feature(blend)
    if denoise:
        # noise filtering
        hm_feature.remove_noise()

    # ga
    target_img = cv2.imread(target_path)
    fname = os.path.join(__GH_DATA__, "scan_height_feature.png")
    initial_img = cv2.imread(fname)
    initial_height = initial_img[:, :, 1]
    ga = GA(target_img, initial_height)
    # get fitness
    prediction = hm_feature.feature[:, :, 1]
    fitness = ga.get_fitness(prediction, fframe=searchspace)

    return fitness


# def prediction_ga(toolpaths, searchspace_corner):
#     # load heigt map
#     f = Feature()
#     original_hm_feature = f.feature_from_file(path=os.path.join(__GH_DATA__, 'scan_height_feature.png'))
#     # flip y to match feature
#     toolpaths = flip_y_value_frame(toolpaths)
#     tp = toolpaths[0]
#     # draw toolpath in feature
#     t = toolpath(tp, __GH_DATA__, original_hm_feature, adaptive=False)
#     # offset crop index
#     offset_crop_idx = offset_crop_area(t.crop_idx, 5)
#     # generate input image
#     input_img, _, M_offset, bbound, tbound = generate_input_img(t, original_hm_feature, __GH_DATA__, offset_crop_idx, offset_dist=5)
#     # create tensor and predict
#     input_tensor = load_input_tensor(input_img)
#     # api prediction
#     predicted_tensor = prediction_from_api(input_tensor)
#     # prediction from loaded model
#     predicted_img = generate_img(predicted_tensor)
#     # remap back into original range
#     remapped_predicted_img = inverse_remap_img(predicted_img, bbound, tbound)
#     # offset image
#     offset_crop_fframe = offset_fframe(remapped_predicted_img, 5)
#     # patch predicted image into original height map
#     inversed_img = inverse_fframe(M_offset, original_hm_feature, offset_crop_fframe)
#     # blend edges prediction and height feature
#     blend = blend_edges(prediction=inversed_img, crop=offset_crop_idx, blur_ksize=39, mask_thickness=9)
#     # update current state of sand
#     predicted_hm_feature = Feature(blend)
#     #predicted_hm_feature.save('test', "C:/Users/simon/Documents/MAS DFAB/04_MAS_THESIS/05_data/ga_predictions")

#     # get crop index for search space
#     searchspace_corner = flip_y_value_point(searchspace_corner)
#     searchspace_crop_idx = [[sc.x, sc.y] for sc in searchspace_corner]

#     # get current feature frame inside search space
#     M_original = original_hm_feature.get_warp_transformation(searchspace_crop_idx)
#     original_hm_fframe = original_hm_feature.get_featureframe(M_original)
#     original_height = split_channel(original_hm_fframe)[1].astype(np.float32)

#     # get predicted feature frame inside search space
#     M = predicted_hm_feature.get_warp_transformation(searchspace_crop_idx)
#     predicted_hm_fframe = predicted_hm_feature.get_featureframe(M)
#     predicted_height = split_channel(predicted_hm_fframe)[1].astype(np.float32)

#     # get target fframe
#     fname = "G:/Shared drives/Ko-Simon MAS thesis (temporary)/01_data/02_demo/01_interactive-gh/01_data/target_img/predicted_fframe.png"
#     target_fframe = cv2.imread(fname)
#     target_height = split_channel(target_fframe)[1].astype(np.float32)

#     # get fitness
#     initial_error = np.abs(original_height - target_height)
#     max_error = np.amax(initial_error) * np.ones([256, 256])
#     weights = np.power(initial_error, 2)

#     predicted_error = np.abs(predicted_height - target_height)
#     prediction_fitness = np.abs(max_error - predicted_error)

#     fitness = prediction_fitness * weights
#     total_fitness = np.sum(fitness) ** 2

#     return total_fitness


def export_design(tp, iteration):
    # move toolpath to sandbox space.
    tp = move_ctrl_frames_to_robot(tp)
    # initiate data
    data = {}
    # store control frames
    for i, f_list in enumerate(tp):
        toolpath_num = str(i).zfill(3)
        toolpath_key = 'toolpath_{}'.format(toolpath_num)
        data[toolpath_key] = {}
        for j, f in enumerate(f_list):
            frame_num = str(j).zfill(3)
            frame_key = 'f_{}'.format(frame_num)
            data[toolpath_key][frame_key] = f.to_jsonstring()
    # export json
    export_json(dir=__GH_EXPORT__, data=data, iter=iteration)
    text = 'json is saved in {}'.format(__GH_EXPORT__)
    return text


def surface2image(dists):
    unum, vnum = 729, 1135
    dists = np.reshape(dists, [unum, vnum])
    image = cv2.merge([dists, dists, dists])
    fname = 'G:/Shared drives/Ko-Simon MAS thesis (temporary)/01_data/02_demo/01_interactive-gh/01_data/target.png'
    cv2.imwrite(fname, image)


if __name__ == '__main__':
    # scan_sandbox()
    # get_height_grid()
    toolpaths = generate_toolpaths(center=True)
    heightmap, a = prediction(toolpaths, 0)
    # model = load_model(dir_name="G:/Shared drives/Ko-Simon MAS thesis (temporary)/01_data/01_gan/01_models/00010_2021-09-06/model/1")
    # modify_model_signatures(model=model, save=False)

    #zeros = np.zeros()
