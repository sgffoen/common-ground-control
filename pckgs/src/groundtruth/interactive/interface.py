from groundtruth.interactive.gh_scanning import *
from groundtruth.interactive.gh_toolpath import *
from groundtruth.toolbox.features import Feature
from groundtruth.toolbox.generative_utils import *
from groundtruth.toolbox.raster_utils import g2height
import cv2
import os


__HERE__ = os.path.dirname(__file__)
# shared dbt drive
# __GH_DATA__ = 'G:/Shared drives/2021_MAS/T3/Common Ground Control/01_data/02_demo/01_interactive-gh/01_data'
# __GH_EXPORT__ = "G:/Shared drives/2021_MAS/T3/Common Ground Control/01_data/02_demo/01_interactive-gh/00_designs"
# shared temporary drive
__GH_EXPORT__ = "G:/Shared drives/Ko-Simon MAS thesis (temporary)/01_data/02_demo/01_interactive-gh/00_designs"
__GH_DATA__ = 'G:/Shared drives/Ko-Simon MAS thesis (temporary)/01_data/02_demo/01_interactive-gh/01_data'
# ko's local
# __GH_EXPORT__ = "C:/Users/trtku/OneDrive/Data/03_MAS/17_common_ground_control/03_demo/01_interactive-gh/00_designs"
# __GH_DATA__ = 'C:/Users/trtku/OneDrive/Data/03_MAS/17_common_ground_control/03_demo/01_interactive-gh/01_data'
# __MODELDIR__ = "C:/Users/trtku/OneDrive/Data/03_MAS/17_common_ground_control/02_gan/01_models/00010_2021-09-06/model"


__OFFSET_DIST__ = 5


def scan_sandbox():
    s = scan()
    hm_feature = get_heightmap(s)
    hm_feature.save(fname='scan_height_feature', path=__GH_DATA__)
    height_g = hm_feature.channel_split()[1]
    height = g2height(height_g)
    path = height2ascii(height, path=__GH_DATA__, fname='scan_grid.asc')
    return path


def get_height_grid(hm_feature, fname):
    height_g = hm_feature.channel_split()[1]
    height = g2height(height_g)
    path = height2ascii(height, path=__GH_DATA__, fname=fname)
    return path


def prediction(toolpaths, adaptive):
    # load heigt map
    f = Feature()
    hm_feature = f.feature_from_file(path=os.path.join(__GH_DATA__, 'scan_height_feature.png'))
    # load model
    model = load_model()
    # flip y to match feature
    toolpaths = flip_y_value(toolpaths)
    adapted_toolpaths = []
    for tp in toolpaths:
        # draw toolpath in feature
        t = toolpath(tp, __GH_DATA__, hm_feature, adaptive=adaptive)
        # offset crop index
        offset_crop_idx = offset_crop_area(t.crop_idx, __OFFSET_DIST__)
        # generate input image
        input_img_path, M, M_offset, bbound, tbound = generate_input_img(t, hm_feature, __GH_DATA__, offset_crop_idx, offset_dist=__OFFSET_DIST__)
        # create tensor and predict
        input_tensor = load_input_tensor(input_img_path)
        predicted_img = generate_img(model, input_tensor)
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
    save_img(predicted_img, __GH_DATA__, 'predicted_fframe.png')
    save_img(hm_feature.feature, __GH_DATA__, 'predicted_feature.png')
    # get path to ascii
    path = get_height_grid(hm_feature=hm_feature, fname='prediction_grid.asc')
    return path, adapted_toolpaths


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


if __name__ == '__main__':
    # scan_sandbox()
    # get_height_grid()
    toolpaths = generate_toolpaths(center=True)
    heightmap, a = prediction(toolpaths, 0)
