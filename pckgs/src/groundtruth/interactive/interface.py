from groundtruth.interactive.gh_scanning import get_heightmap, gray2height, height2ascii, scan
from groundtruth.interactive.gh_toolpath import export_json, flip_y_value, toolpath, generate_input_img, inverse_fframe, generate_toolpaths, save_img, blend_edges, move_ctrl_frames_to_robot
from groundtruth.interactive.gh_prediction import load_model, generate_img
from groundtruth.toolbox.features import Feature
import os


__HERE__ = os.path.dirname(__file__)
#__GH_DATA__ = os.path.join(__HERE__, '..', '..', '..', '..', 'grasshopper/data')
__GH_DATA__ = 'G:/Shared drives/2021_MAS/T3/Common Ground Control/01_data/02_demo/01_interactive-gh/01_data'
__GH_EXPORT__ = "G:/Shared drives/2021_MAS/T3/Common Ground Control/01_data/02_demo/01_interactive-gh/00_designs"


def scan_sandbox():
    s = scan()
    hm_feature = get_heightmap(s)
    hm_feature.save(fname='height_feature', path=__GH_DATA__)
    height_b = hm_feature.channel_split()[0]
    height = gray2height(height_b)
    path = height2ascii(height, path=__GH_DATA__)
    return path


def get_height_grid(hm_feature=None):
    height_b = hm_feature.channel_split()[0]
    height = gray2height(height_b)
    path = height2ascii(height, path=__GH_DATA__)
    return path


def prediction(toolpaths, adaptive):
    # load heigt map
    f = Feature()
    hm_feature = f.feature_from_file(path=os.path.join(__GH_DATA__, 'height_feature.png'))
    # load model
    model = load_model()
    # flip y to match feature
    toolpaths = flip_y_value(toolpaths)
    adapted_toolpaths = []
    for tp in toolpaths:
        # draw toolpath in feature
        t = toolpath(tp, __GH_DATA__, hm_feature, adaptive=adaptive)
        # generate input image
        input_img, M = generate_input_img(t, hm_feature, __GH_DATA__)
        # prediction
        predicted_img = generate_img(model, input_img)
        # patch predicted image into original height map
        inversed_img = inverse_fframe(M, t.crop_idx, hm_feature, predicted_img)
        # blend edges prediction and height feature
        blend = blend_edges(background=hm_feature.feature, prediction=inversed_img, crop=t.crop_idx)
        # update current state of sand
        hm_feature = Feature(blend)
        # store control points
        adapted_toolpaths.append(t.ctrlframes_feature)
    # save image for checking
    save_img(predicted_img, __GH_DATA__, 'predicted_fframe.png')
    save_img(inversed_img, __GH_DATA__, 'predicted_feature.png')
    # get path to ascii
    path = get_height_grid(hm_feature=hm_feature)
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


if __name__ == '__main__':
    # get_height_grid()
    toolpaths = generate_toolpaths()
    heightmap, a = prediction(toolpaths, 0)
    # scan_sandbox()
