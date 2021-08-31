from groundtruth.interactive.gh_scanning import get_heightmap, gray2height, height2ascii, scan
from groundtruth.interactive.gh_toolpath import export_json, flip_y_value, toolpath, generate_input_img, inverse_fframe, generate_toolpaths, save_img, feature
from groundtruth.interactive.gh_prediction import load_model, generate_img
import os


__HERE__ = os.path.dirname(__file__)
__GH_DATA__ = os.path.join(__HERE__, '..', '..', '..', '..', 'grasshopper/data')
__GH_FIX__ = "G:/Shared drives/2021_MAS/T3/Common Ground Control/01_data/02_demo/01_interactive-gh/00_designs"


def scan_sandbox():
    s = scan()
    hm_feature = get_heightmap(s)
    height_b = hm_feature.channel_split()[0]
    height = gray2height(height_b)
    path = height2ascii(height)
    return path


def get_height_grid(hm_feature=None):
    height_b = hm_feature.channel_split()[0]
    height = gray2height(height_b)
    path = height2ascii(height)
    return path


def prediction(toolpaths, adaptive):
    # scan
    s = scan()
    hm_feature = get_heightmap(s)
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
        # update current state of sand
        hm_feature = feature(inversed_img)
        # store control points
        adapted_toolpaths.append(t.ctrlframes_feature)
    # save image for checking
    save_img(predicted_img, __GH_DATA__, 'predicted_fframe.png')
    save_img(inversed_img, __GH_DATA__, 'predicted_feature.png')
    # get path to ascii
    path = get_height_grid(hm_feature=hm_feature)
    return path, adapted_toolpaths


def fix_design(tp, iteration):
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
    export_json(dir=__GH_FIX__, data=data, iter=iteration)


if __name__ == '__main__':
    # get_height_grid()
    # toolpaths = generate_toolpaths()
    # heightmap = prediction(toolpaths, 0)
    pass
