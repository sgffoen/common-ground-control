from groundtruth.interactive.gh_scanning import get_heightmap, gray2height, height2ascii, scan
from groundtruth.interactive.gh_toolpath import export_json, flip_y_value, toolpath, generate_input_img, inverse_fframe
from groundtruth.interactive.gh_prediction import load_model, generate_img
import os


__HERE__ = os.path.dirname(__file__)
__GH_DATA__ = os.path.join(__HERE__, '..', '..', '..', '..', 'grasshopper/data')
__GH_FIX__ = "G:/Shared drives/2021_MAS/T3/Common Ground Control/01_data/02_demo/01_interactive-gh/00_designs"


def get_height_grid(predicted_img):
    if predicted_img:
        height_b = predicted_img[:,:,0]
    else:
        s = scan()
        hm_feature = get_heightmap(s)
        height_b = hm_feature.channel_split()[0]
    height = gray2height(height_b)
    path = height2ascii(height)
    return path


def prediction(toolpaths, iteration):
    # load model
    model = load_model()
    # scan
    s = scan()
    heightmap = get_heightmap(s)
    # flip y to match feature
    toolpaths = flip_y_value(toolpaths)
    for tp in toolpaths:
        # draw toolpath in feature
        t = toolpath(tp, __GH_DATA__, heightmap, adaptive=False)
        # get feature frame
        input_img, M = generate_input_img(t, heightmap, __GH_DATA__)
        # prediction
        predicted_img = generate_img(model, input_img)
        # patch predicted image into original height map
        inversed_img = inverse_fframe(M, t.crop_idx, heightmap, predicted_img)
        # update current state of sand
        heightmap = inversed_img
    return heightmap


def fix_design(toolpaths, iteration):
    data = {}
    export_json(dir=__GH_FIX__, data=data, iter=iteration)


if __name__ == '__main__':
    get_height_grid()
