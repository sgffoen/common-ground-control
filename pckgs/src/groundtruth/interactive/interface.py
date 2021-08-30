from groundtruth.interactive.gh_scanning import get_heightmap, gray2height, height2ascii, scan
from groundtruth.interactive.gh_toolpath import export_json, flip_y_value
import os


__HERE__ = os.path.dirname(__file__)
__GH_DATA__ = os.path.join(__HERE__, '..', '..', '..', '..', 'grasshopper/data')
__GH_FIX__ = "G:/Shared drives/2021_MAS/T3/Common Ground Control/01_data/02_demo/01_interactive-gh/00_designs"


def get_height_grid():
    s = scan()
    hm_feature = get_heightmap(s)
    height_b = hm_feature.channel_split()[0]
    height = gray2height(height_b)
    path = height2ascii(height)
    return path


def prediction(toolpaths, iteration):
    # flip y to match feature
    toolpaths = flip_y_value(toolpaths)
    # draw toolpath in feature

    # warp

    # prediction

    # warp back

    return toolpaths


def fix_design(toolpaths, iteration):
    data = {}
    export_json(dir=__GH_FIX__, data=data, iter=iteration)


if __name__ == '__main__':
    get_height_grid()
