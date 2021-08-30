import groundtruth.toolbox.compas_utils as cu
from groundtruth.toolbox import Feature
from groundtruth.toolbox import Toolpath, Dimension
import compas.geometry as cg
import compas.datastructures as cd
import numpy as np
import datetime
import json
import cv2
import os


def export_json(dir, data, iter):
    id_num = str(iter).zfill(3)
    filepath = dir + '/{}_toolpaths.json'.format(id_num)
    with open(filepath, 'w') as o:
        json.dump(data, o, indent=4)


def import_json(dir):
    # filepath = dir +
    # with open(filepath, 'r') as i:
    #     data = json.load()
    pass


def flip_y_value(toolpaths):
    for tp in toolpaths:
        for p in tp.points:
            p.y *= (-1)
    return toolpaths


def toolpath(toolpath, save_dir, hm_feature, adaptive):
    d = Dimension()
    t = Toolpath(toolpath, save_dir, d, hm_feature, adaptive=adaptive)
    return t


def generate_input_img(t, height_map, dir):
    # get feature
    t_feature = Feature(t.img)
    hm_feature = Feature(height_map)
    # get warp transformation
    M = t_feature.get_warp_transformation(t.crop_idx)
    # get fframe
    t_fframe = t_feature.get_featureframe(M)
    hm_fframe = hm_feature.get_featureframe(M)
    # overlay
    input_img = t_feature.img_overlay(hm_fframe, t_fframe)
    # save image
    fname = 'input.png'
    path = os.path.join(dir, fname)
    cv2.imwrite(path, input_img)

    return path, M


def inverse_fframe(M, crop_idx, feature, fframe):
    rows, cols, chs = feature.shape
    fframe_feature = cv2.warpPerspective(fframe, M, (cols, rows), flags=cv2.WARP_INVERSE_MAP)
    patched_img = overlay_fframe(feature, fframe_feature)
    return patched_img


def overlay_fframe(feature, fframe_feature):
    condition = (fframe_feature != 0)
    patched = np.where(condition, fframe_feature, feature)
    return patched


if __name__ == '__main__':
    pass
