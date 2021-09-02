import groundtruth.toolbox.compas_utils as cu
from groundtruth.toolbox.features import Feature
from groundtruth.toolbox.toolpath import Toolpath, Dimension
import compas.geometry as cg
import compas.datastructures as cd
import random as r
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
    t = Toolpath(toolpath, save_dir, d, hm_feature, adaptive)
    return t


def generate_input_img(t, hm_feature, dir):
    # get feature
    t_feature = Feature(t.img)
    # get warp transformation
    M = t_feature.get_warp_transformation(t.crop_idx)
    # get fframe
    t_fframe = t_feature.get_featureframe(M)
    hm_fframe = hm_feature.get_featureframe(M)
    # turn black to white
    t_fframe = t_feature.turn_background(t_fframe)
    # channels
    b, g, _ = hm_feature.channel_split(hm_fframe)
    _, _, r = t_feature.channel_split(t_fframe)
    r = r.astype(np.uint8)
    input_img = cv2.merge([b, g, r])

    input_fname = 'input.png'
    input_path = os.path.join(dir, input_fname)
    cv2.imwrite(input_path, input_img)

    t_fframe_name = 't_fframe.png'
    path = os.path.join(dir, t_fframe_name)
    cv2.imwrite(path, t_fframe)

    hm_fframe_name = 'hm_fframe.png'
    path = os.path.join(dir, hm_fframe_name)
    cv2.imwrite(path, hm_fframe)

    return input_path, M


def inverse_fframe(M, crop_idx, feature, fframe):
    rows, cols, chs = feature.feature.shape
    fframe_feature = cv2.warpPerspective(fframe, M, (cols, rows), flags=cv2.WARP_INVERSE_MAP)
    patched_img = overlay_fframe(feature, fframe_feature)
    return patched_img


def blend_edges(background, prediction, crop):
    size = prediction.shape
    # mask
    black = np.zeros(size,dtype=np.uint8)
    black.fill(0)
    pts_from = np.int32(crop)
    mask = cv2.fillConvexPoly(black, pts_from, (255,255,255))
    mask_blur  = cv2.GaussianBlur(mask,(3,3),0).astype('float') / 255.
    # blur edges
    img = prediction.astype('float') / 255.
    bg = background.astype('float') / 255.
    out  = bg * (1 - mask_blur)  + img * mask_blur
    out = (out * 255).astype('uint8')
    return out


def overlay_fframe(feature, fframe_feature):
    condition = (fframe_feature != 0)
    patched = np.where(condition, fframe_feature, feature.feature)
    return patched


def generate_toolpaths():
    xmax = 1135
    ymin = -737
    zmax = 100
    offset = 125
    num_crv = 1
    length = 50

    toolpaths = []
    for i in range(num_crv):
        x_coord = r.randint(offset, xmax-offset)
        y_coord = r.randint(ymin+offset, 0-offset)
        z_coord = r.randint(50, zmax)
        pt = cg.Point(x_coord, y_coord, z_coord)

        vec = cg.Vector(0, length, 0)

        T = cg.Translation.from_vector(vec)
        pt_end = pt.transformed(T)

        pl = cg.Polyline([pt, pt_end])
        toolpaths.append(pl)
    return toolpaths


def save_img(img, dir, fname):
    path = os.path.join(dir, fname)
    cv2.imwrite(path, img)


def move_ctrl_frames_to_robot(frames):
    d = Dimension()
    # move frames to origin so that coordinates of ctrl_frame match the pixel order
    framefrom = cg.Frame(cg.Point(0, 0, 0),
                         cg.Vector.Xaxis(),
                         cg.Vector.Yaxis())
    frameto = cg.Frame(cg.Point(d.feature_origin_x,
                                d.feature_origin_y,
                                d.feature_origin_z),
                       cg.Vector.Xaxis(),
                       cg.Vector.Yaxis())
    T = cg.Transformation.from_frame_to_frame(framefrom, frameto)
    ctrl_frame_robot = []
    for fs in frames:
        for f in fs:
            f = f.transformed(T)
            ctrl_frame_robot.append(f)
    return ctrl_frame_robot


if __name__ == '__main__':
    pass
