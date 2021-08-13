from argparser import parse_args
from learning import LearningData
from process import Processing
import matplotlib.pyplot as plt
import compas.geometry as cg
import compas.utilities as cu
import numpy as np
import math as m
import datetime
import time
import json
import cv2
import os

__FOLDER__ = "G:/Shared drives/2021_MAS/T3/Common Ground Control/01_data/"


def remapValue(v, ori_Min, ori_Max, targetMin, targetMax):
    rv = ((v-ori_Min)/(ori_Max-ori_Min))*(targetMax-targetMin)+targetMin
    return rv


def get_iteration_dirs():
    iteration_dirs = os.listdir('G:/Shared drives/2021_MAS/T3/Common Ground Control/01_data/00_data_collection/01_production')
    iteration_dirs.remove('meta.json')
    iteration_dirs.sort()
    return iteration_dirs


def aliasing(env):
    start = time.time()
    print("starting aliasing mode")
    print("environment: {} \n".format(env))

    iteration_dirs = get_iteration_dirs()

    for i,id in enumerate(iteration_dirs):
        # 1. accessing data_collection path/img
        data = LearningData(i, id, env)
        data.get_iter_dirs()

        # load json
        dir = data.path_name_raw
        fname = data.id + '_toolpath.json'
        fpath = os.path.join(dir, fname)
        with open(fpath, 'r') as i:
            facts = json.load(i)

        # get control frames from json
        ctrl_frames = []
        for i, c in enumerate(facts['ctrl_frames']):
            id_num = str(i).zfill(3)
            key = 'f_{}'.format(id_num)
            jsonstring = facts['ctrl_frames'][key]
            frame = cg.Frame.from_jsonstring(jsonstring)
            ctrl_frames.append(frame)

        # draw polyline in sandbox2d
        img = np.zeros(shape=[732,
                              1135,
                              3],
                              dtype=np.uint8)

        for a, b in cu.pairwise(range(len(ctrl_frames))):
            pt_s = ctrl_frames[a].point
            pt_e = ctrl_frames[b].point
            z = remapValue(pt_s[2], 0, 150, 0, 255)
            cv2.line(img,
                     (int(pt_s[0]), int(pt_s[1])),
                     (int(pt_e[0]), int(pt_e[1])),
                     color=(0, 0, z),  # red channel for toolpath height
                     thickness=2,
                     lineType=cv2.FILLED)

        # crop_toolpathbox2d_oriented(self, img, d):
        crop_idx = []
        for i in range(4):
            crop_idx.append(facts['frame_corner_pts'][str(i)])
        pts_from = np.float32(crop_idx)
        pts_to = np.float32([[0, 0],
                            [256, 0],
                            [256, 256],
                            [0, 256]])
        M = cv2.getPerspectiveTransform(pts_from, pts_to)
        img_cropped = cv2.warpPerspective(img,
                                          M,
                                          (int(256),int(256)),
                                          flags=cv2.WARP_FILL_OUTLIERS,
                                          borderMode=cv2.BORDER_TRANSPARENT)

        # check image
        # fig, ax = plt.subplots()
        # im = ax.imshow(img)
        # plt.show()
        # fig, ax = plt.subplots()
        # im = ax.imshow(img_cropped)
        # plt.show()

        # export images
        filename = data.path_name_processed + '/' + data.id + '_toolpath_feature_fix.png'
        cv2.imwrite(filename, img)
        filename = data.path_name_processed + '/' + data.id + '_toolpath_featureframe_fix.png'
        cv2.imwrite(filename, img_cropped)

        if i % 100 == 0:
            lap = (time.time()-start)/60
            print('\nfixing id: {} / {}\nLAP-TIME: {}\n'.format(i, len(iteration_dirs), lap))

    print('\nTotal fixing time: ', (time.time()-start)/60, ' min\n\n')


def meta_data(env):
    filepath = os.path.join(__FOLDER__,
                            "00_data_collection",
                            ("00_test" if env == 'test' else "01_production"),
                            "meta.json")
    with open(filepath, 'r') as f:
        data = json.load(f)

    max_scan_id = data['max_scan_id']

    return max_scan_id



if __name__ == "__main__":
    aliasing(env='production')
