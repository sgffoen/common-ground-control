import os
import cv2 as cv
import json
import datetime
import math as m
import numpy as np
import random as r
import compas.geometry as cg
import compas.utilities as cu


class Artist(object):
    def __init__(self, dna, num_ctrl_pts, fframe_bounds):
        self.dna = dna
        self.facts = self.call_fact()
        self.num_ctrl_pts = num_ctrl_pts
        self.fframe_bounds = fframe_bounds
        self.ctrl_pts = self.dna.random_ctrl_pts(self.num_ctrl_pts, self.fframe_bounds)
        self.ctrl_frames = self.tuple_to_compas_frame()
        self.toolpath_feature = self.draw_polyline_in_sandbox2d()

    def call_fact(self):
        dir = os.getcwd()
        fname = "data_collection/data/facts.json"
        path = os.path.join(dir, fname)
        with open(path) as f:
            facts = json.load(f)
        return facts

    def tuple_to_compas_frame(self, curve_type='bezier', segments_num=50):
        ctrl_frames = []
        ctrl_pts = [cg.Point(tl[0], tl[1], tl[2]) for tl in self.ctrl_pts]

        if curve_type == 'polyline':
            polyline = cg.Polyline(ctrl_pts)
            pts_on_curve = polyline.divide_polyline(segments_num)

        elif curve_type == 'bezier':
            curve = cg.Bezier(ctrl_pts)
            pts_on_curve = []
            step = 1 / (segments_num-1)
            for i in range(segments_num):
                t = step * i
                pt_on_curve = curve.point(t)
                pts_on_curve.append(pt_on_curve)

        for a, b in cu.pairwise(range(len(pts_on_curve))):
            # get first pt
            pta = cg.Point(pts_on_curve[a][0],
                            pts_on_curve[a][1],
                            pts_on_curve[a][2])
            # get end pt
            ptb = cg.Point(pts_on_curve[b][0],
                            pts_on_curve[b][1],
                            pts_on_curve[b][2])
            # calc axis on xy plane
            xaxis = cg.Vector.from_start_end(pta, ptb)
            yaxis = cg.Vector.Zaxis().cross(xaxis)
            # flatten vectors
            xaxis.z = 0.
            yaxis.z = 0.
            ctrl_frames.append(cg.Frame(pta, yaxis, -xaxis))
        return ctrl_frames

    def remapValue(self, v, ori_Min, ori_Max, targetMin, targetMax):
        rv = ((v-ori_Min)/(ori_Max-ori_Min))*(targetMax-targetMin)+targetMin
        return rv

    def draw_polyline_in_sandbox2d(self):
        feature_xsize = int(abs(self.facts['feature_bounds']['max_bound'][1]
                                - self.facts['feature_bounds']['min_bound'][1]))
        feature_ysize = int(abs(self.facts['feature_bounds']['max_bound'][0]
                                - self.facts['feature_bounds']['min_bound'][0]))

        img = 255 * np.ones(shape=[m.floor(feature_ysize),
                                    m.floor(feature_xsize),
                                    3], dtype=np.uint8)

        for a, b in cu.pairwise(range(len(self.ctrl_frames))):
            pt_s = self.ctrl_frames[a].point
            pt_e = self.ctrl_frames[b].point
            z = self.remapValue(pt_s[2], 50, 100, 0, 255)
            cv.line(img,
                    (int(pt_s[0]), int(pt_s[1])),
                    (int(pt_e[0]), int(pt_e[1])),
                    color=(0, 0, z),  # red channel for toolpath height
                    thickness=2,
                    lineType=cv.FILLED)
        return img


class LineArtist(object):
    def __init__(self):
        img = 255 * np.ones(shape=[m.floor(256),
                                   m.floor(256),
                                   3], dtype=np.uint8)
        self.original_image = img
        self.clone = self.original_image.copy()

        cv.namedWindow('image')
        cv.setMouseCallback('image', self.extract_coordinates)

        # List to store start/end points
        self.image_coordinates = []
        self.count = 0

    def extract_coordinates(self, event, x, y, flags, parameters):
        # Record starting (x,y) coordinates on left mouse button click
        if (event == cv.EVENT_LBUTTONDOWN) and (self.count % 2 == 0):
            self.image_coordinates = [(x, y)]
            self.count += 1

        # Record ending (x,y) coordintes on left mouse bottom click
        elif (event == cv.EVENT_LBUTTONDOWN) and (self.count % 2 == 1):
            self.count += 1
            self.image_coordinates.append((x, y))
            print('Line: {}, Starting: {}, Ending: {}'.format(int(self.count/2), self.image_coordinates[0], self.image_coordinates[1]))

            # Draw line
            cv.line(self.clone, self.image_coordinates[0], self.image_coordinates[1], (0, 0, 255), thickness=2)
            cv.imshow("image", self.clone)

        # Clear drawing boxes on right mouse button click
        elif event == cv.EVENT_RBUTTONDOWN:
            self.clone = self.original_image.copy()

    def show_image(self):
        return self.clone

    def save_img(self):
        pass


if __name__ == '__main__':
    pass
    # line_artist = LineArtist()
    # while True:
    #     img = line_artist.show_image()
    #     cv.imshow('image', img)
    #     key = cv.waitKey(1)

    #     # Close program with keyboard 'q'
    #     if key == ord('q'):
    #         id = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    #         filename = 'G:/Shared drives/2021_MAS/T3/Common Ground Control/01_data/02_demo/00_test/' + 'test_{}.png'.format(id)
    #         cv.imwrite(filename, img)

    #         cv.destroyAllWindows()
    #         exit(1)
