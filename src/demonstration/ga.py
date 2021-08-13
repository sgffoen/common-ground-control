import compas.geometry as cg
import compas.utilities as cu
import numpy as np
import random as r
import math as m
import cv2 as cv
import time
import json
import os


class DNA(object):  # GENOTYPE
    def __init__(self, num_ctrl_pts, fframe_bounds):
        self.facts = self.call_fact()
        self.num_ctrl_pts = num_ctrl_pts
        self.fframe_bounds = fframe_bounds
        self.ctrl_pts = self.random_ctrl_pts()
        self.ctrl_frames = self.tuple_to_compas_frame()
        self.toolpath_feature = self.draw_polyline_in_sandbox2d()

    def call_fact(self):
        dir = os.getcwd()
        fname = "data_collection/data/facts.json"
        path = os.path.join(dir, fname)
        with open(path) as f:
            facts = json.load(f)
        return facts

    def random_ctrl_pts(self):
        arr = np.zeros((self.num_ctrl_pts, 3))
        for i in range(arr.shape[0]):
            arr[i][0] = r.randint(int(self.fframe_bounds[0][0]), int(self.fframe_bounds[0][1]))
            arr[i][1] = r.randint(int(self.fframe_bounds[1][0]), int(self.fframe_bounds[1][1]))
            arr[i][2] = r.randint(50, 100)
        return arr

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

    def fit(self, target_arr):
        score = 0
        for i in range(self.genes.shape[0]):
            if self.genes[i] == target_arr[i]:
                score += 1
        self.fitness = score / target_arr.shape[0]
        # exponential fitness
        # self.fitness = score ** 2
        # self.fitness = 2 ** score

    def crossover(self, dna2):
        child_dna = DNA(self.num_ctrl_pts, self.fframe_bounds)
        midpoint = int(r.randint(0, self.num_ctrl_pts))
        for i in range(self.num_ctrl_pts):
            if i > midpoint:
                child_dna.genes[i] = self.genes[i]
            else:
                child_dna.genes[i] = dna2.genes[i]
        return child_dna


    def mutation(self, metation_rate):
        for i in range(self.genes.shape[0]):
            if r.random() < metation_rate:
                self.genes[i] = r.randint(0, 10)


if __name__ == '__main__':
    pass
