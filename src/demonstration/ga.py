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
    def __init__(self):
        self.genes = self.random_ctrl_pts()

    def random_ctrl_pts(self, num_ctrl_pts, fframe_bounds):
        arr = np.zeros((num_ctrl_pts, 3))
        for i in range(arr.shape[0]):
            arr[i][0] = r.randint(int(fframe_bounds[0][0]), int(fframe_bounds[0][1]))
            arr[i][1] = r.randint(int(fframe_bounds[1][0]), int(fframe_bounds[1][1]))
            arr[i][2] = r.randint(50, 100)
        return arr

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
        child_dna = DNA()
        midpoint = int(r.randint(0, self.genes.shape[0]))
        for i in range(self.genes.shape[0]):
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
