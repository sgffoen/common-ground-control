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
        self.genes = self.random_assign()

    def get_gene(self):
        arr = np.zeros((10))
        return arr

    def random_assign(self):
        arr = self.get_gene()
        for i in range(arr.shape[0]):
            arr[i] = r.randrange(0, 10)
        return arr

    def cal_zdiff(self, arr1, arr2):
        arr_diff = np.subtract(arr1, arr2)
        arr_abs_diff = np.absolute(arr_diff)
        return arr_abs_diff

    def get_minmax(self, arr):
        min = np.amin(arr)
        max = np.amax(arr)
        return min, max

    def fit(self, target_arr):
        # evaluation
        zdiff = self.cal_zdiff(self.genes, target_arr)
        zdiff_min, zdiffmax = self.get_minmax(zdiff)
        mean_zdiff = np.mean(zdiff)

        self.fitness = m.exp(-mean_zdiff)
        # exponential fitness
        # self.fitness = score ** 2
        # self.fitness = 2 ** score

    def crossover(self, dna2):
        child_dna = DNA()
        midpoint1 = int(r.randint(0, self.genes.shape[0]))
        midpoint2 = int(r.randint(0, self.genes.shape[1]))
        for i in range(self.genes.shape[0]):
            for j in range(self.genes.shape[1]):
                if i > midpoint1 and j > midpoint2:
                    child_dna.genes[i][j] = self.genes[i][j]
                else:
                    child_dna.genes[i][j] = dna2.genes[i][j]
        return child_dna

    def mutation(self, metation_rate):
        for i in range(self.genes.shape[0]):
            for j in range(self.genes.shape[1]):
                if r.random() < metation_rate:
                    self.genes[i][j] = r.randint(np.amin(self.genes),
                                                 np.amax(self.genes))


if __name__ == '__main__':
    pass
