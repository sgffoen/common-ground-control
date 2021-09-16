# python libs
import random as r
import numpy as np
import cv2

# package
from groundtruth.toolbox.helper import Facts
import groundtruth.toolbox.generative_utils as gu


class GA(object):
    def __init__(self, target_img, initial_img, prediction_img):
        self.facts = Facts().facts
        self.target = self.get_height(target_img)
        self.initial_height = self.get_height(initial_img)
        self.prediction = self.get_height(prediction_img)

    def get_height(self, arr):
        arr1 =  gu.split_channel(arr)[1].astype(np.float32)
        cropped_arr = self.create_designspace(arr1)
        return cropped_arr

    def get_weights_matrix(self):
        initial_error = np.abs(self.initial_height - self.target)
        return np.power(initial_error, 2)

    def get_max_error_matrix_fframe(self):
        initial_error = np.abs(self.initial_height - self.target)
        return np.amax(initial_error) * np.ones([256, 256])

    def get_max_error_matrix_feature(self):
        initial_error = np.abs(self.initial_height - self.target)
        return np.amax(initial_error) * np.ones(self.target.shape)

    def create_designspace(self, arr):
        offset = 100  # 100mm
        return arr[offset:729-offset, offset:1135-offset]

    def get_fitness(self, fframe):
        # get fitness
        predicted_error = np.abs(self.prediction - self.target)
        if fframe:
            prediction_fitness = np.abs(self.get_max_error_matrix_fframe() - predicted_error)
        else:
            prediction_fitness = np.abs(self.get_max_error_matrix_feature() - predicted_error)

        fitness = prediction_fitness * self.get_weights_matrix()
        total_fitness = np.sum(fitness) ** 2

        return total_fitness

    def get_fitness_minimize(self):
        errors = np.abs(self.prediction - self.target)
        ones = np.ones(errors.shape, dtype=np.float32)
        weights = np.abs(self.initial_height - self.target)
        cond = (weights != 0)
        total_fitness = np.sum( errors * np.divide(ones, weights, out=np.zeros_like(ones), where=cond) ) ** 2

        return total_fitness

    def generate_img(self):

        input_tensor = gu.load_input_tensor(input_img)
        # generate_img
        pred = gu.prediction_from_model(model=self.model, input_tensor=input_tensor)
        # denormalized
        self.phenotype = gu.generate_img(pred)

    def fit(self):
        # generate phenotype
        # fitness function
        raw_fitness = self.get_fitness(self.target_img, self.phenotype)
        # put at least one item into pool
        if raw_fitness < 0.01:
            raw_fitness += 0.01
        self.fitness = raw_fitness

    def crossover(self, parent_a, parent_b):

        midpoint = int(r.randint(0, self.genotype.shape[0]-1))
        for i in range(self.genotype.shape[0]):
            if i < midpoint:
                self.genotype[i] = parent_a.genotype[i]
            else:
                self.genotype[i] = parent_b.genotype[i]

    def mutation(self, mutation_rate):
        for i in range(self.genotype.shape[0]):
            if r.random() < mutation_rate:
                self.genotype[i][0] = r.randint(int(self.fframe_bounds[0][0]), int(self.fframe_bounds[0][1]))
                self.genotype[i][1] = r.randint(int(self.fframe_bounds[1][0]), int(self.fframe_bounds[1][1]))
                self.genotype[i][2] = r.randint(50, 100)
