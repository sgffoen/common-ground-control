# python libs
import random as r
import numpy as np

# package
from groundtruth.toolbox.helper import Facts
import groundtruth.toolbox.generative_utils as gu


class GA(object):
    def __init__(self, target_img, initial_height):
        self.facts = Facts().facts
        self.target = self.target_height(target_img)
        self.initial_height = initial_height


    def target_height(self, target_img):
        return gu.split_channel(target_img)[1].astype(np.float32)


    def get_weights_matrix(self):
        initial_error = np.abs(self.initial_height - self.target)
        return np.power(initial_error, 2)


    def get_max_error_matrix(self):
        initial_error = np.abs(self.initial_height - self.target)
        return np.amax(initial_error) * np.ones([256, 256])


    def get_fitness(self, prediction):
        # get fitness
        predicted_error = np.abs(prediction - self.target)
        prediction_fitness = np.abs(self.get_max_error_matrix() - predicted_error)

        fitness = prediction_fitness * self.get_weights_matrix()
        total_fitness = np.sum(fitness) ** 2

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
