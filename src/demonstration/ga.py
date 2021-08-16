import compas.geometry as cg
import compas.utilities as cu
import numpy as np
import random as r


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
