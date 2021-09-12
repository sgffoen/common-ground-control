# python libs
import random as r

# package
from groundtruth.toolbox.helper import Facts


class GA(object):
    def __init__(self, target):
        self.facts = Facts().facts
        self.target = target


    def get_fitness(self, img1, img2):
        self.zdiff = self.cal_zdiff(img1, img2)
        return self.zdiff

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


    def mutation(self, metation_rate):
        for i in range(self.genotype.shape[0]):
            if r.random() < metation_rate:
                self.genotype[i][0] = r.randint(int(self.fframe_bounds[0][0]), int(self.fframe_bounds[0][1]))
                self.genotype[i][1] = r.randint(int(self.fframe_bounds[1][0]), int(self.fframe_bounds[1][1]))
                self.genotype[i][2] = r.randint(50, 100)


