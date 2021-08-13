import numpy as np
import random as r
import math as m
import time


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
    start = time.time()
    population_num = 100
    population = []
    mating_pool = []
    mutation_rate = 0.01
    generation_num = 100

    target_arr = np.zeros((10))
    for i in range(target_arr.shape[0]):
        target_arr[i] = r.randrange(0, 10)

    # initialize
    for i in range(population_num):
        population.append(DNA())

    for g in range(generation_num):
        # selection
        for i in range(len(population)):
            population[i].fit(target_arr)
            # mating pool
            for j in range(int(population[i].fitness * 100)):
                mating_pool.append(population[i])

        # reprodiction
        for i in range(len(population)):
            # pick up two parents
            a = int(r.randint(0, len(mating_pool)-1))
            b = int(r.randint(0, len(mating_pool)-1))
            parent_a = mating_pool[a]
            parent_b = mating_pool[b]
            # crossover
            child_dna = parent_a.crossover(parent_b)
            # mutation
            child_dna.mutation(mutation_rate)
            population[i] = child_dna
            population[i].fit(target_arr)

            # check fitness
            if population[i].fitness == 1.0:
                final_generation = g
                break
            else:
                final_generation = g

    # evaluation
    print("\n{}".format(target_arr))
    print(population[i].genes)

    total_fitness = sum([population[i].fitness for i in range(len(population))])
    average_fitness = total_fitness / len(population)
    calc_time = time.time() - start

    print("total generation: {}".format(final_generation))
    print("average fitness : {}".format(average_fitness))
    print("total population: {}".format(population_num))
    print("mutation rate   : {}".format(mutation_rate))
    print("calculation time: {}sec".format(calc_time))

    '''until 9.8'''
