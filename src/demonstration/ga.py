import numpy as np
import random as r


class DNA(object):
    def __init__(self):
        self.genes = self.random_assign()

    def get_gene(self):
        arr = np.zeros((18))
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


def crossover(dna1, dna2):
    child_dna = DNA()
    midpoint = int(r.randint(0, dna1.genes.shape[0]))
    for i in range(dna1.genes.shape[0]):
        if i > midpoint:
            child_dna.genes[i] = dna1.genes[i]
        else:
            child_dna.genes[i] = dna2.genes[i]
    return child_dna


def mutation(dna, metation_rate):
    for i in range(dna.genes.shape[0]):
        if r.random() < metation_rate:
            dna.genes[i] = r.randint(0, 10)
    return dna


if __name__ == '__main__':
    population_num = 100
    population = []
    mating_pool = []
    mutation_rate = 0.01

    target_arr = np.zeros((18))
    for i in range(target_arr.shape[0]):
        target_arr[i] = r.randrange(0, 10)

    # initialize
    for i in range(population_num):
        population.append(DNA())

    # selection
    for i in range(len(population)):
        population[i].fit(target_arr)

    for i in range(len(population)):
        for j in range(int(population[i].fitness * 100)):
            mating_pool.append(population[i])

    # reprodiction
    for i in range(len(population)):
        a = int(r.randint(0, len(mating_pool)-1))
        b = int(r.randint(0, len(mating_pool)-1))

        parent_a = mating_pool[a]
        parent_b = mating_pool[b]

        child_dna = crossover(parent_a, parent_b)
        child_dna = mutation(child_dna, mutation_rate)
        population[i] = child_dna

    '''until 9.8'''
