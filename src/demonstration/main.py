# built-in
import sys

# package
from ga import DNA
# package from parent folders
sys.path.insert(0, 'C:/Users/trtku/OneDrive/Data/03_MAS/17_common_ground_control/01_git/common-ground-control/src/data_collection')
import UR as ur
from data import TrainingData
from scanning import ScanData, HeightMap, PointCloud


def demo():
    pass

if __name__ == '__main__':
    pass
"""

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
"""
