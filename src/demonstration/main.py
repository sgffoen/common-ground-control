# built-in
import sys
import time
import cv2 as cv
import random as r
import numpy as np
import tensorflow as tf
from tkinter.filedialog import askopenfilename, askdirectory

# package
from ga import DNA
from artist import Artist
from helper import Helper
# package from parent folders
sys.path.insert(1, 'C:/Users/trtku/OneDrive/Data/03_MAS/17_common_ground_control/01_git/common-ground-control/src/data_collection')
# import UR as ur
# from data import TrainingData
# from scanning import ScanData, HeightMap, PointCloud
sys.path.insert(2, 'C:/Users/trtku/OneDrive/Data/03_MAS/17_common_ground_control/01_git/common-ground-control/src/generative')
from process import Processing


def demo():
    save_dir = "G:/Shared drives/2021_MAS/T3/Common Ground Control/01_data/02_demo/00_test"

    # # scan and get the current state of the sandbox
    # ur.ur_helper.scan_pose(scanning_time=5)
    # data = TrainingData(iteration=0, environment='test')
    # scan = ScanData()
    # pcl_obj = PointCloud(scan)
    # heightmap = HeightMap(scan)

    # # store data
    # data.scan_data = scan
    # data.pointcloud = pcl_obj
    # data.heightmap = heightmap
    # data.store_data()
    # print('{}: data is collected and stored'.format(data.identifier))
    # height_feature = None

    # set target img
    filepath = "G:/Shared drives/2021_MAS/T3/Common Ground Control/01_data/00_data_collection/01_production/00000_2021-08-05/01_processed/00000_2021-08-05_height_featureframe_after.png"
    target_img = cv.imread(filepath)

    # get fframe
    helper = Helper()
    filepath = "G:/Shared drives/2021_MAS/T3/Common Ground Control/01_data/00_data_collection/01_production/00000_2021-08-05/01_processed/00000_2021-08-05_height_feature.png"
    temp_feature = cv.imread(filepath)
    height_fframe = helper.crop_feature(temp_feature)
    p = Processing()
    # load model
    loaded_model = helper.load_model()

    # GA parameters
    start = time.time()
    population_num = 2
    population = []
    mating_pool = []
    mutation_rate = 0.01
    generation_num = 100

    # initialize
    for i in range(population_num):
        # set population
        a = Artist(5, helper.fframe_bounds, height_fframe, loaded_model, target_img)
        population.append(a)

    for g in range(generation_num):
        # selection
        for i in range(population_num):
            population[i].dna.fit(target_img)
            # mating_pool
            for j in range(int(population[i].dna.fitness * 100)):
                mating_pool.append(population[i])

        # reproduciton
        for i in range(population_num):
            # pick up two parents
            a = int(r.randint(0, len(mating_pool)-1))
            b = int(r.randint(0, len(mating_pool)-1))
            parent_a = mating_pool[a]
            parent_b = mating_pool[b]
            # next gen
            child_a = Artist(5, helper.fframe_bounds, height_fframe, loaded_model, target_img)
            # crossover
            child_a = parent_a.dna.crossover(child_a, parent_b)
            # mutation
            child_a.dna.mutation(mutation_rate)
            # metabolize
            population[i] = child_a
            population[i].dna.fit(target_img)
            # check fitness
            if population[i].dna.fitness == 1.0:
                final_generation = g
                break
            else:
                final_generation = g

    # save image
    for i,p in enumerate(population):
        id = str(i).zfill(3)
        fname = save_dir + '/' + str(id) + '_' + "toolpath_prediction.png"
        p.dna.save_fig(fname)

    # evaluation
    total_fitness = sum([population[i].fitness for i in range(len(population))])
    average_fitness = total_fitness / len(population)
    calc_time = time.time() - start

    print("total generation: {}".format(final_generation))
    print("average fitness : {}".format(average_fitness))
    print("total population: {}".format(population_num))
    print("mutation rate   : {}".format(mutation_rate))
    print("calculation time: {}sec".format(calc_time))


if __name__ == '__main__':
    demo()
