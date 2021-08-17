# built-in
import sys
import time
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import cv2 as cv
import random as r
from tkinter.filedialog import askopenfilename, askdirectory

# package
from artist import Artist
from helper import Helper
# package from parent folders
# sys.path.insert(1, 'C:/Users/trtku/OneDrive/Data/03_MAS/17_common_ground_control/01_git/common-ground-control/src/data_collection')
# import UR as ur
# from data import TrainingData
# from scanning import ScanData, HeightMap, PointCloud


def demo():
    save_dir = "G:/Shared drives/2021_MAS/T3/Common Ground Control/01_data/02_demo/00_test"

    # # scan and initialize
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
    # height_fframe = helper.crop_feature(height_feature)
    filepath = "G:/Shared drives/2021_MAS/T3/Common Ground Control/01_data/00_data_collection/01_production/00000_2021-08-05/01_processed/00000_2021-08-05_height_feature.png"
    temp_feature = cv.imread(filepath)
    height_fframe = helper.crop_feature(temp_feature)
    # load model
    print("select the model you want to test.")
    loaded_model = helper.load_model()

    # GA parameters
    start = time.time()
    population_num = 2
    population = []
    mating_pool = []
    mutation_rate = 0.01
    generation_num = 1000
    plot_fitness = []

    # initialize
    for i in range(population_num):
        a = Artist(5, height_fframe, target_img, loaded_model)
        population.append(a)

    for g in range(generation_num):
        # selection
        for p in population:
            p.fit()
            # mating_pool
            for j in range(int(p.fitness * 100)):
                mating_pool.append(p)

        # reproduciton
        for i in range(population_num):
            # next gen
            child_a = Artist(5, height_fframe, target_img, loaded_model)
            # crossover
            if population_num == 2:
                child_a.crossover(mating_pool[0], mating_pool[1])
            else:
                child_a.crossover(r.sample(mating_pool, 1)[0],
                                r.sample(mating_pool, 1)[0])
            # mutation
            child_a.mutation(mutation_rate)
            # fit
            child_a.fit()
            population[i] = child_a
            # plot
            fitness = population[i].fitness
            plot_fitness.append(fitness)
            # evaluate
            if fitness > 0.9:
                final_generation = g
                break
            else:
                final_generation = g

        if g % 10 == 0:
            print('\ngeneration: {} / {}'.format(g, generation_num))
            print('LAPTIME   : ', (time.time()-start)/60, ' min\n')

    # save image
    for i,p in enumerate(population):
        id = str(i).zfill(3)
        tp_fname = save_dir + '/' + str(id) + '_' + "toolpath_prediction.png"
        ex_fname = save_dir + '/' + str(id) + '_' + "excavate_prediction.png"
        p.save_fig(tp_fname, ex_fname)

    # check parameters
    total_fitness = sum([population[i].fitness for i in range(len(population))])
    average_fitness = total_fitness / len(population)
    calc_time = time.time() - start

    print("total generation: {}".format(final_generation))
    print("average fitness : {}".format(average_fitness))
    print("total population: {}".format(population_num))
    print("mutation rate   : {}".format(mutation_rate))
    print("calculation time: {}sec".format(calc_time))

    # save plot
    id = str(0).zfill(3)
    fname = save_dir + '/' + str(id) + '_' + "fitness.png"
    plt.plot(plot_fitness, 'bo')
    plt.ylabel('fitness')
    plt.savefig(fname)


if __name__ == '__main__':
    demo()
