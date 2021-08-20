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
from toolbox import TrainingData
from toolbox import ScanData, HeightMap, PointCloud
import toolbox.ur_helper as ur


def demo():
    save_dir = "G:/Shared drives/2021_MAS/T3/Common Ground Control/01_data/02_demo/00_test"
    save_dir_ga = "G:/Shared drives/2021_MAS/T3/Common Ground Control/01_data/02_demo/00_test/ga"

    # scan and initialize
    # ur.scan_pose(scanning_time=5)
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
    filepath = "G:/Shared drives/2021_MAS/T3/Common Ground Control/01_data/02_demo/00_test/test_target.png"
    target_img = cv.imread(filepath)

    # get fframe
    helper = Helper()
    # height_fframe = helper.crop_feature(height_feature)
    filepath = "G:/Shared drives/2021_MAS/T3/Common Ground Control/01_data/02_demo/00_test/test_scan.png"
    height_fframe = cv.imread(filepath)
    # load model
    print("select the model you want to test.")
    loaded_model = helper.load_model()

    # GA parameters
    start = time.time()
    population_num = 2
    population = []
    mating_pool = []
    mutation_rate = 0.01
    generation_num = 10
    plot = []

    # initialize
    for i in range(population_num):
        a = Artist(2, height_fframe, target_img, loaded_model)
        population.append(a)

    for g in range(generation_num):
        # selection
        for p in population:
            p.custom_img_addition()
            p.fit()
            # mating_pool
            for j in range(int(p.fitness * 100)):
                mating_pool.append(p)
        # reproduciton
        for i in range(population_num):
            # next gen
            child_a = Artist(2, height_fframe, target_img, loaded_model)
            # crossover
            parent_a = r.sample(mating_pool, 1)[0]
            parent_b = r.sample(mating_pool, 1)[0]
            child_a.crossover(parent_a, parent_b)
            # mutation
            # child_a.mutation(mutation_rate)
            # fit
            child_a.custom_img_addition()
            child_a.fit()

            # # compete paraents
            # if parent_a.fitness > parent_b.fitness:
            #     better_parent = parent_a
            # else:
            #     better_parent = parent_b
            # # compete parent and child
            # if child_a.fitness > better_parent.fitness:
            #     population[i] = child_a
            population[i] = child_a

            # plot
            fitness = population[i].fitness
            plot.append([population[i].fitness])

        if g % 10 == 0:
            print('\ngeneration: {} / {}'.format(g, generation_num))
            print('LAPTIME   : ', (time.time()-start)/60, ' min')
            print('FITNESS   : ', fitness, '\n')
            id = str(g).zfill(3)
            fname = save_dir_ga + '/' + str(id) + '_' + 'ga_toolpath.png'
            cv.imwrite(fname, population[i].input_img)

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

    print("average fitness : {}".format(average_fitness))
    print("total population: {}".format(population_num))
    print("mutation rate   : {}".format(mutation_rate))
    print("calculation time: {}sec".format(calc_time))

    # save plot
    fig, ax = plt.subplots()
    mean = [p[0] for p in plot]
    # z_min = [p[1] for p in plot]
    # z_max = [p[2] for p in plot]
    ax.plot(mean, color='green', label='fitness')
    # ax.plot(z_min, color='blue', label='min')
    # ax.plot(z_max, color='red', label='max')
    ax.legend(loc='upper left')
    plt.xlabel('generation')
    plt.ylabel('fitness')
    plot_fname = save_dir + '/' + 'plot_fitness.png'
    plt.savefig(plot_fname)


def single_prediction():
    h = Helper()
    loaded_model = h.load_model()
    path = "G:/Shared drives/2021_MAS/T3/Common Ground Control/01_data/02_demo/00_test/left_img.png"
    input_img = h.load_img(path)
    h.generate_img(loaded_model, input_img)
    cv.imshow('single_prediction', h.phenotype)
    cv.waitKey(0)

if __name__ == '__main__':
    demo()
    # single_prediction()
