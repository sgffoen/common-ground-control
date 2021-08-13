# built-in
import sys
import time
import cv2 as cv
import random as r
from tkinter.filedialog import askopenfilename, askdirectory

# package
from ga import DNA
from artist import Artist
from helper import Helper
# package from parent folders
sys.path.insert(1, 'C:/Users/trtku/OneDrive/Data/03_MAS/17_common_ground_control/01_git/common-ground-control/src/data_collection')
import UR as ur
from data import TrainingData
from scanning import ScanData, HeightMap, PointCloud
sys.path.insert(2, 'C:/Users/trtku/OneDrive/Data/03_MAS/17_common_ground_control/01_git/common-ground-control/src/generative')
from process import Processing


def demo():
    save_dir = "G:/Shared drives/2021_MAS/T3/Common Ground Control/01_data/02_demo/00_test"

    # scan and initialize
    ur.ur_helper.scan_pose(scanning_time=5)
    data = TrainingData(iteration=0, environment='test')
    scan = ScanData()
    pcl_obj = PointCloud(scan)
    heightmap = HeightMap(scan)

    # store data
    data.scan_data = scan
    data.pointcloud = pcl_obj
    data.heightmap = heightmap
    data.store_data()
    print('{}: data is collected and stored'.format(data.identifier))
    height_feature = None

    # set target img
    filepath = "G:/Shared drives/2021_MAS/T3/Common Ground Control/01_data/00_data_collection/01_production/00000_2021-08-05/01_processed/00000_2021-08-05_height_featureframe_after.png"
    target_img = cv.imread(filepath)

    # get fframe
    helper = Helper()
    height_fframe = helper.crop_feature(height_feature)
    p = Processing()
    # load model
    loaded_model = helper.load_model()

    # evaluation
    # zdiff = helper.cal_zdiff(input_img, target_img)
    # zdiff_min, zdiffmax = helper.get_minmax(zdiff)

    # GA parameters
    start = time.time()
    population_num = 100
    population = []
    mating_pool = []
    mutation_rate = 0.01
    generation_num = 1000

    # initialize
    for i in range(population_num):
        # draw toolpath
        a = Artist(5, helper.fframe_bounds)
        toolpath_fframe = helper.crop_feature(a.toolpath_feature)
        # overlay image
        input_img = p.simple_img_addition(height_fframe, toolpath_fframe)
        # prediction
        input_img = helper.decode(input_img, target_img)
        input_img = helper.normalize(input_img, target_img)
        generated_img = helper.generate_images(loaded_model, input_img, target_img, save_dir)
        # initialize DNA
        population.append(a.DNA(generated_img))

    for g in range(generation_num):
        # selection
        for i in range(population_num):
            population[i].fit(target_img)
            # mating_pool
            for j in range(int(population[i].fitness * 100)):
                mating_pool.append(population[i])

        # reproduciton
        for i in range(population_num):
            # pick up two parents
            a = int(r.randint(0, len(mating_pool)-1))
            b = int(r.randint(0, len(mating_pool)-1))
            parent_a = mating_pool[a]
            parent_b = mating_pool[b]
            # crossover
            child_a = Artist(5, helper.fframe_bounds)
            child_dna = child_a.DNA(generated_img)
            child_dna = parent_a.crossover(child_dna, parent_b)
            # mutation
            child_dna.mutation(mutation_rate)
            population[i] = child_dna
            population[i].fit(target_img)
            # check fitness
            if population[i].fitness == 1.0:
                final_generation = g
                break
            else:
                final_generation = g

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
