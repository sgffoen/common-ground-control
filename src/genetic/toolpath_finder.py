from groundtruth.toolbox.genetic import ga
import groundtruth.toolbox.generative_utils as gu
from ga_helper import load_model

# built-in
import time
import matplotlib.pyplot as plt
import cv2
import random as r
import tensorflow as tf
import numpy as np


# def demo():

#     # GA parameters
#     start = time.time()
#     population_num = 200
#     population = []
#     mating_pool = []
#     mutation_rate = 0.01
#     generation_num = 200
#     plot = []

#     # initialize
#     for i in range(population_num):
#         a = Artist(2, height_fframe, target_img, loaded_model)
#         population.append(a)

#     for g in range(generation_num):
#         # selection
#         for p in population:
#             p.custom_img_addition()
#             p.fit()
#             # mating_pool
#             for j in range(int(p.fitness * 100)):
#                 mating_pool.append(p)
#         # reproduciton
#         for i in range(population_num):
#             # next gen
#             child_a = Artist(2, height_fframe, target_img, loaded_model)
#             # crossover
#             parent_a = r.sample(mating_pool, 1)[0]
#             parent_b = r.sample(mating_pool, 1)[0]
#             child_a.crossover_tween(parent_a, parent_b)
#             # mutation
#             # child_a.mutation(mutation_rate)
#             # fit
#             child_a.custom_img_addition()
#             child_a.fit()


#             population[i] = child_a

#             # plot
#             fitness = population[i].fitness
#             plot.append([population[i].fitness])

#         if g % 10 == 0:
#             print('\ngeneration: {} / {}'.format(g, generation_num))
#             print('LAPTIME   : ', (time.time()-start)/60, ' min')
#             print('FITNESS   : ', fitness, '\n')
#             id = str(g).zfill(3)
#             fname = save_dir_ga + '/' + str(id) + '_' + 'ga_toolpath.png'
#             cv.imwrite(fname, population[i].input_img)

#     # save image
#     for i,p in enumerate(population):
#         id = str(i).zfill(3)
#         tp_fname = save_dir + '/' + str(id) + '_' + "toolpath_prediction.png"
#         ex_fname = save_dir + '/' + str(id) + '_' + "excavate_prediction.png"
#         p.save_fig(tp_fname, ex_fname)

#     # check parameters
#     total_fitness = sum([population[i].fitness for i in range(len(population))])
#     average_fitness = total_fitness / len(population)
#     calc_time = time.time() - start

#     print("average fitness : {}".format(average_fitness))
#     print("total population: {}".format(population_num))
#     print("mutation rate   : {}".format(mutation_rate))
#     print("calculation time: {}sec".format(calc_time))

#     # save plot
#     fig, ax = plt.subplots()
#     mean = [p[0] for p in plot]
#     # z_min = [p[1] for p in plot]
#     # z_max = [p[2] for p in plot]
#     ax.plot(mean, color='green', label='fitness')
#     # ax.plot(z_min, color='blue', label='min')
#     # ax.plot(z_max, color='red', label='max')
#     ax.legend(loc='upper left')
#     plt.xlabel('generation')
#     plt.ylabel('fitness')
#     plot_fname = save_dir + '/' + 'plot_fitness.png'
#     plt.savefig(plot_fname)



def load_img(img_path):
    # Read and decode an image file to a uint8 tensor
    image = tf.io.read_file(img_path)
    image = tf.image.decode_png(image)

    # Split each image tensor into two tensors:
    # - one with a real building facade image
    # - one with an architecture label image
    w = tf.shape(image)[1]
    w = w // 2
    print('W: ',w)
    input_image = image[:, :w, :]
    print(input_image)
    real_image = image[:, w:, :]
    # Convert an image to float32 tensors
    input_image = tf.cast(input_image, tf.float32)

    #ground_truth = tf.cast(real_image, tf.float32)
    # Normalizing the images to [-1, 1]
    input_image = (input_image / 127.5) - 1
    input_tensor = np.reshape(input_image, [1, 256, 256, 3]).astype(np.float32)
    # ground truth
    ground_truth = cv2.imread(img_path)
    ground_truth = ground_truth[:, w:, :]
    # toolpath
    toolpath = cv2.imread(img_path)
    toolpath = toolpath[:, :w, :]

    return input_tensor, ground_truth, toolpath


def single_prediction_from_model(test_sample_path="G:/Shared drives/Ko-Simon MAS thesis (temporary)/01_data/01_gan/00_dataset/dataset_lvl_all/tgd2tgd_thick/test/00000_2021-08-05_tgd2tgd_training_thick.png"):
    img = cv2.imread(test_sample_path)
    input_tensor, ground_truth = gu.test_image2tensor(test_sample_path)
    #input_tensor, gt, toolp = load_img(test_sample_path)
    pred = gu.prediction_from_model(model=load_model(dir_name="C:/Users/simon/Documents/MAS DFAB/04_MAS_THESIS/06_backup/models/00010_2021-09-06/model"), input_tensor=input_tensor)
    img = gu.generate_img(pred)
    cv2.imshow('prediction', img)
    cv2.waitKey(0)


def single_prediction_from_api(test_sample_path="G:/Shared drives/Ko-Simon MAS thesis (temporary)/01_data/01_gan/00_dataset/dataset_lvl_all/tgd2tgd_thick/test/00000_2021-08-05_tgd2tgd_training_thick.png"):
    img = cv2.imread(test_sample_path)
    input_tensor, ground_truth = gu.test_image2tensor(test_sample_path)
    pred = gu.prediction_from_api(input_tensor=input_tensor)
    img = gu.generate_img(pred)
    cv2.imshow('prediction', img)
    cv2.waitKey(0)


if __name__ == '__main__':
    # demo()
    single_prediction_from_model()
