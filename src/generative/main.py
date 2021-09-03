from argparser import parse_args
from learning import LearningData
from process import Processing
from pix2pix import ML
import tensorflow as tf
from tensorflow import keras
import datetime
import time
import json
import cv2
import os


__FOLDER__ = "G:/Shared drives/2021_MAS/T3/Common Ground Control/01_data/"


def get_iteration_dirs():
    iteration_dirs = os.listdir('G:/Shared drives/2021_MAS/T3/Common Ground Control/01_data/00_data_collection/01_production')
    iteration_dirs.remove('meta.json')
    iteration_dirs.sort()
    return iteration_dirs


def hello_cgc(env, lvl, img_type):
    print('\n\nHello cgc, this is a test.\n')
    print('\nenv: ', env)
    print('\nlvl: ', lvl)
    print('\nimg_type: ', img_type)
    print('\n\n..........fin, enjoy cgc\n')


def processing(env):
    start = time.time()
    print("starting processing mode")
    print("environment: {} \n".format(env))

    iteration_dirs = get_iteration_dirs()

    for i, id in enumerate(iteration_dirs):
        # 1. accessing data_collection path/img
        data = LearningData(iter=i, id=id, env=env)
        data.get_iter_dirs()
        data.get_fframe()

        # 2. do processing
        p = Processing()

        # h2h (height2height)
        # toolpath_on_height = p.custom_img_addition(data.height_fframe,
        #                                            data.toolpath_fframe)
        # h2h = p.horizontal_stack(toolpath_on_height,
        #                          data.height_fframe_after)
        # p.save_img(h2h, data.get_save_path('h2h'))

        # split channel
        o = np.zeros([256, 256, 3])
        b_hff, g_hff, r_hff = cv2.split(data.height_fframe)
        b_hffa, g_hffa, r_hffa = cv2.split(data.height_fframe_after)
        b_tp, g_tp, r_tp = cv2.split(data.toolpath_fframe)
        b_depth = get_pix_below_tp(r_tp, g_hff)
        b_depth_a = get_pix_below_tp(r_tp, g_hffa)

        # gb2gb
        before = cv2.merge([o, g_hff, r_tp])
        after = cv2.merge([o, g_hffa, o])
        img_stacked = np.hstack((before, after))
        p.save_img(img_stacked, data.get_save_path('gb2gb'))

        # seperate channel
        before = cv2.merge([b_depth, g_hff, r_tp])
        after = cv2.merge([o, g_hffa, o])
        img_stacked = np.hstack((before, after))
        p.save_img(img_stacked, data.get_save_path('thd2thd'))

        if i % 100 == 0:
            lap = (time.time()-start)/60
            print('\nprocessing id: {} / {}\nLAP-TIME: {}\n'.format(i, len(iteration_dirs), lap))

    print('\nTotal processing time: ', (time.time()-start)/60, ' min\n\n')


def datasetting(env, lvl):
    start = time.time()
    print("starting data setting mode")
    print("environment: {} \n".format(env))

    iteration_dirs = get_iteration_dirs()

    for i, id in enumerate(iteration_dirs):
        # 1. accessing data_collection path/img
        data = LearningData(iter=i, id=id, env=env)
        data.get_iter_dirs()
        data.get_fframe()
        if lvl == 'all':
            curve_lvl = 'all'
        else:
            curve_lvl = data.get_toolpath_level()

        # 2. create new dir for dataset
        data.create_dataset_dir(lvl=curve_lvl)

        # 3. copy imgs to new directory
        data.store_data()

        if i % 100 == 0:
            lap = (time.time()-start)/60
            print('\ndataset id: {} / {}\nLAP-TIME: {}\n'.format(i, len(iteration_dirs), lap))

    print('\nTotal data setting time: ', (time.time()-start)/60, ' min\n\n')


def learning(lvl, img_type):
    start = time.time()
    print("starting leaning mode")

    # 0. set environment / level / img_type
    ld = LearningData(lvl=lvl, img_type=img_type)

    # 1. get dir of dataset
    ld.get_dataset_dir()

    # 2. create dir for save process
    ld.create_learning_dir()

    # 3. call ML
    ml = ML(ld,
            BUFFER_SIZE=10,
            BATCH_SIZE=1,
            IMG_WIDTH=256,
            IMG_HEIGHT=256,
            OUTPUT_CHANNELS=3,
            LAMBDA=100,
            STEPS=150000)

    # get random image for test
    img_path = ld.get_random_img_path()
    inp, re = ml.load(img_path, save=True)

    # 4. build and input pipeline
    train_dataset = tf.data.Dataset.list_files(ld.train_dir + '/*.png')
    train_dataset = train_dataset.map(ml.load_image_train,
                                      num_parallel_calls=tf.data.AUTOTUNE)
    train_dataset = train_dataset.shuffle(ml.BUFFER_SIZE)
    train_dataset = train_dataset.batch(ml.BATCH_SIZE)

    test_dataset = tf.data.Dataset.list_files(ld.test_dir + '/*.png')
    test_dataset = test_dataset.map(ml.load_image_test)
    test_dataset = test_dataset.batch(ml.BATCH_SIZE)

    # 5. set generator / loss
    generator = ml.Generator()
    tf.keras.utils.plot_model(generator,
                              to_file=os.path.join(ld.plots_dir, "generator_model.png"),
                              show_shapes=True,
                              dpi=64)
    # generator test
    gen_output = generator(inp[tf.newaxis, ...], training=False)
    ml.save_fig(path=(ld.plots_dir + '/gen_test.png'), img=gen_output[0, ...])
    # generator loss
    ml.loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=True)

    # 6. set discriminator / loss
    discriminator = ml.Discriminator()
    tf.keras.utils.plot_model(discriminator,
                              to_file=os.path.join(ld.plots_dir, "discriminator_model.png"),
                              show_shapes=True,
                              dpi=64)
    # discriminator test
    disc_out = discriminator([inp[tf.newaxis, ...], gen_output],
                             training=False)
    ml.save_fig(path=(ld.plots_dir + '/disc_test.png'),
                img=disc_out[0, ..., -1],
                disc=True)
    # discriminator loss
    ml.generator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
    ml.discriminator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
    checkpoint = tf.train.Checkpoint(generator_optimizer=ml.generator_optimizer,
                                     discriminator_optimizer=ml.discriminator_optimizer,
                                     generator=generator,
                                     discriminator=discriminator)

    # 7. learning
    ml.summary_writer = tf.summary.create_file_writer(ld.log_dir
                                                      + "/"
                                                      + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    trained_generator = ml.fit(generator,
                               discriminator,
                               train_dataset,
                               test_dataset,
                               checkpoint,
                               ld.ckpt_dir,
                               ld.model_dir,
                               ld.plots_dir)

    # 8. save params
    ml.save_model(trained_generator, path=ld.model_dir)
    loaded_model = ml.load_model(ld.model_dir)
    fitness_time = (time.time()-start)/60
    ml.export_json(ld, fitness_time)

    # fin
    print('\nTotal learning time: ', fitness_time, ' min\n\n')


def main():
    run_mode, environment, level, img_type = parse_args()

    # gan
    if run_mode == 'hello':
        hello_cgc(environment, level, img_type)
    elif run_mode == 'process':
        processing(environment)
    elif run_mode == 'dataset':
        datasetting(environment, level)
    elif run_mode == 'learn':
        learning(level, img_type)
    # else
    else:
        print("Run mode is not identified")


if __name__ == "__main__":
    main()

