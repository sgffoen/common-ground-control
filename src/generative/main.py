from argparser import parse_args
from learning import LearningData
from process import Processing
from pix2pix import ML
import tensorflow as tf
import datetime
import time
import json
import os


__FOLDER__ = "G:/Shared drives/2021_MAS/T3/Common Ground Control/01_data/"


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

    max_id = meta_data(env)

    for i in range(max_id):
        # 1. accessing data_collection path/img
        data = LearningData(i, env)
        data.get_iter_dirs(i)
        data.get_fframe()

        # 2. do processing
        p = Processing()

        # c2c (rgb2rgb)
        toolpath_on_rgb = p.img_overlay(data.rgb_fframe, data.toolpath_fframe)
        c2c = p.horizontal_stack(toolpath_on_rgb, data.rgb_fframe_after)
        p.save_img(c2c, data.get_save_path('c2c'))

        # h2h (height2height)
        toolpath_on_height = p.img_overlay(data.height_fframe, data.toolpath_fframe)
        h2h = p.horizontal_stack(toolpath_on_height, data.height_fframe_after)
        p.save_img(h2h, data.get_save_path('h2h'))

        # h2c (height2rgb)
        h2c = p.horizontal_stack(toolpath_on_height, data.rgb_fframe_after)
        p.save_img(h2c, data.get_save_path('h2c'))

        # c2h (rgb2height)
        c2h = p.horizontal_stack(toolpath_on_rgb, data.height_fframe_after)
        p.save_img(c2h, data.get_save_path('c2h'))

        # split channel
        height_fframe_split = p.chennel_edit(data.height_fframe)
        height_fframe_after_split = p.chennel_edit(data.height_fframe_after)

        # g2b
        toolpath_on_green = p.img_overlay(height_fframe_split[1], data.toolpath_fframe)
        g2b = p.horizontal_stack(toolpath_on_green, height_fframe_after_split[2])
        p.save_img(g2b, data.get_save_path('g2b'))

        # g2b
        toolpath_on_blue = p.img_overlay(height_fframe_split[2], data.toolpath_fframe)
        b2g = p.horizontal_stack(toolpath_on_blue, height_fframe_after_split[1])
        p.save_img(b2g, data.get_save_path('b2g'))

        if i % 50 == 0:
            print('\nprocessing id: {} / {}\n'.format(i, max_id))

    print('\nTotal processing time: ', (time.time()-start)/60, ' min\n\n')


def datasetting(env, lvl):
    start = time.time()
    print("starting data setting mode")
    print("environment: {} \n".format(env))

    max_id = meta_data(env)

    for i in range(max_id):
        # 1. accessing data_collection path/img/json
        data = LearningData(i, env)
        data.get_iter_dirs(i)
        data.get_fframe()
        if lvl == 'all':
            curve_lvl = 'all'
        else:
            curve_lvl = data.get_toolpath_level()

        # 2. create new dir for dataset
        data.create_dataset_dir(lvl=curve_lvl)

        # 3. copy imgs to new directory
        data.store_data()

        if i % 50 == 0:
            print('\ndata setting id: {} / {}\n'.format(i, max_id))

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
    ld.create_json()

    # 3. call ML
    ml = ML(BUFFER_SIZE=10,
            BATCH_SIZE=1,)
    img_path = ld.get_random_img_path()
    print('\ntest image: ', img_path, '\n')
    inp, re = ml.load(img_path)
    ml.save_fig(path=(ld.plots_dir + '/input.png'), img=inp/255.)
    ml.save_fig(path=(ld.plots_dir + '/real.png'), img=re/255.)
    # inspect some of preprocessed output
    ml.preprocess_sample(inp, re, path=(ld.plots_dir+'/preprocess.png'))

    # 4. build and input pipeline
    train_dataset = tf.data.Dataset.list_files(ld.train_dir + '/*.png')
    train_dataset = train_dataset.map(ml.load_image_train, num_parallel_calls=tf.data.AUTOTUNE)
    train_dataset = train_dataset.shuffle(ml.BUFFER_SIZE)
    train_dataset = train_dataset.batch(ml.BATCH_SIZE)

    test_dataset = tf.data.Dataset.list_files(ld.test_dir + '/*.png')
    test_dataset = test_dataset.map(ml.load_image_test)
    test_dataset = test_dataset.batch(ml.BATCH_SIZE)

    # 5. set decoder
    down_model = ml.downsample(3, 4)
    down_result = down_model(tf.expand_dims(inp, 0))
    print('\ndecoder shape', down_result.shape, '\n')

    # 6. set encoder
    up_model = ml.upsample(3, 4)
    up_result = up_model(down_result)
    print('\nencoder shape', up_result.shape, '\n')

    # 7. set generator / loss
    generator = ml.Generator()
    filepath = os.path.join(ld.plots_dir, "generator_model.png")
    tf.keras.utils.plot_model(generator, to_file=filepath, show_shapes=True, dpi=64)
    # generator test
    gen_output = generator(inp[tf.newaxis, ...], training=False)
    ml.save_fig(path=(ld.plots_dir + '/gen_test.png'), img=gen_output[0, ...])
    # generator loss
    ml.loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=True)

    # 8. set discriminator / loss
    discriminator = ml.Discriminator()
    filepath = os.path.join(ld.plots_dir, "discriminator_model.png")
    tf.keras.utils.plot_model(ml.discriminator, to_file=filepath, show_shapes=True, dpi=64)
    # discriminator test
    disc_out = discriminator([inp[tf.newaxis, ...], gen_output], training=False)
    ml.save_fig(path=(ld.plots_dir + '/disc_test.png'), img=disc_out[0, ..., -1], disc=True)
    # discriminator loss
    ml.generator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
    ml.discriminator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
    ml.checkpoint = tf.train.Checkpoint(generator_optimizer=ml.generator_optimizer,
                                        discriminator_optimizer=ml.discriminator_optimizer,
                                        generator=ml.generator,
                                        discriminator=ml.discriminator)

    # 9. run generator
    for example_input, example_target in test_dataset.take(1):
        ml.generate_images(ml.generator, example_input, example_target, ld.plots_dir)

    # 10. learning
    ml.summary_writer = tf.summary.create_file_writer(
        ld.fit_dir + "/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    ml.fit(train_dataset, test_dataset, ld.ckpt_dir, ld.model_dir, ld.plots_dir, steps=1001)

    # 11. save params
    ld.export_json(ml)

    # fin
    print('\nTotal learning time: ', (time.time()-start)/60, ' min\n\n')


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


def meta_data(env, new_max_id=None, time=None, tp_level=None):
    filepath = os.path.join(__FOLDER__, "00_data_collection", ("00_test" if env=='test' else "01_production"), "meta.json")
    with open(filepath, 'r') as f:
        data = json.load(f)

    max_scan_id = data['max_scan_id']

    if new_max_id is not None:
        data["max_scan_id"] = new_max_id
        # export and overwrite json
        with open(filepath, 'w') as o:
            json.dump(data, o, indent=4)

    if time is not None:
        training_time = float(data["total_training_time"]) + time
        data["total_training_time"] = training_time
        # export and overwrite json
        with open(filepath, 'w') as o:
            json.dump(data, o, indent=4)

    if tp_level is not None:
        num = data["toolpath_lvls"][tp_level]
        data["toolpath_lvls"][tp_level] = int(num+1)
        # export and overwrite json
        with open(filepath, 'w') as o:
            json.dump(data, o, indent=4)

    return max_scan_id


if __name__ == "__main__":
    main()
