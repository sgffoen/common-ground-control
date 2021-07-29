import tensorflow as tf

import os
import datetime
import shutil

from loader import get_img_path, create_dataset_dir
from training import ML


# RUN CODE #
BUFFER_SIZE = 400  # The facade training set consist of 400 images
BATCH_SIZE = 1  # The batch size of 1 produced better results for the U-Net in the original pix2pix experiment
environment = 'test'
iteration = 0
type = 'height2rgb'
data_num = int(BUFFER_SIZE / 2)


# create new directory for data set
new_dir_train = create_dataset_dir(env=environment, iter=1, data='train')
new_dir_test = create_dataset_dir(env=environment, iter=1, data='test')


# copy img into one folder
COLLECT = False
if COLLECT:
    for i in range(data_num):
        filepath = get_img_path(environment, i, type)
        shutil.copy(filepath, new_dir_train)

    for i in range(data_num, data_num*2):
        filepath = get_img_path(environment, i, type)
        shutil.copy(filepath, new_dir_test)


# create directory for save predicted img
save_path = "G:/Shared drives/2021_MAS/T3/Common Ground Control/01_data/01_training/data"
try:
    os.mkdir(save_path)
except FileExistsError:
    print("Directory ", save_path,  " already exists")

plot_path = save_path + "/plots"
try:
    os.mkdir(plot_path)
except FileExistsError:
    print("Directory ", save_path,  " already exists")


if not COLLECT:
    ml = ML()
    img_path = get_img_path(environment, iteration, type)
    inp, re = ml.load(img_path)

    # BUILD AN INPUT PIPELINE
    train_dataset = tf.data.Dataset.list_files(new_dir_train + '/*.png')
    train_dataset = train_dataset.map(ml.load_image_train,num_parallel_calls=tf.data.AUTOTUNE)
    train_dataset = train_dataset.shuffle(BUFFER_SIZE)
    train_dataset = train_dataset.batch(BATCH_SIZE)

    test_dataset = tf.data.Dataset.list_files(new_dir_test + '/*.png')
    test_dataset = test_dataset.map(ml.load_image_test)
    test_dataset = test_dataset.batch(BATCH_SIZE)

    # DECODER
    down_model = ml.downsample(3, 4)
    down_result = down_model(tf.expand_dims(inp, 0))
    print(down_result.shape)

    # ENCODER
    up_model = ml.upsample(3, 4)
    up_result = up_model(down_result)
    print(up_result.shape)

    # generator
    generator = ml.Generator()
    filepath = plot_path + "/generator_model.png"
    tf.keras.utils.plot_model(generator, to_file=filepath, show_shapes=True, dpi=64)
    # test generator
    gen_output = generator(inp[tf.newaxis, ...], training=False)

    # generator loss
    ml.loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=True)

    # discriminator
    discriminator = ml.Discriminator()
    filepath = plot_path + "/discriminator_model.png"
    tf.keras.utils.plot_model(ml.discriminator, to_file=filepath, show_shapes=True, dpi=64)

    # test discriminator
    disc_out = ml.discriminator([inp[tf.newaxis, ...], gen_output], training=False)

    # discriminator loss
    ml.generator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
    ml.discriminator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
    ml.checkpoint = tf.train.Checkpoint(generator_optimizer=ml.generator_optimizer,
                                        discriminator_optimizer=ml.discriminator_optimizer,
                                        generator=ml.generator,
                                        discriminator=ml.discriminator)

    for example_input, example_target in test_dataset.take(1):
        ml.generate_images(ml.generator, example_input, example_target)

    # training
    log_dir = save_path + "/logs/"

    ml.summary_writer = tf.summary.create_file_writer(
        log_dir + "fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))

    ml.fit(train_dataset, test_dataset, steps=40000)
