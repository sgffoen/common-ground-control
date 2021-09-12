# python libs
import tensorflow as tf


def load_model(dir_name="G:/Shared drives/Ko-Simon MAS thesis (temporary)/01_data/01_gan/01_models/00010_2021-09-06/model/1"):
    try:
        loaded_model = tf.keras.models.load_model(dir_name)
        print('model is loaded from {}\n'.format(dir_name))
        return loaded_model
    except FileNotFoundError:
        print('model, {}, does not exist\n'.format(dir_name))
