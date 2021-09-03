import numpy as np
import tensorflow as tf
import cv2


def g2gray(g):
    arr = cv2.merge([g, g, g])
    return arr


def split_channel(arr):
    b, g, r = cv2.split(arr)
    return b, g, r


def load_model():
    dir_name = "G:/Shared drives/2021_MAS/T3/Common Ground Control/01_data/01_gan/01_models/00000_2021-08-17/model"
    try:
        loaded_model = tf.keras.models.load_model(dir_name)
        print('model is loaded from {}\n'.format(dir_name))
        return loaded_model
    except FileNotFoundError:
        print('model, {}, does not exist\n'.format(dir_name))


def denormalize(img_to_denormalize):
    img_to_denormalize = np.add(img_to_denormalize, 1)
    img_to_denormalize = np.multiply(img_to_denormalize, 127.5)
    return img_to_denormalize


def encode(img_to_encode):
    img_to_encode = img_to_encode.astype(np.uint8)
    return img_to_encode


def generate_img(model, input_tensor):
    """
    input: loaded model , tensor
    return: predict image
    """
    prediction = model(input_tensor, training=True)

    output_img = prediction[0].numpy()
    img_denormalized = denormalize(output_img)
    img_encoded = encode(img_denormalized)
    return img_encoded

def load_input_tensor(img_path):
    input_img = tf.io.read_file(img_path)
    input_img = tf.image.decode_png(input_img)
    input_img = tf.cast(input_img, tf.float32)
    input_img = (input_img / 127.5) - 1
    input_tensor = np.reshape(input_img, [1, 256, 256, 3])
    return input_tensor
