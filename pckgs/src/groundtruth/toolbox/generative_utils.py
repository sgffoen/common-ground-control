import numpy as np
import tensorflow as tf
import cv2
import compas.utilities as cu


__RANGEPIXEL__ = 200
__BOTTOMRANGE__ = 60


def g2gray(g):
    arr = cv2.merge([g, g, g])
    return arr


def split_channel(arr):
    b, g, r = cv2.split(arr)
    return b, g, r


def load_model(dir_name="G:/Shared drives/2021_MAS/T3/Common Ground Control/01_data/01_gan/01_models/00004_2021-08-18/model"):
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


def get_pix_below_tp(r_tp, g_hff):
    zeros = np.zeros([256, 256])
    cond1 = (r_tp > 0)
    cond2 = (r_tp < g_hff)
    depth = np.where(cond1, g_hff, zeros)
    depth = np.where(cond2, depth, zeros)

    return depth.astype(np.uint8)


def get_mean_fframe(g):
    return round(np.mean(g))


def get_min_fframe(g):
    return np.min(g)


def get_max_fframe(g):
    return np.max(g)


def get_bounds(g):
    min = get_min_fframe(g)
    max = get_max_fframe(g)
    return max.astype('float') - min.astype('float')


def get_remap_range(g):
    mean = get_mean_fframe(g)
    bottom_bound = mean - __RANGEPIXEL__/2
    top_bound = mean + __RANGEPIXEL__/2
    return bottom_bound, top_bound


def remap_fframe(g, bottom_bound, top_bound):
    # remap zero to the bottom of range
    zeros = np.zeros([256, 256], dtype=float)
    bottoms = bottom_bound * np.ones([256, 256], dtype=float)
    cond = (g == zeros)
    g = np.where(cond, bottoms, g)
    remapped_list = cu.remap_values(g,
                                    original_min=bottom_bound,
                                    original_max=top_bound,
                                    target_min=0,
                                    target_max=255)
    remapped_g = np.reshape(remapped_list, [256, 256])
    round_g = np.round(remapped_g)
    return round_g.astype(np.uint8)


def inverse_remap_img(img, bottom_bound, top_bound):
    """
    img = 3d array
    """
    # remap
    remapped_list = cu.remap_values(img,
                                    original_min=0,
                                    original_max=255,
                                    target_min=bottom_bound,
                                    target_max=top_bound)
    remapped_img = np.reshape(remapped_list, [256, 256, 3])
    remapped_img = cv2.merge([remapped_img[:,:,1], remapped_img[:,:,1], remapped_img[:,:,1]])
    round_img = np.round(remapped_img)
    return round_img.astype(np.uint8)
