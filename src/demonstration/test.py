import tensorflow as tf
from tkinter.filedialog import askopenfilename, askdirectory
import cv2
import numpy as np
from matplotlib import pyplot as plt
import os


def browse_file():
    filename = askopenfilename()
    return filename


def browse_dir():
    dir_name = askdirectory()
    return dir_name


def load_image(show=False):
    filename = browse_file()
    loaded_img = cv2.imread(filename)

    if show:
        cv2.imshow("loaded_img", loaded_img)
        cv2.waitKey(0)

    return loaded_img


def save_fig(img_to_save, path):
    fname = path + '/test.png'
    cv2.imwrite(fname, img_to_save)


def cal_zdiff(arr1, arr2):
    arr_diff = np.subtract(arr1, arr2)
    arr_abs_diff = np.absolute(arr_diff)
    return arr_abs_diff


def get_minmax(arr):
    min = np.amin(arr)
    max = np.amax(arr)
    return min, max


def crop_image(img, path):
    # Cropping an image
    left_image = img[0:256, 0:256]
    right_image = img[0:256, 256:512]

    # Save the cropped image
    fname = path + '/' + 'left_img.png'
    cv2.imwrite(fname, left_image)
    fname = path + '/' + 'right_img.png'
    cv2.imwrite(fname, right_image)


def cal_MSE(img1, img2):
    # mean square error
    mse = (np.square(img1.astype(int)-img2.astype(int))).mean(axis=None)
    return mse


def load_model():
    dir_name = browse_dir()
    try:
        loaded_model = tf.keras.models.load_model(dir_name)
        print('model is loaded from {}\n'.format(dir_name))
        return loaded_model
    except FileNotFoundError:
        print('model, {}, does not exist\n'.format(dir_name))


def load(img_path, save=False):
    # Read and decode an image file to a uint8 tensor
    image = tf.io.read_file(img_path)
    image = tf.image.decode_png(image)

    # Split each image tensor into two tensors:
    # - one with a real building facade image
    # - one with an architecture label image
    w = tf.shape(image)[1]
    w = w // 2
    input_image = image[:, :w, :]
    real_image = image[:, w:, :]

    # Convert both images to float32 tensors
    input_image = tf.cast(input_image, tf.float32)
    real_image = tf.cast(real_image, tf.float32)

    return input_image, real_image


# Normalizing the images to [-1, 1]
def normalize(input_image, real_image):
    input_image = (input_image / 127.5) - 1
    real_image = (real_image / 127.5) - 1

    return input_image, real_image


def generate_images(model, test_input, tar, plot_dir, step=0):
    test_input = np.reshape(test_input, [1, 256, 256, 3])
    # tar = np.reshape(tar, [1, 256, 256, 3])

    prediction = model(test_input, training=True)
    plt.figure(figsize=(15, 15))

    display_list = [test_input[0], tar[0], prediction[0]]
    title = ['Input Image', 'Ground Truth', 'Predicted Image']

    for i in range(3):
        plt.subplot(1, 3, i+1)
        plt.title(title[i])
        # Getting the pixel values in the [0, 1] range to plot.
        plt.imshow(display_list[i] * 0.5 + 0.5)
        plt.axis('off')
    # plt.show()
    fname = os.path.join(plot_dir,
                         'predicted_img_{}.png'.format(step))
    plt.savefig(fname)


if __name__ == '__main__':
    # load 2 image to compare
    img1 = load_image()
    img2 = load_image()
    print(img1.shape, img2.shape)

    # calc z value difference
    arr_diff = cal_zdiff(img1, img2)
    print(arr_diff.shape)

    # differenr type of difference
    # mse = cal_MSE(img1, img2)
    # print(mse)

    # get min & max difference
    min, max = get_minmax(arr_diff)
    print(min, max)

    # save difference image
    save_path = browse_dir()
    save_fig(arr_diff, save_path)


    # loaded_model = load_model()
    # image_path = "G:/Shared drives/2021_MAS/T3/Common Ground Control/01_data/02_demo/00_test/00000_2021-08-05_h2h_training.png"
    # input_img, target_img = load(image_path)
    # input_img, target_img = normalize(input_img, target_img)
    # save_dir = "G:/Shared drives/2021_MAS/T3/Common Ground Control/01_data/02_demo/00_test"
    # # crop_image(test_input, save_dir)
    # generate_images(loaded_model, input_img, target_img, save_dir)
