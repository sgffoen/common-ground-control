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


def crop_image(img, path):
    # Cropping an image
    cropped_image = img[0:256, 0:256]

    # Save the cropped image
    fname = path + '/' + 'cropped_img.png'
    cv2.imwrite(fname, cropped_image)


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


def generate_images(model, test_input, save_dir, show=False):

    input_image = np.reshape(test_input, [1, 256, 256, 3])  # 1 is BATCH_SIZE

    prediction = model(input_image, training=False)
    plt.figure(figsize=(15, 15))

    print(input_image[0].shape, prediction[0].shape)
    display_list = [input_image[0], prediction[0]]
    title = ['Input Image', 'Predicted Image']

    for i in range(2):
        plt.subplot(1, 3, i+1)
        plt.title(title[i])
        # Getting the pixel values in the [0, 1] range to plot.
        plt.imshow(display_list[i] * 0.5 + 0.5)
        plt.axis('off')

    if show:
        plt.show()
    fname = os.path.join(save_dir, 'predicted_img.png')
    plt.savefig(fname)


if __name__ == '__main__':
    # img1 = load_image()
    # img2 = load_image()
    # print(img1.shape, img2.shape)

    # mse = cal_MSE(img1, img2)
    # print(mse)

    loaded_model = load_model()
    test_input = load_image()
    save_dir = "G:/Shared drives/2021_MAS/T3/Common Ground Control/01_data/02_demo/00_test"
    # crop_image(test_input, save_dir)
    generate_images(loaded_model, test_input, save_dir)
