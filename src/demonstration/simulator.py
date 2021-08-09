from tkinter.filedialog import askopenfilename
import cv2
import numpy as np


def image_loader(show=False):
    filename = askopenfilename()
    loaded_img = cv2.imread(filename)

    if show:
        cv2.imshow("loaded_img", loaded_img)
        cv2.waitKey(0)

    return loaded_img


def cal_MSE(img1, img2):
    mse = (np.square(img1.astype(int)-img2.astype(int))).mean(axis=None)
    return mse


if __name__ == '__main__':
    img1 = image_loader()
    img2 = image_loader()
    print(img1.shape, img2.shape)

    mse = cal_MSE(img1, img2)
    print(mse)
