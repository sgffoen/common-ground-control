import numpy as np
import cv2
import os


class Processing():
    def __init__(self):
        pass

    def channel_edit(self, img_to_edit):
        # split img
        b, g, r = cv2.split(img_to_edit)
        zeros = np.zeros(img_to_edit.shape[:2], dtype="uint8")
        # merge img
        r_img = cv2.merge([zeros, zeros, r])
        g_img = cv2.merge([zeros, g, zeros])
        b_img = cv2.merge([b, zeros, zeros])
        bg_img = cv2.merge([b, g, zeros])
        return [r_img, g_img, b_img, bg_img]

    def custom_img_addition(self, height_img, toolpath_img):
        arr = np.zeros([256, 256, 3])
        for i in range(256):
            for j in range(256):
                if toolpath_img[i][j][2] > 0:
                    arr[i][j] = toolpath_img[i][j]
                else:
                    arr[i][j] = height_img[i][j]
        return arr

    def simple_img_addition(self, img1, img2):
        arr_add = np.add(img1, img2)
        return arr_add

    def horizontal_stack(self, img_left, img_right):
        img_h_stack = np.hstack((img_left, img_right))
        return img_h_stack

    def save_img(self, img_to_save, filepath):
        cv2.imwrite(filepath, img_to_save)

    def white2black(self, img):
        for i in range(256):
            for j in range(256):
                if img[i][j][2] < 255:
                    img[i][j][0] = 0
                    img[i][j][1] = 0
                else:
                    img[i][j][0] = 0
                    img[i][j][1] = 0
                    img[i][j][2] = 0
        return img

    def data_augmentation(self):
        """
        takes the original training samples and augments new samples by moving the all height pixels up or down
        """
        pass


if __name__ == "__main__":
    pass
