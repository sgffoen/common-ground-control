import numpy as np
import cv2
import os


class Processing():
    def __init__(self):
        pass

    def chennel_edit(self, img_to_edit):
        # split img
        b, g, r = cv2.split(img_to_edit)
        zeros = np.zeros(img_to_edit.shape[:2], dtype="uint8")
        # merge img
        r_img = cv2.merge([zeros, zeros, r])
        g_img = cv2.merge([zeros, g, zeros])
        b_img = cv2.merge([b, zeros, zeros])
        return [r_img, g_img, b_img]

    def img_overlay(self, base_img, mask_img):
        gray_img = cv2.cvtColor(mask_img, cv2.COLOR_BGR2GRAY)
        ret, mask = cv2.threshold(gray_img, 200, 255, cv2.THRESH_BINARY_INV)
        mask_inv = cv2.bitwise_not(mask)

        base = cv2.bitwise_and(base_img, base_img, mask=mask_inv)
        mask = cv2.bitwise_and(mask_img, mask_img, mask=mask)

        img_overlay = cv2.add(base, mask)
        return img_overlay

    def horizontal_stack(self, img_left, img_right):
        img_h_stack = np.hstack((img_left, img_right))
        return img_h_stack

    def save_img(self, img_to_save, filepath):
        if os.path.isfile(filepath):
            pass
            # print('file: {} already exists'.format(filepath))
        else:
            cv2.imwrite(filepath, img_to_save)
            # print('Save PNG image in: {}'.format(filepath))


if __name__ == "__main__":
    pass
