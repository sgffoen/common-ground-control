import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
from tkinter import Tk
from tkinter.filedialog import askdirectory


class Feature(object):
    def __init__(self, feature):
        self.feature = feature
        self.frame_shape = (int(256), int(256))

    def display(self, height = 5):
        plt.figure(figsize=(height * (self.feature.shape[1] / self.feature.shape[0]), height))
        plt.imshow(self.feature, cmap = 'gray')
        plt.tight_layout()
        plt.axis('off')
        plt.show()

    def imshow(self, frame=None):
        if frame:
            cv2.imshow("display feature", frame)
        cv2.imshow("display feature", self.feature)
        k = cv2.waitKey(0)

    def imshow_featureframe(self):
        f = self.get_featureframe()
        self.imshow(frame=f)

    def save(self, fname, path=None, frame=None):
        if path is None:
            path = askdirectory(title='Select Folder') # shows dialog box and return the path
        path = os.path.join(path, fname + '.png')
        print('Save PNG image in: ', path)
        if frame is None:
            cv2.imwrite(path, self.feature)
        else:
            cv2.imwrite(path, frame)

    def save_featureframe(self, fname, path=None, frame_corner_pts=None):
        f = self.get_featureframe(frame_corner_pts)
        self.save(fname=fname, path=path, frame=f)

    def get_warp_transformation(self, frame_corner_pts):
        pts_from = np.float32(frame_corner_pts)
        pts_to = np.float32([[0, 0],
                            [self.frame_shape[0], 0],
                            [self.frame_shape[0], self.frame_shape[1]],
                            [0, self.frame_shape[1]]])
        M = cv2.getPerspectiveTransform(pts_from, pts_to)
        return M

    def get_featureframe(self, M):
        feature_frame = cv2.warpPerspective(self.feature,
                                            M,
                                            (self.frame_shape[0],
                                             self.frame_shape[1]))
        return feature_frame

    def channel_split(self):
        img = self.feature
        # split img
        b, g, r = cv2.split(img)
        return b, g, r

    def img_overlay(self, height_img, toolpath_img):
        condition = np.zeros([256,256,3])
        condition[:,:,0] = (toolpath_img[:,:,2] > 0)
        condition[:,:,1] = (toolpath_img[:,:,2] > 0)
        condition[:,:,2] = (toolpath_img[:,:,2] > 0)
        arr = np.where(condition, toolpath_img, height_img)
        return arr
