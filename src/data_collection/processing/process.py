import numpy as np
import cv2
import datetime
import os
import shutil


class ImgProcessing():
    def __init__(self, environment, iteration):
        self.environment = environment
        self.iteration = iteration
        self.env_folder = self.get_environment_folder()
        self.id = self.create_identifier()
        self.filedir = self.get_fframe_path()

    def get_environment_folder(self):
        if self.environment == 'test':
            return "G:/Shared drives/2021_MAS/T3/Common Ground Control/01_data/00_data_collection/00_test/"
        elif self.environment == 'production':
            return "G:/Shared drives/2021_MAS/T3/Common Ground Control/01_data/00_data_collection/01_production/"

    def create_identifier(self):
        id_num = str(self.iteration).zfill(5)
        return str(id_num) + '_' + str(datetime.date.today())

    def create_test_dir(self):
        # create folder for training data
        new_path = self.env_folder + self.id + '/' + '03_test'
        try:
            os.makedirs(new_path)
        except FileExistsError:
            print("Directory ", new_path,  " already exists")

    def get_dir(self, type):
        if type == 'processed':
            new_dir = self.id + '/' + '01_processed'
        elif type == 'training':
            new_dir = self.id + '/' + '02_training'
        elif type == 'test':
            new_dir = self.id + '/' + '03_test'
        return self.env_folder + new_dir

    def get_fframe_path(self):
        filedir = self.get_dir('processed')
        rgb_fframe_fname = filedir + '/' + self.id + '_rgb_featureframe.png'
        rgb_fframe_after_fname = filedir + '/' + self.id + '_rgb_featureframe_after.png'
        height_fframe_fname = filedir + '/' + self.id + '_height_featureframe.png'
        height_fframe_after_fname = filedir + '/' + self.id + '_height_featureframe_after.png'
        toolpath_fframe_fname = filedir + '/' + self.id + '_toolpath_featureframe.png'

        self.rgb_fframe = cv2.imread(rgb_fframe_fname)
        self.rgb_fframe_after = cv2.imread(rgb_fframe_after_fname)
        self.height_fframe = cv2.imread(height_fframe_fname)
        self.height_fframe_after = cv2.imread(height_fframe_after_fname)
        self.toolpath_fframe = cv2.imread(toolpath_fframe_fname)

        return filedir

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

    def get_save_path(self, type):
        # get filepath
        filedir = self.get_dir('test')
        if type == 'rgb2rgb':
            filename = self.id + '_rgb2rgb_training.png'
        elif type == 'height2height':
            filename = self.id + '_height2height_training.png'
        elif type == 'height2rgb':
            filename = self.id + '_height2rgb_training.png'
        elif type == 'rgb2height':
            filename = self.id + '_rgb2height_training.png'
        elif type == 'g2b':
            filename = self.id + '_g2b_training.png'
        elif type == 'b2g':
            filename = self.id + '_b2g_training.png'
        filepath = filedir + '/' + filename
        return filepath

    def save_img(self, img_to_save, type):
        fname = self.get_save_path(type)
        cv2.imwrite(fname, img_to_save)


if __name__ == "__main__":
    __IMGNUM__ = 284
    __DELETE__ = False

    if not __DELETE__:
        for i in range(__IMGNUM__):
            ip = ImgProcessing(environment='test', iteration=i)
            ip.create_test_dir()

            # save rgb img
            rgb_img_left = ip.img_overlay(ip.rgb_fframe, ip.toolpath_fframe)
            rgb_img_out = ip.horizontal_stack(rgb_img_left, ip.rgb_fframe_after)
            ip.save_img(rgb_img_out, 'rgb2rgb')

            # save height img
            height_img_left = ip.img_overlay(ip.height_fframe, ip.toolpath_fframe)
            height_img_out = ip.horizontal_stack(height_img_left, ip.height_fframe_after)
            ip.save_img(height_img_out, 'height2height')

            # edit channel
            height_fframe_rgb = ip.chennel_edit(ip.height_fframe)
            height_fframe_after_rgb = ip.chennel_edit(ip.height_fframe_after)

            # save edited imgs
            # height2rgb
            height2rgb = ip.horizontal_stack(height_img_left, ip.rgb_fframe_after)
            ip.save_img(height2rgb, 'height2rgb')
            # rgb2height
            rgb2height = ip.horizontal_stack(rgb_img_left, ip.height_fframe_after)
            ip.save_img(rgb2height, 'rgb2height')
            # g2b
            g_left = ip.img_overlay(height_fframe_rgb[1], ip.toolpath_fframe)
            g2b = ip.horizontal_stack(g_left, height_fframe_after_rgb[2])
            ip.save_img(g2b, 'g2b')
            # g2b
            b_left = ip.img_overlay(height_fframe_rgb[2], ip.toolpath_fframe)
            b2g = ip.horizontal_stack(b_left, height_fframe_after_rgb[1])
            ip.save_img(b2g, 'b2g')

    # run if you want to DELETE 03_test dir you made above
    else:
        for i in range(__IMGNUM__):
            ip = ImgProcessing(environment='test', iteration=i)
            path_to_delete = ip.get_dir('test')
            shutil.rmtree(path_to_delete, ignore_errors=False, onerror=None)
