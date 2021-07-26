import numpy as np
import cv2
import datetime


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

    def get_dir(self, type):
        if type == 'processed':
            new_dir = self.id + '/' + '01_processed'
        elif type == 'training':
            new_dir = self.id + '/' + '02_training'
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

    def chennel_edit(self):
        pass

    def color_edit(self):
        pass

    def range_remap(self):
        pass

    def res_edit(self):
        pass

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
        filedir = self.get_dir('training')
        if type == 'rgb':
            filename = self.id + '_rgb_training.png'
        elif type == 'height':
            filename = self.id + '_height_training.png'
        filepath = filedir + '/' + filename
        return filepath

    def save_img(self, img_to_save, type):
        fname = self.get_save_path(type)
        cv2.imwrite(fname, img_to_save)


if __name__ == "__main__":
    __IMGNUM__ = 100

    for i in range(__IMGNUM__):
        ip = ImgProcessing(environment='test', iteration=i)

        # save rgb img
        rgb_img_left = ip.img_overlay(ip.rgb_fframe, ip.toolpath_fframe)
        rgb_img_out = ip.horizontal_stack(rgb_img_left, ip.rgb_fframe_after)
        ip.save_img(rgb_img_out, 'rgb')

        # save height img
        height_img_left = ip.img_overlay(ip.height_fframe, ip.toolpath_fframe)
        height_img_out = ip.horizontal_stack(height_img_left, ip.height_fframe_after)
        ip.save_img(height_img_out, 'height')
