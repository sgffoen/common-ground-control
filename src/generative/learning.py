import os
import cv2
import json
import random
import shutil
import datetime


class LearningData(object):
    def __init__(self, iter=None, env=None, lvl=None, img_type=None):
        self.iter = iter
        self.env_folder = self.get_environment_folder(env)
        self.gan_folder = self.get_gan_folder()
        self.lvl = lvl
        self.img_type = img_type

    def get_environment_folder(environment='test'):
        if environment == 'test':
            return "G:/Shared drives/2021_MAS/T3/Common Ground Control/01_data/00_data_collection/00_test/"
        elif environment == 'production':
            return "G:/Shared drives/2021_MAS/T3/Common Ground Control/01_data/00_data_collection/01_production/"
        else:
            return None

    def get_gan_folder():
        return "G:/Shared drives/2021_MAS/T3/Common Ground Control/01_data/01_gan/"

    def create_identifier(self, iter, delta=0):
        id_num = str(iter).zfill(5)
        today = datetime.date.today()
        if delta != 0:
            today -= datetime.timedelta(days=delta)
        return str(id_num) + '_' + str(today)

    def get_iter_dirs(self, iteration):
        dir = self.env_folder
        self.id = self.create_identifier(iteration, delta=0)
        parent_folder = os.path.join(dir, self.id)
        # check the existence of the parent folder
        i = 0
        while not os.path.isdir(parent_folder):
            i += 1
            dir = self.env_folder
            self.id = self.create_identifier(iteration, delta=i)
            parent_folder = os.path.join(dir, self.id)
            if i > 100:
                raise FileNotFoundError('path: {} does not exist'.format(path_name_raw))

        self.path_name_raw = os.path.join(parent_folder, '00_RAW')
        self.path_name_processed = os.path.join(parent_folder, '01_processed')
        self.path_name_train = os.path.join(parent_folder, '02_training')

    def get_fframe(self):
        rgb_fframe_fname = os.path.join(self.path_name_processed, self.id + '_rgb_featureframe.png')
        rgb_fframe_after_fname = os.path.join(self.path_name_processed, self.id + '_rgb_featureframe_after.png')
        height_fframe_fname = os.path.join(self.path_name_processed, self.id + '_height_featureframe.png')
        height_fframe_after_fname = os.path.join(self.path_name_processed, self.id + '_height_featureframe_after.png')
        toolpath_fframe_fname = os.path.join(self.path_name_processed, self.id + '_toolpath_featureframe.png')

        if os.path.isfile(rgb_fframe_fname):
            self.rgb_fframe = cv2.imread(rgb_fframe_fname)
        else:
            raise FileNotFoundError('file: {} does not exist'.format(rgb_fframe_fname))
        if os.path.isfile(rgb_fframe_after_fname):
            self.rgb_fframe_after = cv2.imread(rgb_fframe_after_fname)
        else:
            raise FileNotFoundError('file: {} does not exist'.format(rgb_fframe_after_fname))
        if os.path.isfile(height_fframe_fname):
            self.height_fframe = cv2.imread(height_fframe_fname)
        else:
            raise FileNotFoundError('file: {} does not exist'.format(height_fframe_fname))
        if os.path.isfile(height_fframe_after_fname):
            self.height_fframe_after = cv2.imread(height_fframe_after_fname)
        else:
            raise FileNotFoundError('file: {} does not exist'.format(height_fframe_after_fname))
        if os.path.isfile(toolpath_fframe_fname):
            self.toolpath_fframe = cv2.imread(toolpath_fframe_fname)
        else:
            raise FileNotFoundError('file: {} does not exist'.format(toolpath_fframe_fname))

    def get_toolpath_level(self):
        filepath = os.path.join(self.path_name_raw, '{}_toolpath.json'.format(self.id))
        with open(filepath, 'r') as i:
            toolpath = json.load(i)
        level = toolpath['lvl']
        return level

    def get_save_path(self, type):
        # get filepath
        filedir = self.path_name_train
        if type == 'c2c':
            filename = self.id + '_c2c_training.png'
        elif type == 'h2h':
            filename = self.id + '_h2h_training.png'
        elif type == 'h2c':
            filename = self.id + '_h2c_training.png'
        elif type == 'c2h':
            filename = self.id + '_c2h_training.png'
        elif type == 'g2b':
            filename = self.id + '_g2b_training.png'
        elif type == 'b2g':
            filename = self.id + '_b2g_training.png'
        filepath = filedir + '/' + filename
        return filepath

    def create_dataset_dir(self, lvl):
        dir = self.gan_folder
        new_dir = os.path.join(dir, '00_dataset')

        # create folder for a parent folder
        dataset_dir = os.path.join(new_dir, 'dataset_lvl_{}'.format(lvl))
        try:
            os.makedirs(dataset_dir)
        except FileExistsError:
            pass
            # print('Directory ', dataset_dir, " already exists")

        # create folder for child folders
        # c2c
        c2c_dir = os.path.join(dataset_dir, 'c2c')
        self.c2c_train_dir = os.path.join(c2c_dir, 'train')
        self.c2c_test_dir = os.path.join(c2c_dir, 'test')
        c2c_logs_dir = os.path.join(c2c_dir, 'logs')
        try:
            os.makedirs(c2c_dir)
            os.makedirs(self.c2c_train_dir)
            os.makedirs(self.c2c_test_dir)
            os.makedirs(c2c_logs_dir)
        except FileExistsError:
            pass
            # print("Directory ", c2c_dir, " already exists")

        # h2h
        h2h_dir = os.path.join(dataset_dir, 'h2h')
        self.h2h_train_dir = os.path.join(h2h_dir, 'train')
        self.h2h_test_dir = os.path.join(h2h_dir, 'test')
        h2h_logs_dir = os.path.join(h2h_dir, 'logs')
        try:
            os.makedirs(h2h_dir)
            os.makedirs(self.h2h_train_dir)
            os.makedirs(self.h2h_test_dir)
            os.makedirs(h2h_logs_dir)
        except FileExistsError:
            pass
            # print("Directory ", h2h_dir, " already exists")

        # h2c
        h2c_dir = os.path.join(dataset_dir, 'h2c')
        self.h2c_train_dir = os.path.join(h2c_dir, 'train')
        self.h2c_test_dir = os.path.join(h2c_dir, 'test')
        h2c_logs_dir = os.path.join(h2c_dir, 'logs')
        try:
            os.makedirs(h2c_dir)
            os.makedirs(self.h2c_train_dir)
            os.makedirs(self.h2c_test_dir)
            os.makedirs(h2c_logs_dir)
        except FileExistsError:
            pass
            # print("Directory ", h2c_dir, " already exists")

        # c2h
        c2h_dir = os.path.join(dataset_dir, 'c2h')
        self.c2h_train_dir = os.path.join(c2h_dir, 'train')
        self.c2h_test_dir = os.path.join(c2h_dir, 'test')
        c2h_logs_dir = os.path.join(c2h_dir, 'logs')
        try:
            os.makedirs(c2h_dir)
            os.makedirs(self.c2h_train_dir)
            os.makedirs(self.c2h_test_dir)
            os.makedirs(c2h_logs_dir)
        except FileExistsError:
            pass
            # print("Directory ", c2h_dir, " already exists")

        # b2g
        b2g_dir = os.path.join(dataset_dir, 'b2g')
        self.b2g_train_dir = os.path.join(b2g_dir, 'train')
        self.b2g_test_dir = os.path.join(b2g_dir, 'test')
        b2g_logs_dir = os.path.join(b2g_dir, 'logs')
        try:
            os.makedirs(b2g_dir)
            os.makedirs(self.b2g_train_dir)
            os.makedirs(self.b2g_test_dir)
            os.makedirs(b2g_logs_dir)
        except FileExistsError:
            pass
            # print("Directory ", b2g_dir, " already exists")

        # g2b
        g2b_dir = os.path.join(dataset_dir, 'g2b')
        self.g2b_train_dir = os.path.join(g2b_dir, 'train')
        self.g2b_test_dir = os.path.join(g2b_dir, 'test')
        g2b_logs_dir = os.path.join(g2b_dir, 'logs')
        try:
            os.makedirs(g2b_dir)
            os.makedirs(self.g2b_train_dir)
            os.makedirs(self.g2b_test_dir)
            os.makedirs(g2b_logs_dir)
        except FileExistsError:
            pass
            # print("Directory ", g2b_dir, " already exists")

    def store_data(self):
        # c2c
        path_from = self.get_save_path(type='c2c')
        if self.iter % 2 == 0:
            # save to train
            path_to = self.c2c_train_dir
        else:
            # save to test
            path_to = self.c2c_test_dir
        try:
            shutil.copy(path_from, path_to)
        except shutil.SameFileError:
            pass
            # print("File ", path_to, ' already exists')

        # h2h
        path_from = self.get_save_path(type='h2h')
        if self.iter % 2 == 0:
            # save to train
            path_to = self.h2h_train_dir
        else:
            # save to test
            path_to = self.h2h_test_dir
        try:
            shutil.copy(path_from, path_to)
        except shutil.SameFileError:
            pass
            # print("File ", path_to, ' already exists')

        # h2c
        path_from = self.get_save_path(type='h2c')
        if self.iter % 2 == 0:
            # save to train
            path_to = self.h2c_train_dir
        else:
            # save to test
            path_to = self.h2c_test_dir
        try:
            shutil.copy(path_from, path_to)
        except shutil.SameFileError:
            pass
            # print("File ", path_to, ' already exists')

        # c2h
        path_from = self.get_save_path(type='c2h')
        if self.iter % 2 == 0:
            # save to train
            path_to = self.c2h_train_dir
        else:
            # save to test
            path_to = self.c2h_test_dir
        try:
            shutil.copy(path_from, path_to)
        except shutil.SameFileError:
            pass
            # print("File ", path_to, ' already exists')

        # g2b
        path_from = self.get_save_path(type='g2b')
        if self.iter % 2 == 0:
            # save to train
            path_to = self.g2b_train_dir
        else:
            # save to test
            path_to = self.g2b_test_dir
        try:
            shutil.copy(path_from, path_to)
        except shutil.SameFileError:
            pass
            # print("File ", path_to, ' already exists')

        # b2g
        path_from = self.get_save_path(type='b2g')
        if self.iter % 2 == 0:
            # save to train
            path_to = self.b2g_train_dir
        else:
            # save to test
            path_to = self.b2g_test_dir
        try:
            shutil.copy(path_from, path_to)
        except shutil.SameFileError:
            pass
            # print("File ", path_to, ' already exists')

    def get_dataset_dir(self):
        dir = os.path.join(self.gan_folder, '00_dataset')
        dataset_dir = os.path.join(dir, 'dataset_lvl_{}'.format(self.lvl))
        datatype_dir = os.path.join(dataset_dir, self.img_type)
        self.train_dir = os.path.join(datatype_dir, 'train')
        self.test_dir = os.path.join(datatype_dir, 'test')
        self.log_dir = os.path.join(datatype_dir, 'logs')

        self.plot_dir = os.path.join(self.log_dir, "plots")
        try:
            os.mkdir(self.plot_dir)
        except FileExistsError:
            print("Directory ", self.plot_dir,  " already exists")

    def get_random_img_path(self):
        fname = random.choice(os.listdir(self.test_dir))
        img_path = os.path.join(self.test_dir, fname)
        return img_path


if __name__ == '__main__':
    pass
