import datetime
import os


def get_environment_folder(environment='test'):
    if environment == 'test':
        return "G:/Shared drives/2021_MAS/T3/Common Ground Control/01_data/00_data_collection/00_test/"
    elif environment == 'production':
        return "G:/Shared drives/2021_MAS/T3/Common Ground Control/01_data/00_data_collection/01_production/"


def create_identifier(iter):
    id_num = str(iter).zfill(5)
    return str(id_num) + '_' + "2021-07-26"
    # return str(id_num) + '_' + str(datetime.date.today())


def get_dir(dir, id):
    new_dir = id + '/' + '03_test'
    path_name_raw = dir + new_dir
    return path_name_raw


def get_img_path(env, iter, type):
    dir = get_environment_folder(environment='test')
    id = create_identifier(iter)
    filedir = get_dir(dir, id)

    if type == 'rgb2rgb':
        filename = id + '_rgb2rgb_training.png'
    elif type == 'height2height':
        filename = id + '_height2height_training.png'
    elif type == 'height2rgb':
        filename = id + '_height2rgb_training.png'
    elif type == 'rgb2height':
        filename = id + '_rgb2height_training.png'
    elif type == 'g2b':
        filename = id + '_g2b_training.png'
    elif type == 'b2g':
        filename = id + '_b2g_training.png'
    filepath = filedir + '/' + filename
    return filepath


def create_dataset_dir(env, iter, data):
    dir = get_environment_folder(env)
    id = create_identifier(iter)

    if data == 'train':
        new_dir = dir + id + '_data_set_train'
    elif data == 'test':
        new_dir = dir + id + '_data_set_test'

    try:
        os.mkdir(new_dir)
    except FileExistsError:
        print("Directory " , new_dir ,  " already exists")
    return new_dir

if __name__ == "__main__":
    # iteration = 0
    # environment = 'test'

    # filepath = get_img_path(env=environment, iter=iteration, type='rgb2rgb')

    # print(filepath)
    # create_dataset_dir(env='test', iter=0)
    pass
