from process import Processing
from learning import LearningData
import time
import os
import cv2
import numpy as np


__FOLDER__ = 'G:/Shared drives/2021_MAS/T3/Common Ground Control/01_data/00_data_collection/01_production'

def augment_data():
    start = time.time()
    iteration_dirs = os.listdir('G:/Shared drives/2021_MAS/T3/Common Ground Control/01_data/00_data_collection/01_production')
    iteration_dirs.remove('meta.json')
    iteration_dirs.sort()

    for i, id in enumerate(iteration_dirs[1128:]):
        # 1. accessing data_collection path/img
        data = LearningData(iter=i, id=id, env='production')
        data.get_iter_dirs()
        data.get_fframe()

        # 2. do processing
        p = Processing()
        data.create_augment_dir()

        # for i in range(number of data augmentations per image pair sample):
        for j in range(2):
            h2h_aug, gb2gb_aug = augment_fframe(data)

            # store augmented image pair
            h2h_fn = data.id + '_h2h_augmented_' + str(j).zfill(2) + '.png'
            gb2gb_fn = data.id + '_gb2gb_augmented_' + str(j).zfill(2) + '.png'

            cv2.imwrite(os.path.join(data.path_name_augmented, h2h_fn), h2h_aug)
            cv2.imwrite(os.path.join(data.path_name_augmented, gb2gb_fn), gb2gb_aug)


        if i % 100 == 0:
            lap = (time.time()-start)/60
            print('\nprocessing id: {} / {}\nLAP-TIME: {}\n'.format(i, len(iteration_dirs), lap))

    print('\nTotal processing time: ', (time.time()-start)/60, ' min\n\n')

def augment_fframe(data):
    # read training image pairs
    h2h = cv2.imread(data.get_save_path(type='h2h'))
    gb2gb = cv2.imread(data.get_save_path(type='gb2gb'))


    # augmented array
    h2h_aug = np.copy(h2h)
    gb2gb_aug = np.copy(gb2gb)

    # get min and max bounds
    dmin = np.min(h2h[h2h>0])
    dmax = 255 - np.max(h2h)

    # add or substract random choice
    k = np.random.randint(0, 1)
    if k == 1:
        #add
        v = np.random.randint(low=3, high=dmax-1, dtype=np.uint8)
    elif k == 0:
        # subtract
        v = np.random.randint(low=3, high=dmin-1, dtype=np.uint8) * -1

    h2h_aug[h2h_aug!=0] += v.astype(np.uint8)
    gb2gb_aug[gb2gb_aug!=0] += v.astype(np.uint8)

    return h2h_aug, gb2gb_aug



if __name__ == "__main__":

    augment_data()
