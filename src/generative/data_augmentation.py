from process import Processing
from learning import LearningData
import time
import os
import cv2
import numpy as np


__FOLDER__ = 'G:/Shared drives/2021_MAS/T3/Common Ground Control/01_data/00_data_collection/01_production'

def augment_data():
    start = time.time()
    iteration_dirs = os.listdir(__FOLDER__)
    iteration_dirs.remove('meta.json')
    iteration_dirs.sort()

    for i, id in enumerate(iteration_dirs[:]): # start stop
        # 1. accessing data_collection path/img
        data = LearningData(iter=i, id=id, env='production')
        data.get_iter_dirs()
        data.get_fframe()

        # 2. do processing
        p = Processing()
        data.create_augment_dir()

        # for i in range(number of data augmentations per image pair sample):
        for j in range(2):
            tg2g_aug, tgd2tgd_aug = augment_fframe(data)

            if tg2g_aug is not None and tgd2tgd_aug is not None:

                # store augmented image pair
                tg2g_fn = data.id + '_tg2g_augmented_' + str(j).zfill(2) + '.png'
                tgd2tgd_fn = data.id + '_tgd2tgd_augmented_' + str(j).zfill(2) + '.png'

                cv2.imwrite(os.path.join(data.path_name_augmented, tg2g_fn), tg2g_aug)
                cv2.imwrite(os.path.join(data.path_name_augmented, tgd2tgd_fn), tgd2tgd_aug)


        if i % 100 == 0:
            lap = (time.time()-start)/60
            print('\nprocessing id: {} / {}\nLAP-TIME: {}\n'.format(i, len(iteration_dirs), lap))

    print('\nTotal processing time: ', (time.time()-start)/60, ' min\n\n')

def augment_fframe(data):
    # read training image pairs
    tg2g = cv2.imread(data.get_save_path(type='tg2g'))
    tgd2tgd = cv2.imread(data.get_save_path(type='tgd2tgd'))

    try:
        # augmented array
        tg2g_aug = np.copy(tg2g)
        tgd2tgd_aug = np.copy(tgd2tgd)

        # get min and max bounds
        dmin = np.min(tg2g[tg2g>0])
        dmax = 255 - np.max(tg2g)

        # add or substract random choice
        k = np.random.randint(0, 1)
        if k == 1:
            #add
            v = np.random.randint(low=3, high=dmax-1, dtype=np.uint8)
        elif k == 0:
            # subtract
            v = np.random.randint(low=3, high=dmin-1, dtype=np.uint8) * -1

        tg2g_aug[tg2g_aug!=0] += v.astype(np.uint8)
        tgd2tgd_aug[tgd2tgd_aug!=0] += v.astype(np.uint8)

        return tg2g_aug, tgd2tgd_aug
    except:
        print('skipped {}'.format(data.id))
        return None, None


if __name__ == "__main__":

    augment_data()
