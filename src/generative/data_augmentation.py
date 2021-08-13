from process import Processing
from learning import LearningData
import datetime
import time
import json
import os


__FOLDER__ = 'G:/Shared drives/2021_MAS/T3/Common Ground Control/01_data/00_data_collection/01_production'

def augment():
    start = time.time()
    iteration_dirs = os.listdir('G:/Shared drives/2021_MAS/T3/Common Ground Control/01_data/00_data_collection/01_production')
    iteration_dirs.remove('meta.json')
    iteration_dirs.sort()

    for i, id in enumerate(iteration_dirs[:10]):
        # 1. accessing data_collection path/img
        data = LearningData(iter=i, id=id, env='production')
        data.get_iter_dirs()
        data.get_fframe()

        # 2. do processing
        p = Processing()

        # for i in range(number of data augmentations per image pair sample):
        # augment fframe

        # augment fframe_after

        # augment toolpath

        # overlay fframe with toolpath augmented

        # create prefered channel split

        # store augmented image pair


        if i % 100 == 0:
            lap = (time.time()-start)/60
            print('\nprocessing id: {} / {}\nLAP-TIME: {}\n'.format(i, len(iteration_dirs), lap))

    print('\nTotal processing time: ', (time.time()-start)/60, ' min\n\n')



if __name__ == "__main__":

    augment()
