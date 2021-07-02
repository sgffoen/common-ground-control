import os
import datetime

FOLDER = "G:/Shared drives/2021_MAS/T3/Common Ground Control/01_data/00_data_collection/00_test/00_scan_data"

def make_iteration_dir(iteration):
    iterName = iteration + '_' + str(datetime.date.today())
    dirName = os.path.join(iterName, '00_RAW')
    pathName = os.path.join(FOLDER, dirName)
    try:
        os.makedirs(pathName)
        print("Directory " , pathName ,  " Created ")
    except FileExistsError:
        print("Directory " , pathName ,  " already exists")


def save_pointcloud(ply_file):
    pass

