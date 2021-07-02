import os
import datetime
import open3d as o3d
import cv2


def environment_folder(environment='test'):
    if environment == 'test':
        return "G:/Shared drives/2021_MAS/T3/Common Ground Control/01_data/00_data_collection/00_test/00_scan_data"
    elif environment == 'production':
        return "G:/Shared drives/2021_MAS/T3/Common Ground Control/01_data/00_data_collection/01_production/00_scan_data"

def create_scan_identifier(iteration_num):
    id_num = str(iteration_num).zfill(5)
    return str(id_num) + '_' + str(datetime.date.today())

def make_iteration_dirs(id, environment):
    dir = environment_folder(environment)

    # create folder for raw data
    newDir = os.path.join(id, '00_RAW')
    pathNameRaw = os.path.join(dir, newDir)
    try:
        os.makedirs(pathNameRaw)
    except FileExistsError:
        print("Directory " , pathNameRaw ,  " already exists")

    # create folder for processed data
    newDir = os.path.join(id, '01_processed')
    pathNameProcessed = os.path.join(dir, newDir)
    try:
        os.makedirs(pathNameProcessed)
    except FileExistsError:
        print("Directory " , pathNameProcessed ,  " already exists")

    # create folder for processed data
    newDir = os.path.join(id, '02_training')
    pathNameTrain = os.path.join(dir, newDir)
    try:
        os.makedirs(pathNameTrain)
    except FileExistsError:
        print("Directory " , pathNameTrain ,  " already exists")

    return pathNameRaw, pathNameProcessed, pathNameTrain


def store_all_data(environment, scanID, pointcloud=None, heightMap=None, depthMap=None, colorMap=None):
    dir_raw, dir_processed, dir_train = make_iteration_dirs(scanID, environment)
    if pointcloud:
        o3d.io.write_point_cloud(os.path.join(dir_raw, scanID+'_pcl.ply'), pointcloud)

    if heightMap.any():
        cv2.imwrite(os.path.join(dir_processed, scanID+'_heightmap.jpg'), heightMap)

    if depthMap.any():
        cv2.imwrite(os.path.join(dir_raw, scanID+'_depthmap.jpg'), depthMap)

    if colorMap.any():
        cv2.imwrite(os.path.join(dir_raw, scanID+'_colormap.jpg'), colorMap)



if __name__ == "__main__":
    pass
