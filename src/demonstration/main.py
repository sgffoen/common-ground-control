# built-in
import sys
import cv2 as cv
from tkinter.filedialog import askopenfilename, askdirectory

# package
from ga import DNA
from artist import Artist
from helper import Helper
# package from parent folders
sys.path.insert(0, 'C:/Users/trtku/OneDrive/Data/03_MAS/17_common_ground_control/01_git/common-ground-control/src/data_collection')
import UR as ur
from data import TrainingData
from scanning import ScanData, HeightMap, PointCloud
sys.path.insert(1, 'C:/Users/trtku/OneDrive/Data/03_MAS/17_common_ground_control/01_git/common-ground-control/src/generative')
import Processing

def demo():
    save_dir = "G:/Shared drives/2021_MAS/T3/Common Ground Control/01_data/02_demo/00_test"

    # scan and initialize
    ur.ur_helper.scan_pose(scanning_time=5)
    data = TrainingData(iteration=0, environment='test')
    scan = ScanData()
    pcl_obj = PointCloud(scan)
    heightmap = HeightMap(scan)

    # store data
    data.scan_data = scan
    data.pointcloud = pcl_obj
    data.heightmap = heightmap
    data.store_data()
    print('{}: data is collected and stored'.format(data.identifier))
    height_feature = None

    # get fframe
    dna = DNA()
    helper = Helper()
    height_fframe = helper.crop_feature(height_feature)
    artist = Artist(dna, 5, helper.fframe_bounds)
    toolpath_fframe = helper.crop_feature(artist.toolpath_feature)

    # overlay image
    p = Processing
    input_img = p.img_overlay(fframe, toolpath_fframe)


if __name__ == '__main__':
    pass
