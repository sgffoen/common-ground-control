from groundtruth.toolbox import ScanData, HeightMap
import groundtruth.toolbox.ur_helper as ur
from groundtruth.toolbox import EsriGrid
from groundtruth.toolbox import Facts
from groundtruth.toolbox.toolpath import Dimension
import compas.utilities as cu
import cv2
# python libs
import numpy as np
import os

__FACTS__ = Facts().facts
__HERE__ = os.path.dirname(__file__)


def scan():
    ur.scan_pose(scanning_time=5)
    scan = ScanData()
    return scan

def get_heightmap(scan):
    heightmap = HeightMap(scan)
    hm_feature = heightmap.height2feature()
    return hm_feature

def height2ascii(arr, path, fname, cellsize=1):
    # get offset value
    d = Dimension()
    grid_data = arr[::cellsize, ::cellsize].copy()
    # print(grid_data.shape) = (729, 1135)
    rows, cols = np.shape(grid_data)
    esri = EsriGrid(
                    ncols=cols,
                    nrows=rows,
                    xllcorner=0 + d.feature_origin_x,
                    yllcorner=-729 - d.feature_origin_y,
                    cellsize=cellsize,
                    grid_data=grid_data,
                    filepath=os.path.join(path, fname),
                    NODATA_VALUE=-9999)

    esri.write_file()
    return esri.filepath


def ascii2height():

    ncols = 1135
    nrows = 729
    xllCorner = 17.0
    yllCorner = -751.0
    cellsize = 1.0
    NODATA_value = -9999.0

    # read ascii
    __HERE__ = os.getcwd()
    fname = "grasshopper/df_target.asc"
    dir = os.path.join(__HERE__ , fname)

    data = open(dir)

    lines = data.readlines()

    arr = np.empty([nrows, ncols], dtype=np.float32)
    for i in range(729):
        index = i + 6
        myArray = np.fromstring(lines[index], dtype = float, sep = ' ')
        pixarray = cu.remap_values(myArray,
                                   original_min=0,
                                   original_max=150,
                                   target_min=0,
                                   target_max=255)
        arr[i] = pixarray
    arr.astype(np.uint8)
    # 3 channels
    im = np.stack((arr,)*3, axis=-1)
    fname = "G:/Shared drives/Ko-Simon MAS thesis (temporary)/01_data/02_demo/01_interactive-gh/01_data/target.png"
    cv2.imwrite(fname, im)


if __name__ == '__main__':
    ascii2height()
