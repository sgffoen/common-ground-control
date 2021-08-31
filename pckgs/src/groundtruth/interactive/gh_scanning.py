from groundtruth.toolbox import ScanData, HeightMap
import groundtruth.toolbox.ur_helper as ur
from groundtruth.toolbox import EsriGrid
from groundtruth.toolbox import Facts
from groundtruth.toolbox.toolpath import Dimension
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

def height2ascii(arr, path, cellsize=1.0):
    # get offset value
    d = Dimension()
    grid_data = arr
    # print(grid_data.shape) = (729, 1135)
    rows, cols = np.shape(grid_data)
    esri = EsriGrid(
                    ncols=cols,
                    nrows=rows,
                    xllcorner=0 + d.feature_origin_x,
                    yllcorner=-729 - d.feature_origin_y,
                    cellsize=cellsize,
                    grid_data=grid_data,
                    filepath=os.path.join(path, 'grid.asc'),
                    NODATA_VALUE=-9999)

    esri.write_file()
    return esri.filepath

def gray2height(v, ori_Min=0, ori_Max=255, targetMin=0.0, targetMax=150.0):
    rv = ((v-ori_Min)/(ori_Max-ori_Min))*(targetMax-targetMin)+targetMin
    return rv


if __name__ == '__main__':
    pass
