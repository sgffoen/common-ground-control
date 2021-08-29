from groundtruth.toolbox import ScanData, HeightMap
import groundtruth.toolbox.ur_helper as ur
from groundtruth.toolbox import EsriGrid
from groundtruth.toolbox import Facts

# python libs
import numpy as np
import os

__FACTS__ = Facts().facts
__GH_DATA__ = 'C:/Users/simon/Documents/MAS DFAB/04_MAS_THESIS/00_git/common-ground-control/grasshopper/data'


def scan():
    ur.scan_pose(scanning_time=5)
    scan = ScanData()
    return scan

def get_heightmap(scan):
    heightmap = HeightMap(scan)
    hm_feature = heightmap.height2feature()
    hm_feature.save(fname='height_feature', path=__GH_DATA__)
    return hm_feature

def height2ascii(arr, cellsize=1.0, path=__GH_DATA__):
    grid_data = arr
    rows,cols = np.shape(grid_data)
    esri = EsriGrid(
                    ncols=cols,
                    nrows=rows,
                    xllcorner=__FACTS__.feature_bounds['min_bound'][0],
                    yllcorner=__FACTS__.feature_bounds['min_bound'][1],
                    cellsize=cellsize,
                    grid_data=grid_data,
                    filepath=os.path.join(path, 'grid.asc'),
                    NODATA_VALUE=-9999)

    esri.write_file()
    return esri

def gray2height(v, ori_Min=0, ori_Max=255, targetMin=0.0, targetMax=150.0):
    rv = ((v-ori_Min)/(ori_Max-ori_Min))*(targetMax-targetMin)+targetMin
    return rv


if __name__ == '__main__':
    pass
