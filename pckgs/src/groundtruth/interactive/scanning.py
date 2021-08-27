from groundtruth.toolbox import ScanData, HeightMap
import groundtruth.toolbox.ur_helper as ur
from groundtruth.toolbox import EsriGrid
import numpy as np
from groundtruth.toolbox import Facts

__FACTS__ = Facts().facts

def get_height_grid():
    pass

def ur_scan_pose():
    ur.scan_pose(scanning_time=5)

def heightmap2grid():
    pass

def write_height2ascii(arr=None, path=None, cellsize=1.0):
    grid_data = arr[:,:,1]
    rows,cols = np.shape(grid_data)
    esri = EsriGrid(
                    ncols=cols,
                    nrows=rows,
                    xllcorner=__FACTS__.feature_bounds['min_bound'][0],
                    yllcorner=__FACTS__.feature_bounds['min_bound'][1],
                    cellsize=cellsize,
                    grid_data=grid_data,
                    filepath=path,
                    NODATA_VALUE=-9999)

    esri.write_file()
    return esri


if __name__ == '__main__':
    write_height2ascii()
