from data import TrainingData
from argparser import parse_args
from scanning import ScanData, HeightMap, PointCloud, Feature
from toolpath import random_toolpath_gen as tp
import scanning
import time
import UR as ur
import scanning.scan

ITERATION = 5


def training(env):
    start = time.time()
    print("starting training mode")
    print("environment: {} \n".format(env))

    for i in range(ITERATION):
        print('#############  iteration {}  #############\n'.format(i))

        # 1. initiate a new training iteration
        data = TrainingData(iteration=i, environment=env)
        path_name_raw, path_name_processed, path_name_train = data.create_iter_dirs()
        print('scan id: {}'.format(data.identifier))

        # 2. get toolpath
        toolpath = tp.get_toolpath(level='1.0',
                                   curve_type='bezier',
                                   iteration=i,
                                   folder=path_name_raw,
                                   id=data.identifier)
        data.frame_corner_pts = toolpath.crop_idx

        # 3. robot to scan pose
        ur.ur_helper.scan_pose(scanning_time=0.5)

        # 4. scan and create data
        scan = ScanData()
        pcl_obj = PointCloud(scan)
        heightmap = HeightMap(scan)

        # 5. store data
        data.toolpath = toolpath
        data.scan_data = scan
        data.pointcloud = pcl_obj
        data.heightmap = heightmap
        data.store_data()
        print('{}: data is collected and stored'.format(data.identifier))

        # adapt toolpath
        adapt_height = None

        # execure toolpath
        ur.execute_toolpath(toolpath.ctrl_frames, adapt_height, excavation_time=0.5)

        # scan
        print('\n#############  iteration {} done  #############\n\n'.format(i))
    print('Total fabrication time: ', (time.time()-start)/60, ' min')


def run():
    pass


def main():
    run_mode, environment = parse_args()

    if run_mode == 'train':
        training(environment)
    elif run_mode == 'run':
        run(environment)
    elif run_mode == 'scan':
        ur.ur_helper.scan_pose(scanning_time=0.1)
        scanning.live_scan_stream()
    elif run_mode == 'toolpath':
        backup = "G:/Shared drives/2021_MAS/T3/Common Ground Control/01_data/01_backup"
        id = "test_0000"
        toolpath = tp.get_toolpath(level='2.2',
                                   curve_type='bezier',
                                   iteration=0,
                                   folder=backup,
                                   id=id,
                                   show=False)
        ur.ur_helper.execute_toolpath(toolpath.ctrl_frames,
                                      z_center_toolpathbox2D=0,
                                      excavation_time=0.5)

    else:
        print("Run mode is not identified")


if __name__ == "__main__":
    main()
