from data import TrainingData
from argparser import parse_args
from scanning import ScanData, HeightMap, PointCloud
from toolpath import random_toolpath_gen, cleaning_toolpath_gen
import scanning
import time
import UR as ur
import scanning.scan
import os
import json

__ITERATION__ = 500
__START__ = 0
__FOLDER__ = "G:/Shared drives/2021_MAS/T3/Common Ground Control/01_data/"


def training(env):
    max_id = meta_data(env)
    if max_id == 0:
        start_id = max_id
    else:
        start_id = max_id + 1

    start = time.time()
    print("starting training mode")
    print("environment: {} \n".format(env))

    for i in range(__START__, __START__+__ITERATION__):
        print('#############  iteration {}  #############\n'.format(i))

        # 1. initiate a new training iteration
        data = TrainingData(iteration=start_id+i, environment=env)
        path_name_raw, path_name_processed, path_name_train = data.create_iter_dirs()
        print('scan id: {}'.format(data.identifier))

        # 2. get toolpath
        toolpath = random_toolpath_gen.get_toolpath(level='2-0',
                                                    curve_type='bezier',
                                                    folder=path_name_raw,
                                                    id=data.identifier)
        data.frame_corner_pts = toolpath.crop_idx

        # 3. robot to scan pose
        ur.ur_helper.scan_pose(scanning_time=15)

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
        ur.execute_toolpath(toolpath.ctrlframes_feature, adapt_height, excavation_time=45)

        # scan
        print('\n#############  iteration {} done  #############\n\n'.format(i))
        # update meta data
        time_spend = (time.time()-start)/60
        meta_data(env=env, new_max_id=start_id+i, time=time_spend, tp_level=toolpath.level)

        # cleaning at every 100 iteration
        if i % 100 == 99:
            print('#############  cleaning {}  #############\n'.format(i))
            ur.ur_helper.scan_pose(scanning_time=10)
            c_frames = cleaning_toolpath_gen.clean()
            ur.ur_helper.execute_toolpath(c_frames,
                                          z_center_toolpathbox2D=0,
                                          excavation_time=125)
            print('\n#############  cleaning {} done  #############\n\n'.format(i))

    print('Total fabrication time: ', (time.time()-start)/60, ' min')


def run():
    pass


def main():
    run_mode, environment = parse_args()

    if run_mode == 'train':
        training(environment)
    elif run_mode == 'run':
        run(environment)
    elif run_mode == 'see':
        ur.ur_helper.scan_pose(scanning_time=0.1)
        scanning.live_scan_stream()
    elif run_mode == 'toolpath':
        backup = "G:/Shared drives/2021_MAS/T3/Common Ground Control/01_data/01_backup"
        id = "test_0000"
        toolpath = random_toolpath_gen.get_toolpath(level='2-2',
                                                    curve_type='bezier',
                                                    iteration=0,
                                                    folder=backup,
                                                    id=id,
                                                    show=False)
        ur.ur_helper.execute_toolpath(toolpath.ctrl_frames,
                                      z_center_toolpathbox2D=0,
                                      excavation_time=0.5)
    elif run_mode == 'calibrate':
        scanning.img_calibration()
    elif run_mode == 'clean':
        frames = cleaning_toolpath_gen.clean()
        ur.ur_helper.execute_toolpath(frames,
                                      z_center_toolpathbox2D=0,
                                      excavation_time=0.5)
    else:
        print("Run mode is not identified")


def meta_data(env, new_max_id=None, time=None, tp_level=None):
    filepath = os.path.join(__FOLDER__, "00_data_collection", ("00_test" if env=='test' else "01_production"), "meta.json")
    with open(filepath, 'r') as f:
        data = json.load(f)

    max_scan_id = data['max_scan_id']

    if new_max_id is not None:
        data["max_scan_id"] = new_max_id
        # export and overwrite json
        with open(filepath, 'w') as o:
            json.dump(data, o, indent=4)

    if time is not None:
        training_time = float(data["total_training_time"]) + time
        data["total_training_time"] = training_time
        # export and overwrite json
        with open(filepath, 'w') as o:
            json.dump(data, o, indent=4)

    if tp_level is not None:
        num = data["toolpath_lvls"][tp_level]
        data["toolpath_lvls"][tp_level] = int(num+1)
        # export and overwrite json
        with open(filepath, 'w') as o:
            json.dump(data, o, indent=4)

    return max_scan_id


if __name__ == "__main__":
    main()
