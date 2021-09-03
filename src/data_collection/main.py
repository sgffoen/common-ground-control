from data import TrainingData
from argparser import parse_args
from scanning import ScanData, HeightMap, PointCloud
from toolpath import random_toolpath_gen, cleaning_toolpath_gen, Toolpath, Dimension
import scanning
import time
import UR as ur
import scanning.scan
import os
import json
import datetime
import random as r

__ITERATION__ = 500
__START__ = 0
# __FOLDER__ = "G:/Shared drives/2021_MAS/T3/Common Ground Control/01_data/"
__FOLDER__ = "G:/My Drive/05_T3/01_data/"


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
        print()
        data = TrainingData(iteration=start_id+i, environment=env)
        path_name_raw, path_name_processed, path_name_train = data.create_iter_dirs()
        print('scan id: {}'.format(data.identifier))

        # 2. robot to scan pose
        ur.ur_helper.scan_pose(scanning_time=6)

        # 3. scan and create data
        scan = ScanData()
        pcl_obj = PointCloud(scan)
        heightmap = HeightMap(scan)

        # 4. set data
        data.scan_data = scan
        data.pointcloud = pcl_obj
        data.heightmap = heightmap

        # 5. get toolpath
        hm_feature = data.get_hm_feature()
        d = Dimension()
        adaptive_depth = r.randint(0, 30)
        # 5-1. check the depth of previous toolpath for multiple excavation
        # check min height of the ctrl_frames
        if data.prev_min_height > 100:  # randomize toolpath
            # set adaptive height if necessary
            digging_depth = r.randint(5, 20)
            level = 'multi'
        else:  # keep diging with the same toolpath]
            digging_depth = data.prev_dd
            level = 'repeat'

        print('\n', level, data.prev_min_height, digging_depth, '\n')

        tp = Toolpath(level=level,
                        curve_type='bezier',
                        num_ctrl_pts=2,
                        segments_num=50,
                        thickness=2,
                        parent_folder=path_name_raw,
                        id=data.identifier,
                        d=d,
                        hm_feature=hm_feature.feature,
                        adaptive_depth=adaptive_depth,
                        digging_depth=digging_depth,
                        prev_frames=data.prev_frames)

        # 5-2. get crop index
        data.toolpath = tp
        data.frame_corner_pts = tp.crop_idx

        # 6. store data
        data.store_data()
        tp.export_json()
        print('{}: data is collected and stored'.format(data.identifier))

        # 7. execure toolpath
        ur.execute_toolpath(tp.ctrlframes, excavation_time=24)

        # 8. update meta data
        time_spend = (time.time()-start)/60
        meta_data(env=env, new_max_id=start_id+i, time=time_spend, tp_level=tp.level)
        print('\n#############  iteration {} done  #############\n\n'.format(i))

        # 9. cleaning at every 100 iteration
        # if i % 100 == 99:
        #     print('#############  cleaning {}  #############\n'.format(i))
        #     ur.ur_helper.scan_pose(scanning_time=10)
        #     c_frames = cleaning_toolpath_gen.clean()
        #     ur.ur_helper.execute_toolpath(c_frames,
        #                                   z_center_toolpathbox2D=0,
        #                                   excavation_time=125)
        #     print('\n#############  cleaning {} done  #############\n\n'.format(i))

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
    elif run_mode == 'scan':
        scan_path = "G:/Shared drives/2021_MAS/T3/Common Ground Control/01_data/03_test/00_test_scans"
        ur.ur_helper.scan_pose(scanning_time=5.0)
        s = ScanData()
        s.display_scan(s.depth_scan)
        pcl_obj = PointCloud(s)
        pcl_obj.write_pointcloud(pcl_obj.get_feature(),
                                             fname='pcl_feature',
                                             path=scan_path)
        heightmap = HeightMap(s)
        f = heightmap.height2feature()
        f.save('height_scan_' + str(datetime.date.today()), path=scan_path)

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
