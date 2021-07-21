from data import TrainingData, Feature
from argparser import parse_args
from scanning import ScanData, HeightMap, PointCloud
import scanning
import time
import UR as ur
import toolpath as tp
import scanning.scan

ITERATION = 50


def training(env):
    start = time.time()
    print("starting training mode")
    print("environment: {} \n".format(env))

    for i in range(ITERATION):
        print('#############  iteration {}  #############\n'.format(i))

        # 1. initiate a new training iteration
        data = TrainingData(iteration=i, environment=env)
        data.create_iter_dirs()
        print('scan id: {}'.format(data.identifier))

        # 2. get toolpath

        # 3. robot to scan pose
        ur.ur_helper.scan_pose(scanning_time=7.5)

        # 4. scan and create data
        scan = ScanData()
        pcl_obj = PointCloud(scan)
        heightmap = HeightMap(scan)

        # 5. store data
        data.scan_data = scan
        data.pointcloud = pcl_obj
        data.heightmap = heightmap

        print('{}: data is collected and stored'.format(data.identifier))



        # adapt toolpath

        # execure toolpath

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
        # level == 1.0 / 1.1 / 1.2 / 2.0 / 2.1 / 2.2
        # curvetype == polyline / bezier
        frames = tp.get_and_store_toolpath(level='2.0', curvetype='bezier')
        ur.ur_helper.execute_toolpath(frames, z_center_toolpathbox2D=0)
    else:
        print("Run mode is not identified")


if __name__ == "__main__":
    main()
