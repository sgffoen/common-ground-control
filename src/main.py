from argparser import parse_args
import scanning
import time
import UR as ur

ITERATION = 50


def training(env):
    start = time.time()
    print("starting training mode")
    print("environment: {} \n".format(env))

    for i in range(ITERATION):
        print('#############  iteration {}  #############\n'.format(i))
        # # scan pose
        ur.ur_helper.scan_pose(scanning_time=7.5)

        scan_id = scanning.data_collection.create_scan_identifier(i)
        print('scan id: {}'.format(scan_id))

        pcl, height_map, depth, color = scanning.collect_data()

        # store data
        scanning.data_collection.store_all_data(environment=env,
                                                scanID=scan_id,
                                                pointcloud=pcl,
                                                heightMap=height_map,
                                                depthMap=depth,
                                                colorMap=color)
        print('{}: data is collected and stored'.format(scan_id))

        # get toolpath

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
    elif run_mode == 'calibration':
        ur.ur_helper.scan_pose(scanning_time=0.1)
        scanning.img_calibration()
    else:
        print("Run mode is not identified")


if __name__ == "__main__":
    main()
