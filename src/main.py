from argparser import parse_args
import scanning
import ur_helper
import time


ITERATION = 1

""
def training(env):
    start = time.time()
    print("starting training mode")
    print("environment: {}".format(env))

    for i in range(ITERATION):
        print('iteration{}'.format(i))
        # scan pose
        ur_helper.scan_pose(scanning_time=7.5)

        scan_id = scanning.data_collection.create_scan_identifier(i)

        pcl, height_map, depth, color = scanning.collect_data()

        # store data
        scanning.data_collection.store_all_data(environment=env,
                                                scanID=scan_id,
                                                pointcloud=pcl,
                                                heightMap=height_map,
                                                depthMap=depth,
                                                colorMap=color)
        print('scanning DONE')

        # get toolpath
        random_toolpath, x_ind, y_ind = ur_helper.get_toolpath(i)

        # adapt toolpath
        z_fig_center = ur_helper.get_z_fig(pcl, x_ind, y_ind)

        # execure toolpath
        ur_helper.execute_toolpath(random_toolpath,
                                   z_fig_center,
                                   excavation_time=20)

        # scan
        print('iteration {} done'.format(i))
    print('fabrication_time: ', (time.time()-start)/60, ' min')


def run():
    pass


def main():
    run_mode, environment = parse_args()
    if run_mode == 'train':
        training(environment)
    elif run_mode == 'run':
        run(environment)
    else:
        print("Run mode is not identified")


if __name__ == "__main__":
    main()
