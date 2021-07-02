import argparse
from argparser import parse_args
import scanning

def training(env):
    print("starting training mode")
    print("environment: {}".format(env))

    for i in range(3):
        scan_id = scanning.data_collection.create_scan_identifier(i)

        pcl, height_map, depth, color = scanning.collect_data()

        # store data
        scanning.data_collection.store_all_data(environment=env, scanID=scan_id, pointcloud=pcl, heightMap=height_map, depthMap=depth, colorMap=color )
        # get toolpath

        # execute toolpath

        # scan

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
