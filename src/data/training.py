import os
import json
import datetime
if __name__ == "__main__":
    pass
else:
    from scanning import Feature

class TrainingData(object):
    def __init__(self, iteration, environment='test', toolpath=None, heightmap=None, pointcloud=None, scan_data=None, frame_corner_pts=None):
        self.toolpath = toolpath
        self.heightmap = heightmap
        self.pointcloud = pointcloud
        self.scan_data = scan_data
        self.environment_folder = self.get_environment_folder(environment=environment)
        self.identifier = self.create_identifier(int(iteration))
        self.frame_corner_pts = frame_corner_pts
        self.prev_frame_corner_pts = self.get_previous_frame_corner_pts(iteration)

    def store_data(self):
        dir_raw, dir_processed, dir_train = self.create_iter_dirs()
        # 1. pointcloud
        try:
            self.pointcloud.write_pointcloud(self.pointcloud.get_pointcloud_transformed(),
                                             fname=self.identifier + '_pcl',
                                             path=dir_raw)
            self.pointcloud.write_pointcloud(self.pointcloud.get_feature(),
                                             fname=self.identifier + '_pcl_feature',
                                             path=dir_processed)
        except:
            print("Could not save point cloud for: {}".format(self.identifier))

        # 2. heightmap
        try:
            self.heightmap.write_height2ascii(path=os.path.join(dir_processed, self.identifier + "_esriGrid.asc"))
            # self.heightmap.write_mesh(mesh=self.heightmap.get_mesh_feature(),
            #                            fname=self.identifier + '_mesh_feature',
            #                            path=dir_processed)

        except:
            print("Could not save height map for: {}".format(self.identifier))

        # 3. scan data
        try:
            self.scan_data.save_scan(self.scan_data.depth_scan, self.identifier + '_depth', dir_raw)
            self.scan_data.save_scan(self.scan_data.rgb_scan, self.identifier + '_rgb', dir_raw)
            self.scan_data.save_scan(self.scan_data.ir_scan, self.identifier + '_IR', dir_raw)
        except:
            print("Could not save scan data: {}".format(self.identifier))

        # 4. features
        try:
            #processed
            rgb_feature = self.scan_data.get_feature(self.scan_data.rgb_scan)
            depth_feature = self.scan_data.get_feature(self.scan_data.depth_scan)
            ir_feature = self.scan_data.get_feature(self.scan_data.ir_scan)
            hm_feature = self.heightmap.height2feature()
            #store features
            rgb_feature.save(fname=self.identifier + '_rgb_feature', path=dir_processed)
            depth_feature.save(fname=self.identifier + '_depth_feature', path=dir_processed)
            ir_feature.save(fname=self.identifier + '_ir_feature', path=dir_processed)
            hm_feature.save(fname=self.identifier + '_height_feature', path=dir_processed)
        except:
            print("Could not save features for: {}".format(self.identifier))

        # store current feature frames
        try:
            rgb_feature.save_featureframe(fname=self.identifier + '_rgb_featureframe', path=dir_processed, frame_corner_pts=self.frame_corner_pts)
            depth_feature.save_featureframe(fname=self.identifier + '_depth_featureframe', path=dir_processed, frame_corner_pts=self.frame_corner_pts)
            ir_feature.save_featureframe(fname=self.identifier + '_ir_featureframe', path=dir_processed, frame_corner_pts=self.frame_corner_pts)
            hm_feature.save_featureframe(fname=self.identifier + '_height_featureframe', path=dir_processed, frame_corner_pts=self.frame_corner_pts)
        except:
            print("Could not save features frames for: {}".format(self.identifier))

        # store previous feature frames
        if self.prev_frame_corner_pts != None:
            try:
                rgb_feature.save_featureframe(fname=self.identifier + '_rgb_prevfeatureframe', path=dir_processed, frame_corner_pts=self.prev_frame_corner_pts)
                depth_feature.save_featureframe(fname=self.identifier + '_depth_prevfeatureframe', path=dir_processed, frame_corner_pts=self.prev_frame_corner_pts)
                ir_feature.save_featureframe(fname=self.identifier + '_ir_prevfeatureframe', path=dir_processed, frame_corner_pts=self.prev_frame_corner_pts)
                hm_feature.save_featureframe(fname=self.identifier + '_height_prevfeatureframe', path=dir_processed, frame_corner_pts=self.prev_frame_corner_pts)
            except:
                print("Could not save previous features frames for: {}".format(self.identifier))

        # store toolpath
        try:
            toolpath_feature = Feature(self.toolpath.img)
            toolpath_feature.save(fname=self.identifier + '_toolpath_feature', path=dir_processed)
            toolpath_feature.save_featureframe(fname=self.identifier + '_toolpath_featureframe', path=dir_processed, frame_corner_pts=self.frame_corner_pts)
        except:
            print("Could not save toolpath feature for: {}".format(self.identifier))




    def get_environment_folder(self, environment='test'):

        if environment == 'test':
            return "G:/Shared drives/2021_MAS/T3/Common Ground Control/01_data/00_data_collection/00_test/"
        elif environment == 'production':
            return "G:/Shared drives/2021_MAS/T3/Common Ground Control/01_data/00_data_collection/01_production/"

    def create_identifier(self, iter):
        id_num = str(iter).zfill(5)
        return str(id_num) + '_' + str(datetime.date.today())

    def create_iter_dirs(self):
        dir = self.environment_folder

        # create folder for raw data
        new_dir = os.path.join(self.identifier, '00_RAW')
        path_name_raw = os.path.join(dir, new_dir)
        try:
            os.makedirs(path_name_raw)
        except FileExistsError:
            print("Directory " , path_name_raw ,  " already exists")

        # create folder for processed data
        new_dir = os.path.join(self.identifier, '01_processed')
        path_name_processed = os.path.join(dir, new_dir)
        try:
            os.makedirs(path_name_processed)
        except FileExistsError:
            print("Directory " , path_name_processed ,  " already exists")

        # create folder for training data
        new_dir = os.path.join(self.identifier, '02_training')
        path_name_train = os.path.join(dir, new_dir)
        try:
            os.makedirs(path_name_train)
        except FileExistsError:
            print("Directory " , path_name_train ,  " already exists")

        return path_name_raw, path_name_processed, path_name_train

    def get_previous_frame_corner_pts(self, i):
        if 0 == i:
            return None
        else:
            dir = self.environment_folder
            prev_id = self.create_identifier(i-1)
            prev_dir = os.path.join(prev_id, '00_RAW')
            prev_path = os.path.join(dir, prev_dir)

            filepath = os.path.join(prev_path, '_toolpath.json')
            with open(filepath, 'r') as f:
                prev_data = json.load(f)

            prev_frame_corner_pts = [prev_data['frame_corner_pts']['0'],
                                     prev_data['frame_corner_pts']['1'],
                                     prev_data['frame_corner_pts']['2'],
                                     prev_data['frame_corner_pts']['3']]
            return prev_frame_corner_pts


if __name__ == "__main__":

    pass
