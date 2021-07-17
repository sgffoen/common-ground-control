from tkinter.constants import S
import ktb
import cv2
import scipy.interpolate as inp
import open3d as o3d
import os
import numpy as np
from compas.geometry import Frame, Transformation, Scale
import compas.utilities as util
import json
from tkinter import Tk
from tkinter.filedialog import askdirectory
import matplotlib.pyplot as plt
from pylibfreenect2 import setGlobalLogger
#from scanning.raster_utils import displayArray

# turn off print logging to command line interface -> to turn on comment out this line of code
setGlobalLogger(None)


class Feature(object):
    def __init__(self, feature):
        self.feature = feature

    def display_feature(self, height = 5):
        plt.figure(figsize=(height * (self.feature.shape[1] / self.feature.shape[0]), height))
        plt.imshow(self.feature, cmap = 'gray')
        plt.tight_layout()
        plt.axis('off')
        plt.show()

    def imshow_feature(self):
        cv2.imshow("display image", self.feature)
        k = cv2.waitKey(0)

    def save_feature(self, fname, path=None):
        if path is None:
            path = askdirectory(title='Select Folder') # shows dialog box and return the path
        path = os.path.join(path, fname + '.png')
        print('Save PNG image in: ', path)
        cv2.imwrite(path, self.feature)


class FeatureFrame(Feature):
    def __init__(self):
        super().__init__()
        self.shape = (int(256), int(256))

    def get_featureframe(self):
        pass


class ScanData():
    def __init__(self):
        self.connect = ktb.Kinect()
        self.intrinsic_params = self.connect.intrinsic_parameters
        self.depth_shape = (int(512), int(424), int(4))
        self.rgb_scan = np.flipud(self.connect.get_frame(ktb.COLOR))
        self.depth_scan = np.flipud(self.connect.get_frame(ktb.DEPTH))
        self.ir_scan = np.flipud(self.connect.get_frame(ktb.IR))
        self.pointcloud = self.PointCloud(self.depth_scan, self.intrinsic_params)

    def display_scan(self, img, height = 5):
        plt.figure(figsize=(height * (img.shape[1] / img.shape[0]), height))
        plt.imshow(img, cmap = 'gray')
        plt.tight_layout()
        plt.axis('off')
        plt.show()

    def save_scan(self, img, fname, path=None):
        if path is None:
            path = askdirectory(title='Select Folder') # shows dialog box and return the path
        path = os.path.join(path, fname + '.png')
        print('Save PNG image in: ', path)
        cv2.imwrite(path, img)

    def crop_box(self, img):
        """crop raw image to the target size"""

        with open('data/facts.json') as f:
            facts = json.load(f)

        crop_ids = facts["crop_idx"]
        return img[crop_ids['yStart'] : crop_ids['yEnd'], crop_ids['xStart'] : crop_ids['xEnd']]

    def resize_1mmpixel(self, img):
        return img

    def get_feature(self, img):
        cropped = self.crop_box(img)
        feature = self.resize_1mmpixel(cropped)
        return Feature(feature)

    def get_heightmap_feature(self):
        hm = self.HeightMap(self.resize_1mmpixel(self.crop_box(self.depth_scan)))
        return Feature(hm.heightmap)


    class PointCloud(object):
        def __init__(self, depth, intrinsic_parameters):
            self.depth_input = depth
            self.intrinsic_params = intrinsic_parameters
            self.points = self.get_pointcloud_transformed()

        def get_feature(self):
            """crop raw image to the target size"""

            with open('data/facts.json') as f:
                facts = json.load(f)

            crop_ids = facts["crop_idx"]
            return self.points[crop_ids['yStart'] : crop_ids['yEnd'], crop_ids['xStart'] : crop_ids['xEnd']]

        def get_feature_flatten(self):
            """crop raw image to the target size"""

            with open('data/facts.json') as f:
                facts = json.load(f)

            crop_ids = facts["crop_idx"]
            points = self.points[crop_ids['yStart'] : crop_ids['yEnd'], crop_ids['xStart'] : crop_ids['xEnd']]
            return points.reshape((points.shape[0] * points.shape[1], 3))

        def get_pointcloud_transformed(self, roi=None, scale=1000, colorized=False):
            '''
                get_ptcld: Returns a point cloud, generated from depth image. Units
                    are mm by default.
                ARGUMENTS:
                    roi: [x, y, w, h]
                        If specified, will crop the point cloud according to the
                        input roi. Does not accelerate runtime.
                    scale: int
                        Scales the point cloud such that ptcl = ptcl (m) / scale.
                        ie scale = 1000 returns point cloud in mm.
                    colorized: bool
                        If True, returns color matrix along with point cloud such
                        that if pt = ptcld[x,y,:], the color of that point is color
                        = color[x,y,:]
            '''
            undistorted = self.depth_input

            camera_params = self.intrinsic_params

            def depth_matrix2pointcloud(z, camera_params, scale=1):

                C, R = np.indices(z.shape)

                R = np.subtract(R, camera_params['cx'])
                R = np.multiply(R, z)
                R = np.divide(R, camera_params['fx'] * scale)

                C = np.subtract(C, camera_params['cy'])
                C = np.multiply(C, z)
                C = np.divide(C, camera_params['fy'] * scale)

                return np.column_stack((z.ravel() / scale, R.ravel(), -C.ravel()))

            # Get point cloud
            ptcld = depth_matrix2pointcloud(undistorted, camera_params, scale=scale)

            transformed = self.transform_pointcloud(ptcld)

            return transformed

        def pcl_flatten(self):
            return self.points.reshape((self.points.shape[0] * self.points.shape[1], 3))

        def write_pointcloud(self, pcl, fname, path):
            if path is None:
                path = askdirectory(title='Select Folder') # shows dialog box and return the path
            path = os.path.join(path, fname + '.ply')
            print('Save .ply pointcloud in: ', path)
            o3d.io.write_point_cloud(path, pcl)

        def transform_pointcloud(self, pcl):
            """crop point cloud to size of sandbed and transform to robot coordinates"""

            with open('data/facts.json') as f:
                facts = json.load(f)

            def get_robot_corner_pts():
                """get robot corner points from sand box with tcp corrected"""

                robot_corner_pts, tcp_len = facts["robot_corner_pts"], facts["tcp_len"]
                robot_corner_pts['pt0'][2] = robot_corner_pts['pt0'][2] - tcp_len
                robot_corner_pts['ptx'][2] = robot_corner_pts['ptx'][2] - tcp_len
                robot_corner_pts['pty'][2] = robot_corner_pts['pty'][2] - tcp_len
                return robot_corner_pts

            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(pcl)

            pcl_corner_pts = facts["pcl_corner_pts"]
            pcl_frame = Frame.from_points(pcl_corner_pts['pt0'], pcl_corner_pts['ptx'], pcl_corner_pts['pty'])

            robot_corner_pts = get_robot_corner_pts()
            robot_frame = Frame.from_points(robot_corner_pts['pt0'], robot_corner_pts['ptx'], robot_corner_pts['pty'])

            S = Scale.from_factors([1000., 1000., 1000.])
            T = Transformation.from_frame_to_frame(pcl_frame, robot_frame)

            pcd.transform(T*S)
            xyz = np.asarray(pcd.points)

            # Reshape to correct size
            pcl = xyz.reshape(self.depth_input.shape[1], self.depth_input.shape[0], 3)

            return pcl

        def get_mesh(self):
            pcd = self.get_o3d_pcd()
            pcd.estimate_normals()
            mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=12)
            #path = askdirectory(title='Select Folder') # shows dialog box and return the path

            mesh_out = mesh.filter_smooth_simple(number_of_iterations=3)
            mesh_out.compute_vertex_normals()
            v=mesh_out.vertices
            np_v = np.asarray(v)
            xmin, ymin, zmin = np.amin(np_v, axis=0)
            xmax, ymax, zmax = np.amax(np_v, axis=0)
            nx = (int(xmax - xmin))
            ny = (int(ymax - ymin))
            xi = np.linspace(xmin, xmax, nx)
            yi = np.linspace(ymin, ymax, ny)
            xi, yi = np.meshgrid(xi, yi)
            print(np_v.shape)
            x = np_v[:,0]
            y = np_v[:,1]
            z = np_v[:,2]
            zi = inp.griddata((x, y), z, (xi, yi), method='nearest')

            rows,cols = np.shape(zi)
            sizex = (xmax-xmin)/float(cols)
            sizey = (ymax-ymin)/float(rows)

            print(sizex, sizey)

            fig = plt.imshow(zi)
            plt.show()

            #o3d.io.write_triangle_mesh(os.path.join(path, "mesh_smooth3.obj"), mesh_out)

        def get_o3d_pcd(self):
            pcl = self.get_feature_flatten()
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(pcl)
            return pcd


    class HeightMap(object):
        def __init__(self, depth):
            self.depth_input = depth
            self.heightmap = self.generate_heightmap()

        def fill_zeros(self, img):
            """fill zero values in depth image with interpolation"""

            mask = (img == 0)
            mask = mask * 1
            mask = mask.astype(np.uint8)
            print(mask)
            print('#############')
            #print( cv2.inpaint(img, mask, 3, cv2.INPAINT_TELEA))


        def local_depth2height(self, low=100., high=150.):
            """remap depth values between 0 and 255 with given high and low crop"""

            average_sand_heigt = np.mean(self.depth)
            base = np.zeros(self.depth.shape)
            base[base==0] = average_sand_heigt + low

            height_map = base - self.depth
            height_remap = util.remap_values(height_map, target_min=0., target_max=255., original_min=0., original_max=high)

            self.heightmap = np.array(height_remap).reshape(self.depth_input.shape)
            return np.array(height_remap).reshape(self.depth_input.shape)

        def abs_depth2height(self, low=1011., high=756.):
            """remap depth values between 0 and 255 with given high and low crop"""

            height_remap = util.remap_values(self.depth_input, target_min=0., target_max=255., original_min=low, original_max=high)

            self.heightmap = np.array(height_remap).reshape(self.depth_input.shape)
            return np.array(height_remap).reshape(self.depth_input.shape)

        def remove_noise(self, img):
            """remove noise from image"""

            img = img.astype(np.uint8)
            return cv2.fastNlMeansDenoising(img,None,2,7,15)

        def generate_heightmap(self):
            hm = self.fill_zeros(self.depth_input)
            hm = self.abs_depth2height()
            hm = self.remove_noise(hm)
            self.heightmap = hm
            return hm





if __name__ == "__main__":

    s = ScanData()
    s.pointcloud.get_mesh()
