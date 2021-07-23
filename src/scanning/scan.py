from compas.geometry.primitives.point import Point
import ktb
import cv2
import scipy.interpolate as inp
import open3d as o3d
import os
import numpy as np
from compas.geometry import Frame, Transformation, Scale
import compas.utilities as util
import json, codecs
from tkinter import Tk
from tkinter.filedialog import askdirectory
from tkinter.constants import S
import matplotlib.pyplot as plt
from pylibfreenect2 import setGlobalLogger

if __name__ == "__main__":
    from raster_utils import EsriGrid
    from features import Feature
    from helper import Facts
else:
    from .raster_utils import EsriGrid
    from .features import Feature
    from .helper import Facts


# turn off print logging to command line interface -> to turn on comment out this line of code
setGlobalLogger(None)

__FACTS__ = Facts().facts


class ScanData():
    def __init__(self):
        self.connect = ktb.Kinect()
        self.intrinsic_params = self.connect.intrinsic_parameters
        self.depth_shape = (int(512), int(424), int(4))
        self.rgb_scan = np.flipud(self.connect.get_frame(ktb.COLOR))
        self.depth_for_pcl = self.connect.get_frame(ktb.DEPTH)
        self.depth_scan = np.flipud(self.depth_for_pcl)
        self.ir_scan = np.flipud(self.connect.get_frame(ktb.IR))

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

        crop_ids = __FACTS__.crop_idx
        return img[crop_ids['yStart'] : crop_ids['yEnd'], crop_ids['xStart'] : crop_ids['xEnd']]

    def resize_1mmpixel(self, img):
        xmin, ymin, zmin = __FACTS__.feature_bounds['min_bound']
        xmax, ymax, zmax = __FACTS__.feature_bounds['max_bound']
        dim = (int(ymax - ymin)), (int(xmax - xmin)),  # ny, nx
        resized = cv2.resize(img, dim, interpolation = cv2.INTER_CUBIC)
        return resized

    def get_feature(self, img):
        cropped = self.crop_box(img)
        feature = self.resize_1mmpixel(cropped)
        return Feature(feature)



class PointCloud(object):
    def __init__(self, scan):
        self.depth_input = scan.depth_for_pcl
        self.intrinsic_params = scan.intrinsic_params
        self.mesh_feature = None

    def get_pointcloud_transformed(self):
        ptcld = self.get_pointcloud_raw()
        transformed = self.transform_pointcloud(ptcld)
        return transformed

    def get_feature(self):
        """crop pcl to the target size"""

        def bounding_box(points, min_x, max_x, min_y,
                        max_y, min_z, max_z):

            bound_x = np.logical_and(points[:, 0] > min_x, points[:, 0] < max_x)
            bound_y = np.logical_and(points[:, 1] > min_y, points[:, 1] < max_y)
            bound_z = np.logical_and(points[:, 2] > min_z, points[:, 2] < max_z)

            bb_filter = np.logical_and(np.logical_and(bound_x, bound_y), bound_z)

            return bb_filter

        min_bound = __FACTS__.feature_bounds['min_bound']
        max_bound = __FACTS__.feature_bounds['max_bound']

        points = self.get_pointcloud_transformed()
        points = self.pcl_flatten(points)

        points_inside = bounding_box(points, min_x=min_bound[0], max_x=max_bound[0],
                             min_y=min_bound[1], max_y=max_bound[1],
                             min_z=min_bound[2], max_z=max_bound[2])

        return points[points_inside]


    def get_pointcloud_raw(self, roi=None, scale=1, colorized=False):
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

        def depth_matrix2pointcloud(z, camera_params, scale_factor):

            C, R = np.indices(z.shape)

            R = np.subtract(R, camera_params['cx'])
            R = np.multiply(R, z)
            R = np.divide(R, camera_params['fx'] * scale_factor)

            C = np.subtract(C, camera_params['cy'])
            C = np.multiply(C, z)
            C = np.divide(C, camera_params['fy'] * scale_factor)

            return np.column_stack((z.ravel() / scale_factor, R.ravel(), -C.ravel()))

        # Get point cloud
        xyz = depth_matrix2pointcloud(undistorted, camera_params, scale)

        # Reshape to correct size
        pcl = xyz.reshape(self.depth_input.shape[1], self.depth_input.shape[0], 3)

        return pcl

    def pcl_flatten(self, pcl):
        """Flatten pcl from 2D array to 1D"""

        return pcl.reshape((pcl.shape[0] * pcl.shape[1], 3))

    def write_pointcloud(self, pcl, fname, path=None):
        pcd = self.get_o3d_format(pcl)
        if path is None:
            path = askdirectory(title='Select Folder') # shows dialog box and return the path
        path = os.path.join(path, fname + '.ply')
        print('Save .ply pointcloud in: ', path)
        o3d.io.write_point_cloud(path, pcd)

    def transform_pointcloud(self, pcl):
        """
        transform to robot coordinates

            Returns:
            --------
                pointcloud - numpy array 3D


        """

        pcd = self.get_o3d_format(pcl)

        pcl_corner_pts = __FACTS__.pcl_corner_pts
        pcl_frame = Frame.from_points(pcl_corner_pts['pt0'], pcl_corner_pts['ptx'], pcl_corner_pts['pty'])

        robot_corner_pts = __FACTS__.robot_corner_pts
        robot_frame = Frame.from_points(robot_corner_pts['pt0'], robot_corner_pts['ptx'], robot_corner_pts['pty'])

        S = Scale.from_factors([0.9765, 0.9765, 1.0], robot_frame)
        T = Transformation.from_frame_to_frame(pcl_frame, robot_frame)

        pcd.transform(S*T)

        xyz = np.asarray(pcd.points)

        # Reshape to correct size
        pcl = xyz.reshape(self.depth_input.shape[1], self.depth_input.shape[0], 3)

        return pcl

    def get_mesh_feature(self, smooth=False):

        if self.mesh_feature is None:

            pcd = self.get_o3d_format(self.get_feature())
            pcd.estimate_normals()
            mesh_out, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=12)
            #path = askdirectory(title='Select Folder') # shows dialog box and return the path
            if smooth is True:
                mesh_out = mesh_out.filter_smooth_taubin(number_of_iterations=30)

            mesh_out.compute_vertex_normals()

            # crop mesh
            # min_bound = __FACTS__.feature_bounds['min_bound']
            # max_bound = __FACTS__.feature_bounds['max_bound']
            # bbox = o3d.geometry.AxisAlignedBoundingBox(min_bound, max_bound)
            # mesh_out = mesh_smooth.crop(bbox)

            self.mesh_feature = mesh_out
            return mesh_out

        else:
            return self.mesh_feature



    def write_mesh(self, mesh, fname, path=None):
        if path is None:
            path = askdirectory(title='Select Folder') # shows dialog box and return the path
        path = os.path.join(path, fname + '.obj')
        print('Save .obj mesh in: ', path)
        o3d.io.write_triangle_mesh(path, mesh)

    def get_o3d_format(self, pcl):
        """get pointcloud in open3d format"""

        if pcl.ndim == 3:
            pcl = self.pcl_flatten(pcl)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pcl)
        return pcd


class HeightMap(PointCloud):
    def __init__(self, scan):
        super().__init__(scan)
        self.base_height = __FACTS__.base_plate_height
        self.max_height = 150.0
        self.height_values = self.get_heightmap()

    def get_heightmap(self):
        mesh = self.get_mesh_feature()
        v = mesh.vertices
        np_v = np.asarray(v)
        xmin, ymin, zmin = __FACTS__.feature_bounds['min_bound']
        xmax, ymax, zmax = __FACTS__.feature_bounds['max_bound']
        # xmin, ymin, zmin = np.amin(np_v, axis=0)
        # xmax, ymax, zmax = np.amax(np_v, axis=0)
        nx = (int(xmax - xmin))
        ny = (int(ymax - ymin))
        xi = np.linspace(xmin, xmax, nx)
        yi = np.linspace(ymin, ymax, ny)
        xi, yi = np.meshgrid(xi, yi)
        x = np_v[:,0]
        y = np_v[:,1]
        z = np_v[:,2]
        zi = inp.griddata((x, y), z, (xi, yi), method='nearest')

        return zi

    def write_height2ascii(self, path, cellsize=1.0):
        grid_data = self.height_values
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

    def height2feature(self, denoise=True):
        im = self.height2image()
        if denoise is True:
            im = self.remove_noise(im)

        return Feature(im)

    def height2image(self):
        gray = util.remap_values(self.height_values,
                                target_min=0, target_max=255,
                                original_min=self.base_height,
                                original_max=self.base_height + self.max_height)

        # reshape
        gray = np.array(gray).reshape(self.height_values.shape)
        # 8 bit
        gray = gray.astype(np.uint8)
        # 3 channels
        im = np.stack((gray,)*3, axis=-1)
        # transform (rotate, flip)
        im = np.flipud(im)
        im = np.rot90(m=im, k=1)

        return im

    def remove_noise(self, img):
        """remove noise from image"""

        return cv2.fastNlMeansDenoising(img,None,3,7,21)

    def display(self):
        h = self.height2image()
        fig = plt.imshow(h, cmap='gray')
        plt.show()


if __name__ == "__main__":

    s = ScanData()
    s.get_feature(s.rgb_scan).imshow()
    #s.display_scan(s.depth_scan)
    #p = PointCloud(s)
    #pcl = p.get_feature()
    #p.write_pointcloud(pcl, "sand_flat")
    #m = p.get_mesh_feature()
    #p.write_mesh(m, 'sandtest_mesh_1')
    #hm = HeightMap(s)
    #hm.display()
    #hf = hm.height2feature(denoise=True)
    #hf.imshow()

    #f = p.get_feature()
    #p.write_pointcloud(f, 'test_crop')
    #print(np.random.rand(10, 3).shape)

    #pcl = PointCloud(s)
    #pcl_r = pcl.get_pointcloud_raw()
    #pcl_t = pcl.get_pointcloud_transformed()
    #pcl.write_pointcloud(pcl_t, 'pcl6_base_transformed_and_scaled_21-07-2021')

    # def remove_noise(self, input):
    #     """remove noise from image"""

    #     t_min, t_max = np.min(input), np.max(input)
    #     img = input.astype(np.uint8)
    #     im = cv2.fastNlMeansDenoising(img,None,2,3,3)
    #     im = np.asarray(im).astype(np.float32)
    #     depth = util.remap_values(im, target_min=t_min, target_max=t_max, original_min=0.0, original_max=255.)
    #     depth = np.array(depth).reshape(input.shape)
    #     return depth

#self.depth_input = self.remove_noise(scan.depth_for_pcl)
