import ktb
import cv2
import os
import open3d as o3d
import numpy as np
from compas.geometry import Frame, Transformation, Scale
import compas.utilities as util
from .raster_utils import displayArray


HERE = os.path.dirname(__file__)

scan_pos_tcp = {'x':419.86, 'y': 53.37, 'z': 484.20, 'rx': 2.24, 'ry': -2.2152, 'rz': 0.0207}
scan_pos = {'base': -16.00, 'shoulder': -111.95, 'elbow': 82.89, 'wrist1': -61.91, 'wrist2': -90.10, 'wrist3': -16.29}

crop_idx = {'xStart': 64, 'xEnd':446, 'yStart':74, 'yEnd':314}

pcl_corner_pts = {'pt0': [650.832177109776,-407.279125083946,-220.620876741041],
                'ptx': [650.832177109776,-400.852772776222,278.939605285324],
                'pty': [650.832177109776,387.866124605473,286.153706626653]}

tcp_len = 214.0
robot_corner_pts = {'pt0': [284.70, -398.30, -63.80 - tcp_len],
                'ptx': [785.00, -398.30, -63.80 - tcp_len],
                'pty': [785.00, 400.25, -63.80 - tcp_len]}

def move_to_scan_position():
    pass

def crop_sandbed(matrix):
    return matrix[crop_idx['yStart']:crop_idx['yEnd'], crop_idx['xStart']:crop_idx['xEnd']]

def transform_pointcloud(pcl):
    pcl_sandbed = crop_sandbed(pcl)

    xyz = pcl_sandbed.reshape((pcl_sandbed.shape[0] * pcl_sandbed.shape[1], 3))

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)

    pcl_frame = Frame.from_points(pcl_corner_pts['pt0'], pcl_corner_pts['ptx'], pcl_corner_pts['pty'])
    robot_frame = Frame.from_points(robot_corner_pts['pt0'], robot_corner_pts['ptx'], robot_corner_pts['pty'])

    S = Scale.from_factors([1000., 1000., 1000.])
    T = Transformation.from_frame_to_frame(pcl_frame, robot_frame)

    pcd.transform(T*S)

    return pcd

def fill_zero_values(img):
    mask = (img == 0)
    mask = mask * 1
    mask = mask.astype(np.uint8)
    return cv2.inpaint(img,mask,3,cv2.INPAINT_TELEA)

def remap_depth(depth, low=100., high=150.):
    average_sand_heigt = np.mean(depth)
    base = np.zeros(depth.shape)
    base[base==0] = average_sand_heigt + low

    height_map = base - depth
    height_remap = util.remap_values(height_map, target_min=0., target_max=255., original_min=0., original_max=high)

    return np.array(height_remap).reshape(depth.shape)

def array2img(array):
    return array.astype(np.uint8)

def remove_noise(array2d):
    img = array2img(array2d)
    return cv2.fastNlMeansDenoising(img,None,2,7,15)

def get_heigt_map(depth_img):
    # crop to dimensions of sandbox
    depth_map = crop_sandbed(depth_img)
    # fill zero values
    depth_map = fill_zero_values(depth_map)
    # remap to height map sand
    height_map = remap_depth(depth_map)
    # remove noise and return map
    return remove_noise(height_map)

def scan():
    """
    Get RAW scan data from Kinect

    Returns
    -------
        pcl : numpy.array
            raw point cloud from kinect 512x424x3
        depth : numpy.array
            depth image 512x424
        color : numpy.array
            color image 512x424
    """
    move_to_scan_position()
    k = ktb.Kinect()
    return k.get_ptcld(), k.get_frame(ktb.DEPTH), k.get_frame(ktb.COLOR)

def collect_data():
    pcl, depth_img, color_img = scan()
    pcl = transform_pointcloud(pcl)
    height_map = get_heigt_map(depth_img)
    return pcl, height_map, depth_img, color_img

if __name__ == "__main__":
    pcl, depth_img, color_img = scan()
    height_map = get_heigt_map(depth_img)
    displayArray(height_map)


