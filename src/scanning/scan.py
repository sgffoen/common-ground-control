import ktb
import cv2
import os
import open3d as o3d
import numpy as np
from compas.geometry import Frame, Transformation, Scale
import compas.utilities as util
import json
from pylibfreenect2 import setGlobalLogger
from .raster_utils import displayArray

# turn off print logging to command line interface -> to turn on comment out this line of code
setGlobalLogger(None)

# set global facts
with open('data/facts.json') as f:
    facts = json.load(f)

def move_to_scan_position():
    """move robot to scan position"""

    scan_pos = facts["scan_pos"]
    scan_pos_tcp = facts['scan_pos_tcp']

def crop_sandbed(matrix):
    """crop 2d array to size of sand only"""

    crop_idx = facts["crop_idx"]
    return matrix[crop_idx['yStart']:crop_idx['yEnd'], crop_idx['xStart']:crop_idx['xEnd']]

def get_robot_corner_pts():
    """get robot corner points from sand box with tcp corrected"""

    robot_corner_pts, tcp_len = facts["robot_corner_pts"], facts["tcp_len"]
    robot_corner_pts['pt0'][2] = robot_corner_pts['pt0'][2] - tcp_len
    robot_corner_pts['ptx'][2] = robot_corner_pts['ptx'][2] - tcp_len
    robot_corner_pts['pty'][2] = robot_corner_pts['pty'][2] - tcp_len
    return robot_corner_pts


def transform_pointcloud(pcl):
    """crop point cloud to size of sandbed and transform to robot coordinates"""

    pcl_sandbed = crop_sandbed(pcl)

    xyz = pcl_sandbed.reshape((pcl_sandbed.shape[0] * pcl_sandbed.shape[1], 3))

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)

    pcl_corner_pts = facts["pcl_corner_pts"]
    pcl_frame = Frame.from_points(pcl_corner_pts['pt0'], pcl_corner_pts['ptx'], pcl_corner_pts['pty'])

    robot_corner_pts = get_robot_corner_pts()
    robot_frame = Frame.from_points(robot_corner_pts['pt0'], robot_corner_pts['ptx'], robot_corner_pts['pty'])

    S = Scale.from_factors([1000., 1000., 1000.])
    T = Transformation.from_frame_to_frame(pcl_frame, robot_frame)

    pcd.transform(T*S)

    return pcd

def fill_zero_values(img):
    """fill zero values in depth image with interpolation"""

    mask = (img == 0)
    mask = mask * 1
    mask = mask.astype(np.uint8)
    return cv2.inpaint(img,mask,3,cv2.INPAINT_TELEA)

def remap_depth(depth, low=100., high=150.):
    """remap depth values between 0 and 255 with given high and low crop"""

    average_sand_heigt = np.mean(depth)
    base = np.zeros(depth.shape)
    base[base==0] = average_sand_heigt + low

    height_map = base - depth
    height_remap = util.remap_values(height_map, target_min=0., target_max=255., original_min=0., original_max=high)

    return np.array(height_remap).reshape(depth.shape)

def array2img(array):
    return array.astype(np.uint8)

def vflip_array(arr):
    """Flip 2d array vertical to match image coordinates with robot origin"""
    return np.flipud(arr)

def remove_noise(array2d):
    """remove noise from image"""

    img = array2img(array2d)
    return cv2.fastNlMeansDenoising(img,None,2,7,15)

def get_heigt_map(depth_img):
    """create height map from depth image"""

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
    scan_data = [ k.get_ptcld(), k.get_frame(ktb.DEPTH), k.get_frame(ktb.COLOR) ]
    # flip image to match robot coordinates
    pcl, depth_img, rgb_img = map( vflip_array(), scan_data )
    return pcl, depth_img, rgb_img

def collect_data():
    pcl, depth_img, color_img = scan()
    pcl = transform_pointcloud(pcl)
    height_map = get_heigt_map(depth_img)
    return pcl, height_map, depth_img, color_img


if __name__ == "__main__":

    pcl, depth_img, color_img = scan()
    height_map = get_heigt_map(depth_img)
    displayArray(height_map)


