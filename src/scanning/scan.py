from compas.geometry.transformations.transformations import rotate_points_xy
from compas.utilities.colors import Colormap
import ktb
import cv2
import os
import open3d as o3d
import numpy as np
from compas.geometry import Pointcloud, Translation, Frame, Transformation, Scale
import matplotlib.pyplot as plt
import compas.utilities as util

HERE = os.path.dirname(__file__)

k = ktb.Kinect()

scan_pos = {'x':419.86, 'y': 53.37, 'z': 484.20, 'rx': 2.24, 'ry': -2.2152, 'rz': 0.0207}

corner_pts = [[0.636782131069357,-0.392597960621397,0.19440351165384], [0.636782131069357,-0.427685782076738,-0.3124429828835], [0.636782131069357,0.391316443046656,-0.375302403183955], [0.636782131069357,0.41762668891836,0.143043049910789]]
pcl = k.get_ptcld()
xs, xe, ys, ye  = 64, 446, 74, 314
pcl_sand = pcl#[ys:ye, xs:xe]
pcl_shape = pcl_sand.shape
xyz = pcl_sand.reshape((pcl_shape[0] * pcl_shape[1], 3))

pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(xyz)
o3d.io.write_point_cloud("./final_pcl_pos.ply", pcd)

ppt0 = [653.722008410069,-408.025588987931,-219.376770234399]     #[635.44198715648,-427.68479004216,-316.892301035813]
pptx = [635.44198715648,-393.250460767929,196.107581600757] # y = x
ppty = [635.44198715648,416.671681035151,144.330978431156]
pcl_frame = Frame.from_points(ppt0, pptx, ppty)

rpt0 = [284.70, -398.30, -63.80-214.0]
rptx = [785.00, -398.30, -63.80-214.0]
rpty = [785.00, 400.25, -63.80-214.0]
robot_frame = Frame.from_points(rpt0, rptx, rpty)

S = Scale.from_factors([1000., 1000., 1000.])
T = Transformation.from_frame_to_frame(pcl_frame, robot_frame)

#pcd = o3d.io.read_point_cloud("data.ply")
pcd.transform(T*S)

#print(xyz)

#o3d.io.write_point_cloud("./pcl_sand.ply", pcd)








print('test\n')
depth = k.get_frame(ktb.DEPTH)

color = k.get_frame(ktb.COLOR)


def displayArray(data, height=5):
    """
    Display a 3D numpy array containing [r,g,b] values per pixel
    -
    Input /
    data = 3D numpy array
    height = maximum height to keep display ratio
    """
    plt.figure(figsize=(height * (data.shape[1] / data.shape[0]), height))
    plt.imshow(data, cmap = 'copper')
    plt.tight_layout()
    plt.axis('off')
    plt.show()
xs, xe, ys, ye  = 64, 446, 74, 314

sand = depth#[ys:ye, xs:xe]
displayArray(color)

# mask = (sand == 0)
# mask = mask * 1
# mask = mask.astype(np.uint8)

# img = cv2.inpaint(sand,mask,3,cv2.INPAINT_TELEA)

# average_sand_heigt = np.mean(img)


# base = np.zeros(img.shape)
# base[base==0] = average_sand_heigt+100.0

# height_map = base - img


# height_remap = util.remap_values(height_map, target_min=0., target_max=255., original_min=0., original_max=150.)

# height_img = np.array(height_remap).reshape(img.shape)

# height_image = height_img.astype(np.uint8)

# # noise = img.astype(np.uint8)

# dst = cv2.fastNlMeansDenoising(height_image,None,2,7,15)


# displayArray(dst)








