import cv2
import numpy as np
import ktb
import open3d as o3d
if __name__=="__main__":
    from scan import scan
else:
    from .scan import scan


__IMG__ = scan(type='rgb').astype(np.uint8)
__IMG_ZOOM__ = cv2.pyrUp(__IMG__, dstsize=(2 * 512, 2 * 424))
__CORNER_PTS__ = []


def click_event(event, x, y, flags, params):

    # checking for left mouse clicks
    if event == cv2.EVENT_LBUTTONDOWN:
        # correct scaling of image by factor 2
        x = int(x/2)
        y = int(y/2)
        if len(__CORNER_PTS__) < 4:
            __CORNER_PTS__.append([x, y])
        else:
            print('all corner points are already saved')

        # displaying the coordinates
        # on the image window
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(__IMG_ZOOM__, str(x) + ',' +
                    str(y), (x, y), font,
                    1, (255, 0, 0), 2)
        cv2.imshow('image', __IMG_ZOOM__)


def skew_correction():
    pts1 = np.float32(__CORNER_PTS__)
    pts2 = np.float32([[0,0],[830,0],[0,540],[830,540]]) # / ([[0,0],[830,0],[0,540],[830,540]])
    M = cv2.getPerspectiveTransform(pts1,pts2)
    dst = cv2.warpPerspective(__IMG__, M, (830,540)) #(830,540)

    cv2.imshow("crop", dst)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return M


def img_calibration():

    # displaying the image
    cv2.imshow('image', __IMG_ZOOM__)

    cv2.setMouseCallback('image', click_event)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
    try:
        print("TOP LEFT: {}\nTOP RIGHT: {}\nBOTTOM LEFT: {}\nBOTTOM RIGHT: {}". format(__CORNER_PTS__[0],
                                                                                __CORNER_PTS__[1],
                                                                                __CORNER_PTS__[2],
                                                                                __CORNER_PTS__[3],))
        M = skew_correction()

    except:
        print("ATTENTION: Calibration unsuccesful (4 points are required)")

    return M

def depthMatrixToPointCloudPos(z, camera_params, scale=1):

    C, R = np.indices(z.shape)

    R = np.subtract(R, camera_params['cx'])
    R = np.multiply(R, z)
    R = np.divide(R, camera_params['fx'] * scale)

    C = np.subtract(C, camera_params['cy'])
    C = np.multiply(C, z)
    C = np.divide(C, camera_params['fy'] * scale)

    return np.column_stack((z.ravel() / scale, R.ravel(), -C.ravel()))


def get_ptcld(depth, roi=None, scale=1000, colorized=False):
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
    DEPTH_SHAPE = (int(512), int(424), int(4))
    # Get undistorted (and registered) frames

    undistorted = depth

    camera_params = get_intrinsic_params()
    # Get point cloud
    ptcld = depthMatrixToPointCloudPos(undistorted, camera_params, scale=scale)

    # Reshape to correct size
    #ptcld = ptcld.reshape(DEPTH_SHAPE[1], DEPTH_SHAPE[0], 3)

    return ptcld


def get_intrinsic_params():
    k = ktb.Kinect()
    return k.intrinsic_parameters


if __name__=="__main__":

    M = img_calibration()
    depth_img = scan(type='depth')
    dst = cv2.warpPerspective(depth_img, M, (830,540))
    # cv2.imshow('image', depth_img)#.astype(np.float32))
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    pcl = get_ptcld(dst)
    #ptcld = pcl.reshape(540, 830, 3)
    print(pcl.shape)
    #xyz = pcl.reshape((pcl.shape[0] * pcl.shape[1], 3))
    #xyz = pcl

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pcl)

    #o3d.io.write_point_cloud('SKEWC_pcl.ply', pcd)
