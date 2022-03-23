from cv2 import cv2 as cv
import numpy as np
import math
from math import degrees
from feature_extraction import extract_feature


def generate_extrinsic():
    """
    generate the extrinsic parameter --- recover camera pose transformation,
    given two images.
    """

    """ SOME GLOBAL PARAMETERS """
    src = 'IMINPUT/iden1.jpg'
    des = 'IMINPUT/iden2.jpg'
    visualize_img = True

    img1 = cv.imread(src, 1)  # query image
    img2 = cv.imread(des, 1)  # train image

    # control possible resize
    RESIZE_FACTOR = 1.00
    img1 = cv.resize(img1, None, fx=RESIZE_FACTOR, fy=RESIZE_FACTOR)
    img2 = cv.resize(img2, None, fx=RESIZE_FACTOR, fy=RESIZE_FACTOR)

    ''' return the index for matched points in two images '''
    src_pts, dst_pts = extract_feature(img1, img2,
                                       RESIZE_FACTOR=RESIZE_FACTOR, whether_visualize=visualize_img)

    ''' calculate the homography '''
    M, mask = cv.findHomography(dst_pts, src_pts, cv.RANSAC, ransacReprojThreshold=5.0)
    '''
    M is the homography matrix
    new config, the last parameter relates to algorithm details 
    WHAT IS mask? is the mask of outliers and inliers?
    ------ mask size equals to goodpoints size -----
    '''

    '''
    calculate the camera transformation matrix, which is just extrinsic parameter
    use 8-point algorithm
    '''
    extrinsic = cv.findFundamentalMat(src_pts[0:8], dst_pts[0:8], cv.FM_8POINT)
    print(extrinsic)

    _, MR, MT, _ = cv.recoverPose(extrinsic[0], src_pts[0:20], dst_pts[0:20])  # output 4 things: retval, R, t, mask
    # MR, MT are the transform from src to dst.
    print("Rotation is: ")
    print(MR)
    print("Translation is: ")
    print(MT)

    print("!!!!!")
    # convert extrinsic matrix to Euler angle for debugging.
    print(rot_to_Euler(MR))


# functions for converting extrinsic matrix to Euler angle.
def is_rotation_matrix(R):
    Rt = np.transpose(R)
    should_be_identity = np.dot(Rt, R)
    I = np.identity(3, dtype=R.dtype)
    n = np.linalg.norm(I - should_be_identity)
    return n < 1e-6


def rot_to_Euler(R):
    assert (is_rotation_matrix(R))
    sy = math.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])
    singular = sy < 1e-6
    if not singular:
        x = degrees(math.atan2(R[2, 1], R[2, 2]))
        y = degrees(math.atan2(-R[2, 0], sy))
        z = degrees(math.atan2(R[1, 0], R[0, 0]))
    else:
        x = degrees(math.atan2(-R[1, 2], R[1, 1]))
        y = degrees(math.atan2(-R[2, 0], sy))
        z = 0

    # in return, 用角度制
    return np.array([x, y, z])
