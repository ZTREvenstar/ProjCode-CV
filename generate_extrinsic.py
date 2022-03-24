from cv2 import cv2 as cv
import numpy as np
import math
from math import degrees

from feature_extraction import extract_feature

from matplotlib import pyplot as plt


def generate_extrinsic():
    """
    generate the extrinsic parameter --- recover camera pose transformation,
    given two images.
    """

    """ SOME GLOBAL PARAMETERS """
    visualize_img = True

    '''CONTROL TEST SET'''
    TESTSET = [['IMINPUT/iden1.jpg', 'IMINPUT/iden2.jpg']]  # case1: two almost identical images

    src = TESTSET[0][0]
    des = TESTSET[0][1]

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
    M, _ = cv.findHomography(dst_pts, src_pts, cv.RANSAC, ransacReprojThreshold=5.0)
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
    F, _ = cv.findFundamentalMat(src_pts[0:8], dst_pts[0:8], cv.FM_8POINT)
    print(F)

    '''
    Draw epipolar lines according to matched points
    for debugging
    '''
    # Find epilines corresponding to points in right image (second image) and
    # drawing its lines on left image
    lines1 = cv.computeCorrespondEpilines(dst_pts.reshape(-1, 1, 2), 2, F)
    lines1 = lines1.reshape(-1, 3)
    img3, _ = draw_epiline(img1, img2, lines1, src_pts, dst_pts)

    # Find epilines corresponding to points in left image (first image) and
    # drawing its lines on right image
    lines2 = cv.computeCorrespondEpilines(src_pts.reshape(-1, 1, 2), 1, F)
    lines2 = lines2.reshape(-1, 3)
    img4, _ = draw_epiline(img2, img1, lines2, dst_pts, src_pts)

    # showing images with epilines on them.
    if visualize_img:
        plt.subplot(121), plt.imshow(img3)
        plt.subplot(122), plt.imshow(img4)
        plt.show()

    '''======================== RECIVER POSE ================================'''
    _, MR, MT, _ = cv.recoverPose(F, src_pts[0:20], dst_pts[0:20])  # output 4 things: retval, R, t, mask
    # MR, MT are the transform from src to dst.
    print("Rotation is: ")
    print(MR)
    print("Translation is: ")
    print(MT)

    print("!!!!! Is using degree")
    # convert extrinsic matrix to Euler angle for debugging.
    print(rot_to_Euler(MR))


def draw_epiline(img1, img2, lines, pts1, pts2):
    """ img1 - image on which we draw the epilines for the points in img2
        lines - corresponding epilines """
    r, c, _ = img1.shape
    for R, pt1, pt2 in zip(lines, pts1, pts2):
        color = tuple(np.random.randint(0, 255, 3).tolist())
        x0, y0 = map(int, [0, -R[2] / R[1]])
        x1, y1 = map(int, [c, -(R[2] + R[0] * c) / R[1]])
        img1 = cv.line(img1, (x0, y0), (x1, y1), color, thickness=3)
        # don't know why, but pt1 pt2 have float coordinates
        c10, c11 = map(round, pt1[0])
        c20, c21 = map(round, pt2[0])
        # while param2 pf cv.circle requires tuple of int
        img1 = cv.circle(img1, (c10, c11), 10, color, -1)
        img2 = cv.circle(img2, (c20, c21), 10, color, -1)
    return img1, img2


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
