from cv2 import cv2 as cv
import numpy as np
from feature_extraction import extract_feature
from matplotlib import pyplot as plt


def stitching_by_homography():
    """
    given two images having overlapping areas,
    src, des,
    calculate the homography H, s.t. src = H * des,
    then draw the stitched images
    """

    """ SOME GLOBAL PARAMETERS """
    visualize_img = True

    '''CONTROL TEST SET'''
    TESTSET = [['IMINPUT/cameralyy/img301.jpg', 'IMINPUT/cameralyy/img302.jpg'],  # case1: two almost identical images
               ['IMINPUT/cameralyy/img100.jpg', 'IMINPUT/cameralyy/img101.jpg'],
               ['IMINPUT/cameralyy/img200.jpg', 'IMINPUT/cameralyy/img201.jpg']]
    casenumber = 2
    c = casenumber
    src = TESTSET[c][0]
    des = TESTSET[c][1]

    src = "IMINPUT/reportuse/im1.1.jpg"
    des = "IMINPUT/reportuse/im1.2.jpg"

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

    '''This can verify the correctness of the calculated H'''
    print(M)
    warpImg = cv.warpPerspective(img1, np.array(M), dsize=(2 * img1.shape[1], 2 * img1.shape[0]),
                                 flags=cv.WARP_INVERSE_MAP)
    # warpImg =


    # warpPerspective: 透视变换。warpAffine: 仿射变换。
    # dsize is the output img size: (outputImgWidth, outputImgHigh)
    # 如果output size太小，可能会显示不全
    # 开了flags=cv.WARP_INVERSE_MAP, 函数功能为把src映射到dst, 反之，是把dst映射到src

    temp = cv.cvtColor(warpImg, cv.COLOR_BGR2RGB)
    plt.imshow(temp), plt.show()

    '''Stitch two image into one. For overlapping areas use alpha 融合'''
    ##################
    result = get_stitched_image(img1, img2, M)
    temp = cv.cvtColor(result, cv.COLOR_BGR2RGB)
    plt.imshow(temp), plt.show()


def get_stitched_image(img1, img2, M):

    w1, h1 = img1.shape[:2]
    w2, h2 = img2.shape[:2]

    # 获得图像维度
    # coordinate of 4 corners
    img1_dims = np.float32([[0, 0], [0, w1], [h1, w1], [h1, 0]]).reshape(-1, 1, 2)
    img2_dims_temp = np.float32([[0, 0], [0, w2], [h2, w2], [h2, 0]]).reshape(-1, 1, 2)

    # 获取第二幅图像的相对透视图
    img2_dims = cv.perspectiveTransform(img2_dims_temp, M)  # 执行矢量的透视矩阵变换
    # coordinate of 4 corners of img2 in img1

    # 得到结果图片的维度
    result_dims = np.concatenate((img1_dims, img2_dims), axis=0)

    # 拼接图像

    # 搞这么多，是因为图像的坐标不能为负的

    # 计算匹配点的维度
    [x_min, y_min] = np.int32(result_dims.min(axis=0).ravel() - 0.5)  # +- 0.5: 小数坐标四舍五入
    # x_min 与 y_min并行地算出了(axis=0是在算每行的最小值)，搜索numpy min的用法。
    [x_max, y_max] = np.int32(result_dims.max(axis=0).ravel() + 0.5)

    # 仿射变换后创建输出数组
    transform_dist = [-x_min, -y_min]
    transform_array = np.array([[1, 0, transform_dist[0]],
                                [0, 1, transform_dist[1]],
                                [0, 0, 1]])
    # 把图像进行平移，否则会离开图的外面

    # 扭曲图像以用于拼接
    result_img = cv.warpPerspective(img2, transform_array.dot(M),
                                    (x_max - x_min, y_max - y_min))  # third parameter: size of the output

    temp = cv.cvtColor(result_img, cv.COLOR_BGR2RGB)
    plt.imshow(temp), plt.show()

    result_img[transform_dist[1]:w1 + transform_dist[1],
               transform_dist[0]:h1 + transform_dist[0]] = img1

    return result_img


