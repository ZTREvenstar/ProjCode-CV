import numpy as np
from cv2 import cv2 as cv
from matplotlib import pyplot as plt
import time as t


def extract_feature(img1, img2, RESIZE_FACTOR=1, whether_visualize=False):

    """ SOME GLOBAL PARAMETERS """
    GOOD_POINTS_LIMITED = 0.99
    # img1 is the query image
    # img2 is the train image

    '''FEATURE EXTRACTION PART STARTS'''
    orb = cv.ORB_create()  # ORB algorithm combines FAST and BRIEF
    '''
    kp1, kp2: arrays to store key points
    des1, des2: descriptor of features found. One feature correspond to one key point
    key point 主要代表位置信息（方向，大小等），每个kp对应的descriptor，这里是32维向量，储存kp周围像素的信息
    '''
    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)

    # visualize key-points. show and store
    img_kp1 = cv.drawKeypoints(img1, kp1, None)
    img_kp2 = cv.drawKeypoints(img2, kp2, None)
    time = t.strftime("%m-%d-%H-%M", t.localtime())  # record the time for experiment convenience
    cv.imwrite("./IMGOUTPUT/showKP/show_kp1_" + time + "_.jpg", img_kp1)
    cv.imwrite("./IMGOUTPUT/showKP/show_kp2_" + time + "_.jpg", img_kp2)
    # cv.imshow('1 testing draw for key points', img_kp1)
    # cv.imshow('2 testing draw for key points', img_kp2)
    # # plt.imshow(img_kp1), plt.show()
    # # plt.imshow(img_kp2), plt.show()

    '''
    MATCH STEP
    #
    行的结果是DMatch对象的列表。该DMatch对象具有以下属性： 
    - DMatch.distance-描述符之间的距离。越低越好。 
    - DMatch.trainIdx-train描述符中的描述符索引 
    - DMatch.queryIdx-query描述符中的描述符索引,索引意思是在kp数组中的index 
    - DMatch.imgIdx-train图像的索引。
    '''
    '''
    a Brute-Force Matcher
    # kp1, kp2: arrays to store key points
    # des1, des2: descriptor of features found. One feature correspond to one key point
    # key point 主要代表位置信息（方向，大小等），每个kp对应的descriptor，这里是32维向量，储存kp周围像素的信息
    '''
    bf = cv.BFMatcher_create(cv.NORM_HAMMING, crossCheck=True)  # new config
    # 根据官方文档设置参数。 ORB应使用NORM_HAMMING, crossCheck=Ture意思是两个特征应彼此匹配而非单向
    # old version # # bf = cv.BFMatcher.create()
    matches = bf.match(des1, des2)  # instead, knnMatch returns k best matches
    matches = sorted(matches, key=lambda x: x.distance)  # sort according to the distance attribute, distance越低认为是最佳匹配

    goodPoints = []
    # select good enough pairs
    for i in range(len(matches) - 1):
        if matches[i].distance < GOOD_POINTS_LIMITED * matches[i + 1].distance:
            # 该参数越小，选择标准越严格
            ############### 这块为什么要这样设计
            goodPoints.append(matches[i])
    ########### IMPORTANT CODE goodPoints = matches[:20] if len(matches) > 20   else matches[:]
    # print(goodPoints)

    img3 = cv.drawMatches(img1, kp1, img2, kp2, goodPoints, flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
                          outImg=None)
    # drawMatches作用是绘制匹配线
    # cv.imshow("THIS IS img3", img3)
    if whether_visualize:
        temp = cv.cvtColor(img3, cv.COLOR_BGR2RGB)
        plt.imshow(temp), plt.show()
    cv.imwrite(
        "./IMGOUTPUT/show_match_newconfig_" + "gplim=" + str(GOOD_POINTS_LIMITED) + "_ResizeF=" + str(RESIZE_FACTOR)
        + ".jpg", img3)
    '''
    # match 情况很乱（不resize比resize到0.25情况好），是不是上面select good points算法有问题
    # 创matcher时，用了教程建议的config，有所改善，是否goodpoint需要只选前20来匹配？
    '''
    # 为了拿到good point pairs的值(各自的像素坐标
    # Contains only the coordinate.
    # size: num_of_goodPoints * 1 * 2
    src_pts = np.float32([kp1[m.queryIdx].pt for m in goodPoints]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in goodPoints]).reshape(-1, 1, 2)
    # first parameter -1 in .reshape(): 根据其他维度推断该维度size

    return src_pts, dst_pts
