import numpy as np
from cv2 import cv2 as cv
from matplotlib import pyplot as plt
import matplotlib.gridspec as gridspec

GOOD_POINTS_LIMITED = 0.999
src = 'img005.jpg'
des = 'img006.jpg'

img1_3 = cv.imread(src, 1)  # query image
img2_3 = cv.imread(des, 1)  # train image

# control possible resize
RESIZE_FACTOR = 1.00
# img1_3 = cv.resize(img1_3, None, fx=RESIZE_FACTOR, fy=RESIZE_FACTOR)
# img2_3 = cv.resize(img2_3, None, fx=RESIZE_FACTOR, fy=RESIZE_FACTOR)

orb = cv.ORB_create()  # ORB algorithm combines FAST and BRIEF

'''
kp1, kp2: arrays to store key points
des1, des2: descriptor of features found. One feature correspond to one key point
key point 主要代表位置信息（方向，大小等），每个kp对应的descriptor，这里是32维向量，储存kp周围像素的信息
'''
kp1, des1 = orb.detectAndCompute(img1_3, None)
kp2, des2 = orb.detectAndCompute(img2_3, None)

# visualize keypoints. show and store
img_kp1 = cv.drawKeypoints(img1_3, kp1, None)
img_kp2 = cv.drawKeypoints(img2_3, kp2, None)
cv.imwrite("./IMGOUTPUT/show_kp1.jpg", img_kp1)
cv.imwrite("./IMGOUTPUT/show_kp2.jpg", img_kp2)
# cv.imshow('1 testing draw for key points', img_kp1)
# cv.imshow('2 testing draw for key points', img_kp2)
# # plt.imshow(img_kp1), plt.show()
# # plt.imshow(img_kp2), plt.show()

'''
match step
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
# old version # # bf = cv.BFMatcher.create()
matches = bf.match(des1, des2)  # instead, knnMatch returns k best matches
matches = sorted(matches, key=lambda x: x.distance)  # sort according to the distance attribute

goodPoints = []
# select good enough pairs
for i in range(len(matches)-1):
    if matches[i].distance < GOOD_POINTS_LIMITED * matches[i+1].distance:
        ############### 为什么要这样设计,跟官方代码不一样
        goodPoints.append(matches[i])

# goodPoints = matches[:20] if len(matches) > 20   else matches[:]
print(goodPoints)

img3 = cv.drawMatches(img1_3, kp1, img2_3, kp2, goodPoints, flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS, outImg=None)
# cv.imshow("THIS IS img3", img3)
plt.imshow(img3), plt.show()
cv.imwrite("./IMGOUTPUT/show_match_cond_newconfig_" + "gplim=" + str(GOOD_POINTS_LIMITED) + "_ResizeF=" + str(RESIZE_FACTOR)
           + ".jpg", img3)
'''
# match 情况很乱（不resize比resize到0.25情况好），是不是上面select good points算法有问题
# 创matcher时，用了教程建议的config，有所改善，是否goodpoint需要只选前20来匹配？
'''

# 为了拿到good point pairs的值，数组大小正好等于good points的数量
src_pts = np.float32([kp1[m.queryIdx].pt for m in goodPoints]).reshape(-1, 1, 2)
dst_pts = np.float32([kp2[m.trainIdx].pt for m in goodPoints]).reshape(-1, 1, 2)

'''
find the HOMOGRAPHY here,
i.e., find the 3x3 transform matrix
'''
# M, mask = cv.findHomography(dst_pts, src_pts, cv.RHO)  # old config
M, mask = cv.findHomography(dst_pts, src_pts, cv.RANSAC, ransacReprojThreshold=5.0)
'''
new config, the last parameter relates to algorithm details 
WHAT IS mask? is the mask of outliers and inliers?
mask size equals to goodpoints size
'''

h1, w1, p1 = img2_3.shape  # p: 通道数量
h2, w2, p2 = img1_3.shape
h = np.maximum(h1, h2)
w = np.maximum(w1, w2)

_movedis = int(np.maximum(dst_pts[0][0][0], src_pts[0][0][0]))  # ?????????????干什么用
imageTransform = cv.warpPerspective(img2_3, M, (w1+w2-_movedis, h))
plt.imshow(imageTransform), plt.show()

M1 = np.float32([[1, 0, 0], [0, 1, 0]])
h_1, w_1, p = img1_3.shape
dst1 = cv.warpAffine(img1_3, M1, (w1+w2-_movedis, h))
plt.imshow(dst1), plt.show()

dst_no = cv.add(dst1, imageTransform)
plt.imshow(dst_no), plt.show()

dst_target = np.maximum(dst1, imageTransform)
plt.imshow(dst_target), plt.show()


'''
Draw using matplotlib  
'''
# fig = plt.figure(tight_layout=True, figsize=(8, 10))
# gs = gridspec.GridSpec(3, 3)
# ax = fig.add_subplot(gs[0, 0])
# ax.imshow(img1_3)
# ax = fig.add_subplot(gs[0, 1])
# ax.imshow(img2_3)
# ax = fig.add_subplot(gs[0, 2])
# ax.imshow(img3)

fig = plt.figure(tight_layout=True, figsize=(8, 10))
gs = gridspec.GridSpec(3, 3)
ax = fig.add_subplot(gs[0, 1])
ax.imshow(imageTransform)
ax = fig.add_subplot(gs[0, 0])
ax.imshow(dst1)
ax = fig.add_subplot(gs[0, 2])
ax.imshow(dst_no)

fig = plt.figure(tight_layout=True, figsize=(8, 8))
gs = gridspec.GridSpec(1, 1)
ax = fig.add_subplot(gs[0, :])
ax.imshow(dst_target)

plt.show()

