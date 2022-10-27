from cv2 import cv2 as cv
import numpy as np
import glob

CHECKERBOARD = (8, 11)

# stopping criteria
# value = (3, 30, 0.001)
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

images = glob.glob("./IMINPUT/cameraself/IMG*.jpg")
for fname in images:
    img = cv.imread(fname)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    cv.imshow("gray", gray)
    cv.waitKey(500)

    # Find the chess board corners
    # If desired num of corners are found in the image then ret = true
    ret, corners = cv.findChessboardCorners(gray, CHECKERBOARD, cv.CALIB_CB_ADAPTIVE_THRESH +
                                            cv.CALIB_CB_FAST_CHECK + cv.CALIB_CB_NORMALIZE_IMAGE)
    if ret == True:
        # refining pixel coordinates for given 2d points
        corners2 = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

        # Draw and display the corners
        cv.drawChessboardCorners(img, CHECKERBOARD, corners2, ret)
        cv.imshow("img", img)
        cv.waitKey(500)

cv.destroyAllWindows()
