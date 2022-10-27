from cv2 import cv2 as cv
import numpy as np
import glob

# Defining the dimensions of checkerboard
CHECKERBOARD = (6, 9)
# directory of images
IMAGEDICT = "./IMINPUT/img_cali/1/*.jpg"

# stopping criteria
# value = (3, 30, 0.001)
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# Arrays to store object points and image points from all the images.
objpoints = []  # 3d point in real world space
imgpoints = []  # 2d points in image plane

# defining the world coordinates for 3D points
objp = np.zeros((1, CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
objp[0, :, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
prev_img_shape = None

# Extracting path of individual image stored in a given directory
images = glob.glob(IMAGEDICT)
for fname in images:
    img = cv.imread(fname)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    cv.imshow("gray", gray)
    cv.waitKey(500)

    # Find the chess board corners
    # If desired num of corners are found in the image then ret = true
    ret, corners = cv.findChessboardCorners(gray, CHECKERBOARD, cv.CALIB_CB_ADAPTIVE_THRESH +
                                            cv.CALIB_CB_FAST_CHECK + cv.CALIB_CB_NORMALIZE_IMAGE)

    """
    If desired number of corner are detected,
    we refine the pixel coordinates and display 
    them on the images of checker board
    """
    if ret == True:
        objpoints.append(objp)
        # refining pixel coordinates for given 2d points
        corners2 = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

        imgpoints.append(corners2)

        # Draw and display the corners
        cv.drawChessboardCorners(img, CHECKERBOARD, corners2, ret)
        cv.imshow("img", img)
        cv.waitKey(500)

cv.destroyAllWindows()

"""
Performing camera calibration by passing the value of 
known 3D points (objpoints) and corresponding pixel 
coordinates of the detected corners (imgpoints)
"""
ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

print("Camera matrix : ")
print(mtx, "\n")
print("dist : ")
print(dist, "\n")
print("rvecs : ")
print(rvecs, "\n")
print("tvecs : ")
print(tvecs, "\n")
