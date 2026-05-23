import numpy as np
import cv2 as cv
import glob
import os

# termination criteria
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((8*13,3), np.float32)
objp[:,:2] = np.mgrid[0:8,0:13].T.reshape(-1,2)

# Arrays to store object points and image points from all the images.
# objpoints = [(0.01, 0.0), (0.09, -0.14), (-0.09, -0.14)] # 3d point in real world space
# imgpoints = [(666, 398), (620, 265), (798, 290)] # 2d points in image plane.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.

images = glob.glob(os.path.expanduser('~/franka_ws/frames/camera/calibrate/*.png'))
print(images)

for fname in images:
    img = cv.imread(fname)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    # Find the chess board corners
    ret, corners = cv.findChessboardCorners(gray, (8, 13), None)

    # If found, add object points, image points (after refining them)
    if ret == True:
        objpoints.append(objp)

        corners2 = cv.cornerSubPix(gray,corners, (11,11), (-1,-1), criteria)
        imgpoints.append(corners2)

        # Draw and display the corners
        cv.drawChessboardCorners(img, (8,13), corners2, ret)
        cv.imshow('img', img)
        cv.waitKey()

cv.destroyAllWindows()

# CALIBRATE!!!
ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
    
mean_error = 0
for i in range(len(objpoints)):
    imgpoints2, _ = cv.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
    error = cv.norm(imgpoints[i], imgpoints2, cv.NORM_L2SQR) / len(imgpoints2)
    mean_error += error

print( "total error: {}".format(np.sqrt(mean_error/len(objpoints))) )

images = glob.glob(os.path.expanduser('~/franka_ws/frames/camera/chessboard.png'))
for fname in images:
    img = cv.imread(fname)
    h,  w = img.shape[:2]
    newcameramtx, roi = cv.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))

    dst = cv.undistort(img, mtx, dist, None, newcameramtx)
    
    # crop the image
    x, y, w, h = roi
    dst = dst[y:y+h, x:x+w]
    cv.imwrite('/home/franka/franka_ws/frames/camera/calibresult.png', dst)