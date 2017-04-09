import cv2
import glob
import matplotlib.pyplot as plt
import numpy as np

NUM_CHESSBOARD_COLS = 9
NUM_CHESSBOARD_ROWS = 6
SX_THRESHOLD = (20, 100)
S_THRESHOLD = (170, 255)

def main():
    # img = cv2.imread('camera_cal/calibration5.jpg')
    img = cv2.imread('test_images/test5.jpg')
    # Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
    chessboard_img_paths = glob.glob('camera_cal/calibration*.jpg')
    camMatrix, distCoeffs = calc_cam_matrix(chessboard_img_paths, img.shape[:2])
    # Apply a distortion correction to raw images.
    undistorted_img = cv2.undistort(img, camMatrix, distCoeffs)
    # Use color transforms, gradients, etc., to create a thresholded binary image.
    binary_img, color_binary = to_binary(undistorted_img)

    # Apply a perspective transform to rectify binary image ("birds-eye view").
    warped_binary_img = warp_with_perspective(binary_img)

    # Detect lane pixels and fit to find the lane boundary.
    # Determine the curvature of the lane and vehicle position with respect to center.
    # Warp the detected lane boundaries back onto the original image.
    # Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.
    cv2.imwrite('output_images/original.jpg', img)
    cv2.imwrite('output_images/undistorted.jpg', undistorted_img)
    write_binary_img('output_images/color_binary.jpg', color_binary)
    write_binary_img('output_images/binary_img.jpg', binary_img)
    write_binary_img('output_images/warped_binary_img.jpg', warped_binary_img)

def write_binary_img(path, img):
    scaled_img = img * 255
    cv2.imwrite(path, scaled_img)

def warp_with_perspective(img):
    src = np.float32(
            [(749, 492), 
             (1015, 661), 
             (287, 661), 
             (538, 491)])
    dst = np.float32(
            [(1015, 492),
             (1015, 661),
             (287, 661),
             (287, 492)])
    M = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(img, M, img.shape[::-1][:2], flags=cv2.INTER_LINEAR)

def to_binary(img):
    hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
    s_channel = hls[:, :, 2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0)
    abs_sobelx = np.absolute(sobelx)
    scaled_sobel = np.uint8(255 * abs_sobelx / np.max(abs_sobelx))
    
    sxbinary = np.zeros_like(scaled_sobel)
    sxbinary[(scaled_sobel >= SX_THRESHOLD[0]) 
             & (scaled_sobel <= SX_THRESHOLD[1])] = 1

    s_binary = np.zeros_like(s_channel)
    s_binary[(s_channel >= S_THRESHOLD[0]) & (s_channel <= S_THRESHOLD[1])] = 1

    color_binary = np.dstack((np.zeros_like(sxbinary), sxbinary, s_binary))

    combined_binary = np.zeros_like(sxbinary)
    combined_binary[(s_binary == 1) | (sxbinary == 1)] = 1

    return combined_binary, color_binary




def calc_cam_matrix(chessboard_img_paths, img_2d_shape):
    objpoints, imgpoints = obj_img_points(chessboard_img_paths)
    _, camMatrix, distCoeffs, _, _ = cv2.calibrateCamera(
            objpoints, imgpoints, img_2d_shape, None, None)
    return camMatrix, distCoeffs


def obj_img_points(chessboard_img_paths):
    objpoints = []  # 3D points in real world specs
    imgpoints = []  # 2D points in image plane
    for chessboard_img_path in chessboard_img_paths:
        chessboard_img = cv2.imread(chessboard_img_path)
        objp, corners = obj_img_points_per_image(chessboard_img)
        if len(objp) and len(corners):
            objpoints.append(objp)
            imgpoints.append(corners)
    return (objpoints, imgpoints)

def obj_img_points_per_image(chessboard_img):
    # Make obj points (0, 0, 0), (1, 0, 0), (2, 0, 0), etc.
    objp = np.zeros((NUM_CHESSBOARD_ROWS * NUM_CHESSBOARD_COLS, 3), np.float32)
    objp[:,:2] = (np.mgrid[0:NUM_CHESSBOARD_COLS , 0:NUM_CHESSBOARD_ROWS].T
            .reshape(-1, 2))

    gray_chessboard_img = cv2.cvtColor(chessboard_img, cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(gray_chessboard_img, 
        (NUM_CHESSBOARD_COLS, NUM_CHESSBOARD_ROWS), None)
    if not ret:
        print('Could not find corners')
        return ([], [])
    else:
        return (objp, corners)
        

if __name__ == '__main__':
    main()
