import cv2
import glob
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import pickle
from moviepy.editor import VideoFileClip

NUM_CHESSBOARD_COLS = 9
NUM_CHESSBOARD_ROWS = 6
SX_THRESHOLD = (20, 100)
S_THRESHOLD = (170, 255)
SRC = np.float32(
        [(680, 444),
         (1088, 718), 
         (216, 718),
         (602, 444),])

DST = np.float32(
        [(900, 0),
         (900, 720),
         (150, 720),
         (150, 0)])

def main():
    input_video_name = sys.argv[1]
    start_time = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    end_time = int(sys.argv[3]) if len(sys.argv) > 3 else None
    debug = bool(end_time)

    img = cv2.imread('test_images/test5.jpg')
    # Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
    camMatrix, distCoeffs = None, None
    if os.path.exists('camera_matrix.pkl'):
        with open('camera_matrix.pkl', 'rb') as input:
            camMatrix, distCoeffs = pickle.load(input)
    else:
        chessboard_img_paths = glob.glob('camera_cal/calibration*.jpg')
        camMatrix, distCoeffs = calc_cam_matrix(chessboard_img_paths, img.shape[:2])

        with open('camera_matrix.pkl', 'wb') as output:
            pickle.dump([camMatrix, distCoeffs], output, pickle.HIGHEST_PROTOCOL)

    original_clip = VideoFileClip(input_video_name).subclip(start_time, end_time)
    lane_fitter = LaneFitter()
    clip_with_overlays = original_clip.fl(
            lambda gf, t: process_img(t, gf(t), camMatrix, distCoeffs, lane_fitter, debug))
    clip_with_overlays.write_videofile('output_images/' + input_video_name, audio=False)

def process_img(tsecs, img, camMatrix, distCoeffs, lane_fitter, 
        debug=False):
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    if debug:
        cv2.imwrite('output_images/' + '{0:.3f}'.format(tsecs) + 'original.jpg', img)

    # Apply a distortion correction to raw images.
    undistorted_img = cv2.undistort(img, camMatrix, distCoeffs)
    if debug:
        cv2.imwrite('output_images/' + '{0:.3f}'.format(tsecs) + '_undistorted.jpg',
                undistorted_img)

    # Use color transforms, gradients, etc., to create a thresholded binary image.
    binary_img, color_binary = to_binary(undistorted_img)
    if debug:
        write_binary_img('output_images/' + '{0:.3f}'.format(tsecs) + '_color_binary.jpg', 
                color_binary)
        write_binary_img('output_images/' + '{0:.3f}'.format(tsecs) + '_binary_img.jpg', 
                binary_img)


    # Apply a perspective transform to rectify binary image ("birds-eye view").
    warped_binary_img = warp_with_perspective(binary_img)
    if debug:
        write_binary_img('output_images/' + '{0:.3f}'.format(tsecs) + '_warped_binary_img.jpg', 
                warped_binary_img)

    # Detect lane pixels and fit to find the lane boundary.
    # Determine the curvature of the lane and vehicle position with respect to center.
    # Warp the detected lane boundaries back onto the original image.
    # Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.
    lane_fitter.fit_next(tsecs, warped_binary_img, debug)

    return draw_lanes(tsecs, lane_fitter, warped_binary_img, undistorted_img, debug)


def draw_lanes(tsecs, lane_fitter, binary_warped, undistort, debug):
    # Create an image to draw the lines on
    warp_zero = np.zeros_like(binary_warped).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

    left_info = lane_fitter.best_fit('left')
    right_info = lane_fitter.best_fit('right')

    if left_info is None or right_info is None:
        return undistort

    left_fit = left_info.fit
    right_fit = right_info.fit

    # Recast the x and y points into usable format for cv2.fillPoly()
    ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))

    Minv = cv2.getPerspectiveTransform(DST, SRC)

    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    newwarp = cv2.warpPerspective(color_warp, Minv, (color_warp.shape[1], color_warp.shape[0])) 
    # Combine the result with the original image
    result = cv2.addWeighted(undistort, 1, newwarp, 0.3, 0)
    
    cv2.putText(result, 'Curvature: {0:.4g}m'.format(
        (left_info.curverads() + right_info.curverads()) / 2), 
        (350, 100), cv2.FONT_HERSHEY_PLAIN, 3, 0, 5)
    offset = center_offset(left_info, right_info, binary_warped.shape[1])
    cv2.putText(result, 'Offset from center: {0:.4g}m'.format(offset),
        (350, 150), cv2.FONT_HERSHEY_PLAIN, 3, 0, 5)
    
    if debug:
        plt.clf()
        plt.imshow(result)
        plt.savefig('output_images/'+ '{0:.3f}'.format(tsecs) + '_lane_overlay.png')

    return result

def write_binary_img(path, img):
    scaled_img = img * 255
    cv2.imwrite(path, scaled_img)


class LaneInfo():
    def __init__(self, fit, xs, ys):
        self.fit = fit
        self.xs = xs
        self.ys = ys

    def curverads(self):
        ym_per_pix = 30/720 # meters per pixel in y dimension
        xm_per_pix = 3.7/700 # meters per pixel in x dimension
        y_eval = np.max(self.ys)

        ## Fit new polynomials to x,y in world space
        fit_cr = np.polyfit(self.ys * ym_per_pix, self.xs * xm_per_pix, 2)
        ## Calculate the new radii of curvature
        curverad = ((1 + (2*self.fit[0]*y_eval*ym_per_pix + self.fit[1])**2)**1.5) / np.absolute(2*self.fit[0])
        return curverad

def center_offset(left_info, right_info, total_x):
    y_eval = max(np.max(left_info.ys), np.max(right_info.ys))
    left_x_pos = np.polyval(left_info.fit, y_eval)
    right_x_pos = np.polyval(right_info.fit, y_eval)
    xm_per_pix = 3.7/700 # meters per pixel in x dimension
    return abs((right_x_pos + left_x_pos)/2 - total_x/2) * xm_per_pix


class LaneFitter():
    def __init__(self):
        self.recent_fits = {'left': [], 'right': []}
        pass

    def best_fit(self, side):
        return self.recent_fits[side][0] if len(self.recent_fits[side]) else None

    def fit_next(self, tsecs, binary_warped, debug=False):
        existing_left_info = self.best_fit('left')
        existing_right_info = self.best_fit('right')


        left_info, right_info = None, None
        if existing_left_info is not None and existing_right_info is not None:
            left_info, right_info = self.filter_fits(
                    self.fit_by_existing_fit(tsecs, binary_warped, 
                        existing_left_info.fit, existing_right_info.fit, debug))
        # If existing fit doesn't give us a good resut, try with window again.
        if left_info is None or right_info is None:
            if tsecs != 0 and debug:
                print('{}: existing fit sucks, resorting to window'.format(tsecs))
            left_info, right_info = self.filter_fits(
                    self.fit_by_window_search(tsecs, binary_warped, debug))

        if left_info is not None and right_info is not None:
            if len(self.recent_fits['left']) > 10:
                self.recent_fits['left'].pop()
            self.recent_fits['left'].insert(0, left_info)

            if len(self.recent_fits['right']) > 10:
                self.recent_fits['right'].pop()
            self.recent_fits['right'].insert(0, right_info)
        else:
            if debug:
                print('{}: window search sucks, resorting to previous'.format(tsecs))

    def filter_fits(self, left_right_info):
        left_info, right_info = left_right_info
        if self.are_sensible_fits(left_info.fit, right_info.fit):
            return left_info, right_info
        else:
            return None, None

    def are_sensible_fits(self, left_fit, right_fit):
        left_der = np.polyder(left_fit)
        right_der = np.polyder(right_fit)
        return abs(np.polyval(left_der, 720) - np.polyval(right_der, 720)) <= 0.3


    def fit_by_existing_fit(self, tsecs, binary_warped, left_fit, right_fit, debug=False):
        nonzero = binary_warped.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        margin = 100
        left_lane_inds = ((nonzerox > (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + left_fit[2] - margin)) & (nonzerox < (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + left_fit[2] + margin))) 
        right_lane_inds = ((nonzerox > (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + right_fit[2] - margin)) & (nonzerox < (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + right_fit[2] + margin)))  

        # Again, extract left and right line pixel positions
        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds] 
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds]

        if leftx.size == 0 or lefty.size == 0 or rightx.size == 0 or righty.size == 0:
            return None, None

        # Fit a second order polynomial to each
        left_fit = np.polyfit(lefty, leftx, 2)
        right_fit = np.polyfit(righty, rightx, 2)
        # Generate x and y values for plotting
        ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
        left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
        right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

        out_img = None
        if debug:
            # Create an output image to draw on and  visualize the result
            out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255
            # Generate x and y values for plotting
            ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
            left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
            right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

            out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
            out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]
            plt.imshow(out_img)
            plt.clf()
            plt.plot(left_fitx, ploty, color='yellow')
            plt.plot(right_fitx, ploty, color='yellow')
            plt.xlim(0, 1280)
            plt.ylim(720, 0)
            plt.savefig('output_images/' + '{0:.3f}'.format(tsecs) + '_find_lines_existing.png')

        return LaneInfo(left_fit, leftx, lefty), LaneInfo(right_fit, rightx, righty)


    def fit_by_window_search(self, tsecs, binary_warped, debug=False):
        # Take a histogram of the bottom half of the image
        histogram = np.sum(binary_warped[int(binary_warped.shape[0]/2):,:], axis=0)
        out_img = None
        if debug:
            # Create an output image to draw on and  visualize the result
            out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255
        # Find the peak of the left and right halves of the histogram
        # These will be the starting point for the left and right lines
        midpoint = np.int(histogram.shape[0]/2)
        leftx_base = np.argmax(histogram[:midpoint])
        rightx_base = np.argmax(histogram[midpoint:]) + midpoint
        # Choose the number of sliding windows
        nwindows = 9
        # Set height of windows
        window_height = np.int(binary_warped.shape[0]/nwindows)
        # Identify the x and y positions of all nonzero pixels in the image
        nonzero = binary_warped.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Current positions to be updated for each window
        leftx_current = leftx_base
        rightx_current = rightx_base
        # Set the width of the windows +/- margin
        margin = 100
        # Set minimum number of pixels found to recenter window
        minpix = 50
        # Create empty lists to receive left and right lane pixel indices
        left_lane_inds = []
        right_lane_inds = []
        # Step through the windows one by one
        for window in range(nwindows):
            # Identify window boundaries in x and y (and right and left)
            win_y_low = binary_warped.shape[0] - (window+1)*window_height
            win_y_high = binary_warped.shape[0] - window*window_height
            win_xleft_low = leftx_current - margin
            win_xleft_high = leftx_current + margin
            win_xright_low = rightx_current - margin
            win_xright_high = rightx_current + margin
            # Draw the windows on the visualization image
            if debug:
                cv2.rectangle(out_img,(win_xleft_low,win_y_low),(win_xleft_high,win_y_high),(0,255,0), 2) 
                cv2.rectangle(out_img,(win_xright_low,win_y_low),(win_xright_high,win_y_high),(0,255,0), 2) 
            # Identify the nonzero pixels in x and y within the window
            good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
            good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]
            # Append these indices to the lists
            left_lane_inds.append(good_left_inds)
            right_lane_inds.append(good_right_inds)
            # If you found > minpix pixels, recenter next window on their mean position
            if len(good_left_inds) > minpix:
                leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
            if len(good_right_inds) > minpix:        
                rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

        # Concatenate the arrays of indices
        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)

        # Extract left and right line pixel positions
        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds] 
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds] 

        # Fit a second order polynomial to each
        left_fit = np.polyfit(lefty, leftx, 2)
        right_fit = np.polyfit(righty, rightx, 2)

        if debug:
            # Generate x and y values for plotting
            ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
            left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
            right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

            out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
            out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]
            plt.clf()
            plt.imshow(out_img)
            plt.plot(left_fitx, ploty, color='yellow')
            plt.plot(right_fitx, ploty, color='yellow')
            plt.xlim(0, 1280)
            plt.ylim(720, 0)
            plt.savefig('output_images/' + '{0:.3f}'.format(tsecs) + '_find_lines_window.png')

        return LaneInfo(left_fit, leftx, lefty), LaneInfo(right_fit, rightx, righty)


def warp_with_perspective(img):
    M = cv2.getPerspectiveTransform(SRC, DST)
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
