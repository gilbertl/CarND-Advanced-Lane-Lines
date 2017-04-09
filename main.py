import cv2
import glob
import matplotlib.pyplot as plt
import numpy as np
from moviepy.editor import VideoFileClip

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
    original_clip = VideoFileClip('project_video.mp4')
    original_clip = original_clip.cutout(5, 50)
    clip_with_overlays = original_clip.fl_image(
            lambda img: process_img(img, camMatrix, distCoeffs))
    clip_with_overlays.write_videofile('output_images/project_video.mp4', audio=False)

def process_img(img, camMatrix, distCoeffs):
    # Apply a distortion correction to raw images.
    undistorted_img = cv2.undistort(img, camMatrix, distCoeffs)
    # Use color transforms, gradients, etc., to create a thresholded binary image.
    binary_img, color_binary = to_binary(undistorted_img)

    # Apply a perspective transform to rectify binary image ("birds-eye view").
    warped_binary_img = warp_with_perspective(binary_img)

    result = find_lines(warped_binary_img, undistorted_img)
    return result

    # Detect lane pixels and fit to find the lane boundary.
    # Determine the curvature of the lane and vehicle position with respect to center.
    # Warp the detected lane boundaries back onto the original image.
    # Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.
    #cv2.imwrite('output_images/original.jpg', img)
    #cv2.imwrite('output_images/undistorted.jpg', undistorted_img)
    #write_binary_img('output_images/color_binary.jpg', color_binary)
    #write_binary_img('output_images/binary_img.jpg', binary_img)
    #write_binary_img('output_images/warped_binary_img.jpg', warped_binary_img)

def write_binary_img(path, img):
    scaled_img = img * 255
    cv2.imwrite(path, scaled_img)

class LaneFitter():
    def __init__(self):
        self.recent_fits_left = []
        self.recent_fits_right = []
        pass

    
    def fit_next(binary_warped, debug_image=False):
        # Take a histogram of the bottom half of the image
        histogram = np.sum(binary_warped[int(binary_warped.shape[0]/2):,:], axis=0)
        if debug_image:
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
            if out_img:
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

        self.recent_fits_left.insert(0, left_fit)
        self.recent_fits_right.insert(0, right_fit)

        if debug_image:
            # Generate x and y values for plotting
            ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
            left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
            right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

            out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
            out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]
            plt.imshow(out_img)
            plt.plot(left_fitx, ploty, color='yellow')
            plt.plot(right_fitx, ploty, color='yellow')
            plt.xlim(0, 1280)
            plt.ylim(720, 0)
            plt.savefig('output_images/find_lines.png')


def find_lines(binary_warped, undist):
    # Take a histogram of the bottom half of the image
    histogram = np.sum(binary_warped[int(binary_warped.shape[0]/2):,:], axis=0)
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

    # Generate x and y values for plotting
    ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

    out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
    out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]
    plt.imshow(out_img)
    plt.plot(left_fitx, ploty, color='yellow')
    plt.plot(right_fitx, ploty, color='yellow')
    plt.xlim(0, 1280)
    plt.ylim(720, 0)
    plt.savefig('output_images/find_lines.png')

    # Define conversions in x and y from pixels space to meters
    #ym_per_pix = 30/720 # meters per pixel in y dimension
    #xm_per_pix = 3.7/700 # meters per pixel in x dimension
    #y_eval = np.max(lefty)
#
    ## Fit new polynomials to x,y in world space
    #left_fit_cr = np.polyfit(lefty*ym_per_pix, leftx*xm_per_pix, 2)
    #right_fit_cr = np.polyfit(righty*ym_per_pix, rightx*xm_per_pix, 2)
    ## Calculate the new radii of curvature
    #left_curverad = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
    #right_curverad = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])
    ## Now our radius of curvature is in meters
    #print(left_curverad, 'm', right_curverad, 'm')

    # Create an image to draw the lines on
    warp_zero = np.zeros_like(binary_warped).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))

    src = np.float32(
            [(680, 444),
             (1088, 718), 
             (216, 718),
             (602, 444),])

    dst = np.float32(
            [(900, 0),
             (900, 720),
             (150, 720),
             (150, 0)])
    Minv = cv2.getPerspectiveTransform(dst, src)

    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    newwarp = cv2.warpPerspective(color_warp, Minv, (color_warp.shape[1], color_warp.shape[0])) 
    # Combine the result with the original image
    result = cv2.addWeighted(undist, 1, newwarp, 0.3, 0)
    return result
    #plt.clf()
    #plt.imshow(result)
    #plt.savefig('output_images/lane_overlay.png')


def calc_center_offset(left_x, right_x, total_x):
    xm_per_pix = 3.7/700 # meters per pixel in x dimension
    return abs((right_x - left_x)/2 - total_x/2) * xm_per_pix


def warp_with_perspective(img):
    src = np.float32(
            [(680, 444),
             (1088, 718), 
             (216, 718),
             (602, 444),])

    #src = np.float32(
            #[(749, 492), 
             #(1015, 661), 
             #(287, 661), 
             #(538, 491)])
    #dst = np.float32(
            #[(1015, 492),
             #(1015, 661),
             #(287, 661),
             #(287, 492)])
    dst = np.float32(
            [(900, 0),
             (900, 720),
             (150, 720),
             (150, 0)])
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
