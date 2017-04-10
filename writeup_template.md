**Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1]: ./output_images/0.000_undistorted.jpg "Undistorted"
[image2]: ./output_images/0.000_binary_img.jpg "Binary threshold"
[image3]: ./output_images/0.000_warped_binary_img.jpg "Warped Binary Example"
[image4]: ./output_images/0.000_lane_overlay.png "Overlay and measurements"
[image5]: ./examples/color_fit_lines.jpg "Fit Visual"
[image6]: ./examples/example_output.jpg "Output"
[video1]: ./project_video.mp4 "Video"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points
###Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
###Writeup / README

####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Advanced-Lane-Lines/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!
###Camera Calibration

####1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step is contained in calc_cam_matrix function inside main.py.

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result: 

![alt text][image1]

###Pipeline (single images)

####1. Provide an example of a distortion-corrected image.
To demonstrate this step, I will describe how I apply the distortion correction to one of the test images like this one:
![alt text][image1]
####2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.
I used a combination of color and gradient thresholds to generate a binary image (thresholding steps at in method `to_binary`).  Here's an example of my output for this step. 

![alt text][image2]

####3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for my perspective transform includes a function called `warp_with_perspective()`, which appears in lines 1 through 8 in the file `example.py` (output_images/examples/example.py) (or, for example, in the 3rd code cell of the IPython notebook).  The `warper()` function takes as inputs an image (`img`), as well as source (`src`) and destination (`dst`) points.  I chose the hardcode the source and destination points in the following manner:

```
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

```
![alt text][image3]


####4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

Then I fit my lane lines with a 2nd order polynomial like this:

![alt text][image5]

####5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

I calculated the radius of curvature in `curverads()` by taking the pixels identified as part of the lanes, converting them into worldspace, and calculating an approximate radius from there.

I then calculated the vehicle's center offset by calculating the calculating the midpoint between the two lanes' position, subtracing that from the middle of the image, and converting it back into world space.

This assumes the vehicle's camera is centered.

####6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented this step in lines # through # in my code in `yet_another_file.py` in the function `map_lane()`.  Here is an example of my result on a test image:

![alt text][image4]

---

###Pipeline (video)

####1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./output_images/project_video.mp4)

---

###Discussion

####1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

My pipeline has a few things that can be improved:

#####1. Identifying lanes in shadows*
More time can be spent fine tuning the sobel gradients and HLS thresholds used to extract the lane pixels. Right now, my pipeline is thrown off by lanes that are hidden under shadows or lanes that look similar to their surrounding environment.

#####2. Avoiding bad lane fits
Currently, I check the slope of the two lanes at a y position near the car to see if they're roughly parellel. Much more can be done though to filter out bad lane fits. For example, I can check if the two lanes are too close or not close enough to each other. I can also check if the lanes are curving towards the same direction.

#####3. Smoothing
I can also average out the recent fits to create less stuttering overlay. Besides aesthestics, this approach will avoid scenarios where the lanes jump quickly to a relatively poor fit.


