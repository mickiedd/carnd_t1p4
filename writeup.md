[image1]: ./p1.png
[image2]: ./p2.png
[image3]: ./p3.png
[image4]: ./p4.png
[image5]: ./p5.png
[image6]: ./p6.png
[image7]: ./p7.png
[image8]: ./p8.png

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

### Camera Calibration

#### 1. Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.Apply a distortion correction to raw images.

The code for this step is contained in the first code cell of the IPython notebook located in "./p4.ipynb" .  

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result: 

![alt text][image1]

#### 2. Use color transforms, gradients, etc., to create a thresholded binary image.

The code for this step is contained in the 15st code cell of the IPython notebook located in "./p4.ipynb".  

I start by convert the undistored image from RGB channels to HLS channels, and with the l channel, I calculate the derivative in x by cv2.Sobel, and then I absolute x derivative to accentuate lines away from horizontal.Then I threshold x gradient and alose threshold color channel by the s channel.The default threash hold for these is (170, 255).At last I stack each channel together to make a combined binary image.

![alt text][image2]

#### 3. Apply a perspective transform to rectify binary image ("birds-eye view").

The code for this step is contained in the 16st code cell of the IPython notebook located in "./p4.ipynb". I chose the hardcode the source and destination points in the following manner:

```python
# transform matrix
src = np.float32([
    [680 + 32, 447],
    [1105 + 200, 720],
    [206 - 200, 720],
    [602 - 32, 447]
])
dst = np.float32(
    [
        [1280, 0],
        [1280, 720],
        [0, 720],
        [0, 0],
    ])
```

This resulted in the following source and destination points:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 712, 447      | 1280, 0        | 
| 1305, 720      | 1280, 720      |
| 6, 720     | 0, 720      |
| 570, 447      | 0, 0        |

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

![alt text][image3]

![alt text][image4]

#### 4. Detect lane pixels and fit to find the lane boundary.

The code for this step is contained in the 22st and 23st code cells of the IPython notebook located in "./p4.ipynb".

I start by take a historgram of the bottom half of the warped binary image.Then I try to find the peak of the left and right halves of the histogram, I can use that as a starting point for where to search for the lines, from that point I can use a sliding window, placed around the line centers, to find and follow the lines up to the top of the frame and identify the x and y positions of all nonzero pixels in the image.After generated all the nonezero pixels in the lists, I concatenate the arrays of indices, and extract left and right line pixels positions to fit a second order polynomial to each line.

![alt text][image7]![alt text][image8]

#### 5. Determine the curvature of the lane and vehicle position with respect to center.

The code for this step is contained in the 10st code cell of the IPython notebook located in "./p4.ipynb".

The equation for radius of curvature is:
![alt text][image5]

First I calculate the center of the output image, and then I calculate the center of the boundary, at last I get the offset of these two center, that is the vehicle position with respect to center.

#### 6. Warp the detected lane boundaries back onto the original image.

The code for this step is contained in the 11st code cell of the IPython notebook located in "./p4.ipynb".  Here is an example of my result on a test image:

![alt text][image6]



### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.

To demonstrate this step, I will describe how I apply the distortion correction to one of the test images like this one:
![alt text][image1]

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

I used a combination of color and gradient thresholds to generate a binary image (thresholding steps in the 15st code cell of the IPython notebook located in "./p4.ipynb").  Here's an example of my output for this step.  (note: this is not actually from one of the test images)

![alt text][image2]

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for this step is contained in the 16st code cell of the IPython notebook located in "./p4.ipynb". I chose the hardcode the source and destination points in the following manner:

```python
# transform matrix
src = np.float32([
    [680 + 32, 447],
    [1105 + 200, 720],
    [206 - 200, 720],
    [602 - 32, 447]
])
dst = np.float32(
    [
        [1280, 0],
        [1280, 720],
        [0, 720],
        [0, 0],
    ])
```

This resulted in the following source and destination points:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 712, 447      | 1280, 0        | 
| 1305, 720      | 1280, 720      |
| 6, 720     | 0, 720      |
| 570, 447      | 0, 0        |

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

![alt text][image4]

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

When I got a warped binary image

The code for this step is contained in the 22st code cell of the IPython notebook located in "./p4.ipynb". 

Then I did some other stuff and fit my lane lines with a 2nd order polynomial kinda like this:

![alt text][image5]

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

The code for this step is contained in the 10st code cell of the IPython notebook located in "./p4.ipynb". 

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

The code for this step is contained in the 11st code cell of the IPython notebook located in "./p4.ipynb".  Here is an example of my result on a test image:

![alt text][image6]

---

### Pipeline (video)

#### 1. Here is my final output video.

Here's a [link to my video result](./output_video.mp4)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.  
