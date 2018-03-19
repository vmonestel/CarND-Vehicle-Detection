**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./output_images/dataset_examples.png
[image2]: ./output_images/hog_examples.png
[image3]: ./output_images/multi_scale_1.png
[image4]: ./output_images/multi_scale_2.png
[image5]: ./output_images/example_1.png
[image6]: ./output_images/example_2.png
[image7]: ./output_images/example_3.png
[image8]: ./output_images/frame_1.png
[image9]: ./output_images/frame_1_heatmap.png
[image10]: ./output_images/frame_1_heatmap_threshold.png
[image11]: ./output_images/frame_1_labels.png
[image12]: ./output_images/frame_1_final.png
[video1]: ./project_video.mp4

### Histogram of Oriented Gradients (HOG)

#### 1. HOG features.

The code for this step is contained in the `get_hog_features()` function of the IPython notebook.

I started by reading in all the `vehicle` and `non-vehicle` images.  Here are some examples of each of the `vehicle` and `non-vehicle` classes:

![alt text][image1]

I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

Here is an example using the `RGB` color space and HOG parameters of `orientations=9`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`:

![alt text][image2]

#### 2. Color features.

I used a function `extract_features()` that concatenates the HOG features with the spatial and histogram features of the images to identify the image type.

#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

I trained a linear SVM using sklearn LinearSVC. 

#### 4. Feature extraction and training.

I tried various combinations of parameters and I focused on a couple of measurements: the feature vector length (which is related to execution time) and the test accuracy of the classifier. I started with the color space (using HOG, spatial and histogram features) and I found that the RGB space had the worst accuracy (around 96%). I tried the other color spaces (HSV, LUV, HLS, YUV, YCrCb) and I got similar results (around 97.5% to 98.8% of test accuracy); YUV and LUV gave me the best results (around 98%) so I decided to use LUV considering that it was a good number.

After defining the color space, I tried to reduce the length of the feature vector (it's size was 6108) which resulted in a high amount of time to extract the feature. I tried to use just HOG features but the accuracy of the classifier decreased, I tried to combine HOG, spatial and histogram features but I get lower accuracy, so my best option was to preserve the 3 of them in the features. 

Them, I decided to modify the HOG parameter. I increased the orientations to 12 and the pixels per cell to 10 and I got similar accuracy results so I kept them because it reduced the feature vector length to 4416 and reduced the feature extraction time.

### Sliding Window Search

#### 1. Sliding window approach. 

Since the car size varies from frame to frame (the closer the car is to the bottom of the image, the bigger it is), a multi scale window search produces better results. So a good approach is to combine different windows sizes and in different positions of the images, using smaller scales when the car will be far from the bottom and bigger scales when the car is near, as shown in the following images:

![alt text][image3]![alt text][image4]

The number, size and position of the windows were chosen empirically. Some things that I saw when chosing the windows: using a high number of scales makes harder to detect cars and it produces more false positives, using a high overlap gives better results, small scales (less than 50x50 pixel windows) gives more false positives.

#### 2. Final pipeline.

After chosing the windows implemented in `combine_windows()` and using the feature parameters explained before I got the following results, where it shows that cars are detected but also there are some false positives:

![alt text][image5]![alt text][image6]![alt text][image7]
---

### Video Implementation

#### 1. Final video output.
Here's a [link to my video result](./project_video_out.mp4)


#### 2. Filter for false positives and combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.

Here's an example result showing the detected windows, the heatmap from a frame of the video, the heatmap after applying the threshold, the result of `scipy.ndimage.measurements.label()` and the bounding boxes on the frame:

![alt text][image8]![alt text][image9]![alt text][image10]![alt text][image11]![alt text][image12]

---

### Discussion

Most of the issues I faced were related to effectively discard the false positives and the non-sense car detections. After getting a good test accuracy of the Linear SVC classifier, I came across with some issues to identify the cars. Initially, I was using around 8 different multiscale detection windows; it produces really bad predictions where in some cases the white car was not detected in several consecutive frames and in other frames it detected a lot of false positives even after applying the heat map (the heat map threshold had to be high to reduce false positives but it was difficult to get a good prediction in the video). After trying to use several window sizes (squares, rectangles) and moving the windows in the y direction, I did not get good results.

I also try to augment the data set. I implemented a function to flip the images that doubled the initial dataset size; however I did not get significant better results. So I decided to keep the initial dataset as it was and focus on the image slicing implementation.

Then, I decided to start reducing the number of multi-scale windows. It showed me better results but it was not enough. I ended with 4 multi-scale windows. I chose square windows and then I started to play with the overlap value of search_cars_windows(). I had a value of 0.5 but I found out that increasing the overlap value gave me better results, so I finally chose 0.8 por both x and y direction.

My implementation suffers some false positives at the end of the video, which were difficult to discard. A better windows searching should be used, however I was able to identify the cars on almost the complete video. I did not try another classifier because the LinearSVC gave me a high accuracy (around 98-99%) but it is worth to try another one.
