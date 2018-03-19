# Vehicle Detection

This project provides a software pipeline to detect vehicles in a video. These are the goals of the project:

The goals / steps of this project are the following:

Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector.
Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
Estimate a bounding box for vehicles detected.

Instructions
------------

1. Read and follow the instructions in https://github.com/udacity/CarND-Term1-Starter-Kit/blob/master/README.md to set up your Conda development environment.

2. Clone this repository

> git clone https://github.com/vmonestel/CarND-Vehicle-Detection.git

3. Open the code located in the IPYthon Notebook

The project code is in a jupyter notebook (Jupyter is an Ipython notebook where you can run blocks of code and see results interactively).

To start Jupyter in your browser, use terminal to navigate to the project directory and then run the following command at the terminal prompt (be sure you've activated your Python 3 carnd-term1 environment as described in point 1):

> jupyter notebook

5. A browser window will appear and show the contents of the current directory. Click on the file called "vehicle_detector.ipynb".

6. Another browser window will appear displaying the notebook. Hit "Run" in every box to run the code.
