# **Behavioral Cloning**

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/center.jpg "Center"
[image2]: ./examples/left.jpg "Left"
[image3]: ./examples/right.jpg "Right"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network
* README.md summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

I used the NVIDIA architecture as suggested in the coursework content.  It consists of five convolutional layers and four connected layers.  The code is in model.py between lines 115-129.

#### 2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting (model.py lines 124-128).

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 25-26). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer (line 131).  I chose a learning rate of 0.001 after some experimentation.

#### 4. Appropriate training data

The training data is shuffled and split (90% test, 10% validation) in lines 25-26.

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

My overall approach was to try to keep the model simple and make incremental improvements.  I did some preliminary analysis on the data and found that most of the time the car drives in a straight line.  This turned out to be a useful feature for the training set.

Initially I found that the car was heavily biased towards left turns, which would drive off the road early on.  I was able to overcome this by randomly flipping images on the vertical axis.  I found that it wasn't necessary to flip all images, just the images where there was a sharp angle.

Additionally, I found it helpful to make use of the left/right images to augment the data set for recovery.  Based on feedback from the forums, I found 0.27 to be a good value for the correction angle.

#### 2. Final Model Architecture

The final model architecture is based on the NVIDIA architecture from the coursework material.  The code is in lines 115-129.

First, the input is normalized to values between -0.5 to 0.5.  Then, the data is run through three 5x5 convolutional layers of increasing depth of 24, 36, and 48.  Then there are two 3x3 layers with a depth of 64.  RELU activation is used at each convolutional layer to avoid overfitting.

The input is then flattened and passed through four connected layers of 100, 50, 10, and 1.  There is a dropout layer of 50% between each connected layer to avoid overfitting.

I used an Adam optimizer with a learning rate of 0.001.  I used 5 epochs for the final result.

#### 3. Creation of the Training Set & Training Process

For the training data, I made two full circuits on the "easy" track.  I made a third loop going clockwise rather than counterclockwise so there would be less bias for turning left.  I also added a run from the "hard" track.  Finally, I added recovery data by driving near the side of the track and correcting towards the middle.

To augment the data set, I randomly flipped images and angles during the training phase.  I thought this would correct for a bias towards steering too much to the left (which is most of the turns when driving counterclockwise on a circular track).

I further augmented the data set by making use of the left and right cameras.  For example, here's a shot from the left, center, and right:

![alt text][image2]
![alt text][image1]
![alt text][image3]

When the car was making a sharp turn to the left or right, it's useful to add a perspective that will correct towards the center (the right camera in the case of a left turn, and the left camera in the case of a right turn).

With the data collection and augmentation, I had roughly 7,000 samples in the training set.  With a batch size of 128, each epoch processed in about 70 minutes on my MacBook.  I found 5 epochs to be sufficient for training the model to drive the "easy" track.
