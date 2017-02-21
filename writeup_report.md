#**Behavioral Cloning**

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./center.jpg "Center Image"
[image2]: ./examples/placeholder.png "Grayscaling"
[image3]: ./examples/placeholder_small.png "Recovery Image"
[image4]: ./examples/placeholder_small.png "Recovery Image"
[image5]: ./examples/placeholder_small.png "Recovery Image"
[image6]: ./examples/placeholder_small.png "Normal Image"
[image7]: ./examples/placeholder_small.png "Flipped Image"

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.

---
###Files Submitted & Code Quality

####1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network
* writeup_report.md or writeup_report.pdf summarizing the results

####2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing
```sh
python drive.py model.h5
```

####3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

###Model Architecture and Training Strategy

####1. An appropriate model architecture has been employed

I choose to implemented this model based on [NVDIA's paper](https://devblogs.nvidia.com/parallelforall/deep-learning-self-driving-cars/)
I added a dropout layer after flatten in order to avoid overfitting.
And I also added a 'relu' activation layer after each full connected layer since it improves my accuracy a lot while in [this video](https://www.youtube.com/watch?v=rpxZ87YFg0M&index=3&list=PLAwxTw4SYaPkz3HerxrHlu1Seq8ZA7-5P), David Silver didn't add this.

####2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting (model.py lines 75).

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 34). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

####3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 82).

####4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road ...

For details about how I created the training data, see the next section.

###Model Architecture and Training Strategy

####1. Solution Design Approach

My first step was to use a convolution neural network model similar to the [NVDIA's](https://devblogs.nvidia.com/parallelforall/deep-learning-self-driving-cars/) I thought this model might be appropriate because it has a similar camera setting with this one.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting.

To combat the overfitting, I modified the model to add the dropout layer so that both mean squared error are very low.(loss: 0.0116 - val_loss: 0.0107)


The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track, like the turn around the lake, to improve the driving behavior in these cases, I re-capture the data focusing on turning around the lake.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

####2. Final Model Architecture

The final model architecture (model.py lines 64-82) consisted of a convolution neural network with the following layers and layer sizes.

Here is a visualization of the architecture (note: visualizing the architecture is optional according to the project rubric)
Layer (type)                     Output Shape          Param #     Connected to
lambda_1 (Lambda)                (None, 160, 320, 3)   0           lambda_input_1[0][0]
cropping2d_1 (Cropping2D)        (None, 80, 320, 3)    0           lambda_1[0][0]
convolution2d_1 (Convolution2D)  (None, 38, 158, 24)   1824        cropping2d_1[0][0]
convolution2d_2 (Convolution2D)  (None, 17, 77, 36)    21636       convolution2d_1[0][0]
convolution2d_3 (Convolution2D)  (None, 7, 37, 48)     43248       convolution2d_2[0][0]
convolution2d_4 (Convolution2D)  (None, 5, 35, 64)     27712       convolution2d_3[0][0]
convolution2d_5 (Convolution2D)  (None, 3, 33, 64)     36928       convolution2d_4[0][0]
flatten_1 (Flatten)              (None, 6336)          0           convolution2d_5[0][0]
dropout_1 (Dropout)              (None, 6336)          0           flatten_1[0][0]
dense_1 (Dense)                  (None, 100)           633700      dropout_1[0][0]
dense_2 (Dense)                  (None, 50)            5050        dense_1[0][0]
dense_3 (Dense)                  (None, 10)            510         dense_2[0][0]
dense_4 (Dense)                  (None, 1)             11          dense_3[0][0]

####3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![center][image1]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to recover from side line.

Then I repeated this process on track two in order to get more data points.

To augment the data sat, I also flipped images and angles thinking that this would balance left turn and right turn image data set.

I also used the left and right image, and applied 0.25 correction to the angle.

After the collection process, I had 51234 number of data points. I then preprocessed this data by normalized them to -0.5 to 0.5, and cropped them by removing top 60 pixels and bottom 20 pixels.


I finally randomly shuffled the data set and put 20% of the data into a validation set.

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 5. I used an adam optimizer so that manually training the learning rate wasn't necessary.
