# **Traffic Sign Recognition** 

## Writeup

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report




## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---


### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 62156
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is 32*32*3
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a bar chart showing the distribution of the training data.

![alt text][/hisogram.png]

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

The preprocessing of data involved the following steps: 
* Image conversion to grayscal
* Applying CLAHE (Contrast Limited Adaptive Histogram Equalization) see https://docs.opencv.org/3.1.0/d5/daf/tutorial_py_histogram_equalization.html
* Normalization of data to have pixel values between -1 and 1
* For the input images belonging the categories that had initially much fewer images, translation, rotation and adding Gaussian Noise


#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 RGB image   							| 
| Convolution       	| 1x1 stride, same padding, outputs 28x28x6 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x6  				|
| Convolution       	| 1x1 stride, same padding, outputs 10x10x16 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 5x5x16   				|
| flatten				| outputs 400									|
| Fully Connected		| Outputs 120									|
| RELU          		|            									|
| Fully Connected		| Outputs 84									|
| RELU          		|            									|
| Fully Connected		| Outputs 43									|

 


#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used an Adam Optimizer, with a batch size of 100, 60 epochs and learning rate = 0.001 

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of 95%
* validation set accuracy of 95% 
* test set accuracy of 94%

 

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][/images/1_modified.jpg] ![alt text][/images/2_modified.jpg] ![alt text][/images/3_modified.jpg] 
![alt text][/images/4_modified.jpg] ![alt text][/images/5_modified.jpg] ![alt text][/images/6_modified.jpg]

The image `5_modified.jpg` (Children crossing) can be tricky to predict due to low resolution, thus the features of this image should be hard to extract.


#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).


The model was able to correctly guess 5 of the 6 traffic signs, which gives an accuracy of 83.33%. The Children Crossing image sign was not properly classified as expectd.

| Image         		|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| General Caution 		| General Caution    							| 
| Roundabout mandatory 	| Roundabout mandatory                       	|
| Priority road  		| Priority road 								|
| Keep right	      	| Keep right                     				|
| Children Crossing    	| Road narrows on the right                  	|
| Speed limit (30km/h)	| Speed limit (30km/h)							|


### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)
The code for making predictions on my final model is located in the Ipython notebook.

The top 5 softmax probabilities for the children crossing sign image were: 
* > 0.6 for Road narrows on the right
* > 0.3 for Children Crossing
* < 0.1 for Pedestrians

For the other images the correct category was given a probability of 1.0 each time.




