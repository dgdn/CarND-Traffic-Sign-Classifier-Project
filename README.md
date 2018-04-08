# **Traffic Sign Recognition** 

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:

* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report

[//]: # (Image References)

[image1]: ./examples/train-distribution.png "Visualization"
[image2]: ./examples/grayscale.jpg "Grayscaling"
[image3]: ./examples/augment.png "Augment"
[image4]: ./examples/road-work.jpg "Traffic Sign 1"
[image5]: ./examples/speed-limit-70.jpg "Traffic Sign 2"
[image6]: ./examples/bumpy-road.jpg "Traffic Sign 3"
[image7]: ./examples/stop.jpg "Traffic Sign 4"
[image8]: ./examples/yield.jpg "Traffic Sign 5"

You're reading it! and here is a link to my [project code](https://github.com/dgdn/CarND-Traffic-Sign-Classifier-Project/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

I used the numpy library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in te data set is 43

Here is an exploratory visualization of the data set. It is a bar chart showing the distribution of classes in trainnings set. The number of each class vary a lot.

![alt text][image1]

### Design and Test a Model Architecture

#### Data Processing

As a first step, I normalized the image data because normalization can speed up optimization.

I decided to generate additional data because during training the model is overfiting significantly, the training accuracy is much high than validating accuracy.

To add more data to the data set, I applied the common data augmentation tecnhiques inlude rotation, translation, zoom, and color perturbation. Keras provide a convenient method `keras.preprocessing.image.ImageDataGenerator` to do this.

Here is an example of an original image and an augmented image:

![alt text][image3] 

The difference between the original data set and the augmented data set is the following:

* Rotated 10 degree clockwise
* Shifted to left a little bit
* perturbed color

#### Model architecture

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							| 
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 28x28x64 	|
| BN					|												|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x64 				|
| Convolution 5x5	    | 1x1 stride, valid padding, outputs 10x10x64   |
| BN					|												|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 5x5x128 				    |
| Fully connected		| outputs 120  									|
| BN					|												|
| RELU					|												|
| Dropout    			|												|
| Fully connected		| outputs 80  									|
| BN					|												|
| RELU					|												|
| Dropout    			|												|
| Fully connected		| outputs 43  									|
| Softmax				|              									|
 
 The model archtechure is based on LeNet. I increased the filter size to capture more features since the traffic sign is much complex than minist data. To speed up traing, Batchnormalization is added below each convolution and fully connected layer. I use dropout as regulizer to tackle the overfit problem.

#### Train

To train the model, I used an Adam optimizer with the default learning rate 0.001. The number of epochs is 30 and the batch size is 128.

My final model results were:

* training set accuracy of 0.999
* validation set accuracy of 0.987
* test set accuracy of 0.978

I firstly tried LeNet model, it reached an accuracy of roughly 89% which was very promising. In the paper, Traffic Sign Recognition with Multi-Scale Convolutional Networks, LeCun mentioned a similar architecture that also consisted 2 convolution layers and 2 fully connected layers. The model can reached an accuracy of 98%. For the above two reasons, I deciede to use LeNet. 

Since the LeNet model was designed to recognize digit number which contains less features than the traffic sign, I tried to increase convolution filter size to 64 and 128, which help solved under fitting. Then the ajusted model can reached accuracy of 100% on training set while 93% on validation set, which strongly indicated over fitting. To improve validation accuracy, I added dropout with keep probobility of 0.5 after each fully connected layer. The validation accuracy was raised to 96% while the training accuracy was still 100%.

More work should be done to over fitting problem. I added 80,000 more images to training data using data augmentation. The validation accuracy finally reached 98.7% even 99%. The model reach an accuracy of 97.8% on test set.

### Test a Model on New Images

Here are five German traffic signs that I found on the web:

![alt text][image4] ![alt text][image5] ![alt text][image6]
![alt text][image7] ![alt text][image8]

The second image might be difficult to classify because in the traning set many speed limit signs of other type share common features, which can be hard for the model the learn.

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Road work      		| Road work   									| 
| 70 km/h		        | 70 km/h             							|
| Bumpy Road	        | Bumpy Road					 				|
| Stop Sign			    | Stop sign           							|
| Yield					| Yield											|

The model was able to correctly guess 5 of the 5 traffic signs, which gives an accuracy of 100%. This compares favorably to the accuracy on the test set of 97.8%.

#### Top 5

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.
For the 1st, 3th, 4th, and 5th image, the model predicted correctly with relatively high confidence (0.99999 probobility).
For the second image, the probability is slighlty lower than others. Here were the top 5 probabilities

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .999678     			| 70 km/h   									| 
| .000171  				| 20 km/h 										|
| .000084		        | 120 km/h	    								|
| .000062   			| 80 km/h    					 				|
| .000002   	        | Ture left           							|
