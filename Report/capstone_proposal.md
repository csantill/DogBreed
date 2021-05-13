# Machine Learning Engineer Nanodegree
## Capstone Proposal
Carlos Santillan

May 7, 2021

## Proposal

For the capstone project for the Udacity Machine Learning Nanodegree, I have selected the classification of dog images using Convolutional Neural Networks (CNN).

### Domain Background

Image classification is a common Machine Learning task, for this project we will be using different ML techniques and will compare the results obtained from them.

I will use different techniques to build an image classifier that will determine the breed of the dog.

Dog breed classification is a well tested machine learning.  For example the following paper describes building CNN to classify the breed, in order to help lost dogs be returned to their owners.

https://arxiv.org/pdf/2007.11986.pdf


### Problem Statement

The purpose of this project is to evaluate different machine learning techniques, and compare and constrating. In order to do this I will use pre trained models, create a cnn from scratch and finally apply transfer learning techniques. 

I will use VGG-16 model pre trained against the ImageNet dataset to build a dog classifer. We will also create a CNN from scratch and train it with the Dog dataset, finally we wil use transfer learning to train a model 

I will use VGG-16 model pre trained against the ImageNet dataset to build a dog classifer. We will also create a CNN from scratch, finally I will use transfer learning to train a model.


### Datasets and Inputs

For this project I will be using the Standford Dog dataset, This dataset consists of 120 different dog breeds with around 150 images per breed for a total of 20,580 images. 
From this dataset I will for each different breed of dogs I will select a balanced subset of :

* 100 images for training
* 20 images for testing

Using a total of 14,400 images.

This is a popular dataset for dog breed classification models used in research. And it is also available on Kaggle Playground Prediction Competition https://www.kaggle.com/c/dog-breed-identification


The original data source is found on http://vision.stanford.edu/aditya86/ImageNetDogs/ 

* Aditya Khosla, Nityananda Jayadevaprakash, Bangpeng Yao and Li Fei-Fei. **Novel dataset for Fine-Grained Image Categorization**. First Workshop on Fine-Grained Visual Categorization (FGVC),* IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2011.*


### Solution Statement

In this project, I will use use different techniques to solve the problem. 
I will create a CNN from scratch and train it with the Dog dataset, finally I wil use transfer learning using ResNet50 and/or Inception to train a model.


### Benchmark Model

I will use the VGG-16 model as my benchmark model, the model is capable of identifying 118 dog breeds. 

### Evaluation Metrics

This is a simple classification problem, we can use classification metrics such as Accuracy, Recall and  F1 score along with confusion matrix to visualize the results. 

### Project Design

In order to accomplish the goals outlined for this project we will build a data pipeline that can perform the pre processing required for the images. Such has image resizing, converting all images to gray scale. 

Also we will need to create function that will output the evaluation metrics for all the models used so that the results of the different models can be compared. 

We will also use the data augmentation techniques such has flipping, cropping and/or rotation in order to improve the performance of the model. Time permiting I may also perform hyper-parameter tuning.

I will need to determine if GPU is required for the project or if CPU is enough to perform the required training, if a GPU is required I will use AWS Sagemaker notebook instance to perform the training on Amazon.
