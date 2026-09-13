CIFAR-10 Image Classification using CNN
Overview

This project implements a Convolutional Neural Network (CNN) using PyTorch for image classification on the CIFAR-10 dataset.

The model is trained to classify images into 10 different categories, including airplanes, automobiles, birds, cats, dogs, and trucks. After training, the model is evaluated on the test dataset to measure its classification accuracy.

Model Architecture

The CNN consists of:

3 Convolutional layers
ReLU activation functions
Max Pooling layers
Fully Connected layers
10 output classes
Training
Dataset: CIFAR-10
Batch Size: 64
Epochs: 10
Optimizer: Adam
Loss Function: Cross-Entropy Loss
Technologies
Python
PyTorch
Torchvision
CNN
Installation
pip install torch torchvision

Usage

Run the Python script:

python main.py


The dataset is downloaded automatically, followed by model training and evaluation.

Result

The model's performance is measured using classification accuracy on the CIFAR-10 test dataset.
