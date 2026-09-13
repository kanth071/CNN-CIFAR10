CIFAR-10 Image Classification using CNN

This project uses a Convolutional Neural Network (CNN) built with PyTorch to classify images from the CIFAR-10 dataset.

Dataset

CIFAR-10 contains 60,000 images divided into 10 classes:

Airplane
Automobile
Bird
Cat
Deer
Dog
Frog
Horse
Ship
Truck

The dataset contains:

50,000 training images
10,000 testing images

Model

The CNN consists of:

3 Convolutional layers
ReLU activation
Max Pooling layers
2 Fully Connected layers

The final layer predicts one of the 10 CIFAR-10 classes.

Training

The model is trained using:

Optimizer: Adam
Loss Function: CrossEntropyLoss
Batch Size: 64
Epochs: 10

Training loss is displayed after each epoch.

Requirements

Install PyTorch and Torchvision:

pip install torch torchvision


Run

Run the Python file:

python main.py


The CIFAR-10 dataset will be downloaded automatically.

Evaluation

After training, the model is evaluated on the test dataset and the test accuracy is calculated.

Technologies

Python
PyTorch
Torchvision
CNN
CIFAR-10
