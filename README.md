AI – Neural Network from Scratch

This repository contains a fully from-scratch implementation of a neural network in Python. Developed as the final project for an introductory machine learning course by Tal Bitton and Matan Moskowitz, it demonstrates how to build and train a multi-layer neural network without using any high-level ML libraries (no TensorFlow, PyTorch, etc.). The project emphasizes core machine learning concepts like custom network architecture design, various activation functions, backpropagation, and rigorous training processes. It has been applied to real-world datasets (e.g. MNIST and a breast cancer classification task) to validate the network’s performance.

Features and Implementation Details

Flexible Architecture: Implements a custom NeuralNetwork class that supports an arbitrary number of layers defined by a layers_sizes array, allowing easy customization of network depth and width.

Activation Functions: Supports multiple activation functions (sigmoid, tanh, ReLU, leaky ReLU, and softmax) that can be applied to different layers. This flexibility enables experimenting with nonlinearities and their impact on learning.

Training Algorithm: Fully implements forward propagation, backpropagation, and gradient computation from scratch. Uses Xavier (Glorot) initialization for weights and biases to promote stable signal propagation, and trains the network using mini-batch Stochastic Gradient Descent (SGD). Hyperparameters like learning rate, batch size, and number of epochs are configurable for tuning performance.

Data Preprocessing: Includes comprehensive preprocessing steps such as feature normalization (scaling inputs to a standard range), log transformation for skewed input distributions, and one-hot encoding for categorical labels. These steps improve training stability and model accuracy.

Experimentation & Hyperparameter Tuning: The project includes scripts to facilitate experimentation with different network configurations and hyperparameters. For example, you can easily adjust the number of layers/neurons, swap activation functions, or try various learning rates and batch sizes to observe their effects. Helper scripts (e.g., MBSearchParams.py, MBTestMain.py, MNISTScoreParams.py, MNISTTestMain.py) are provided to automate hyperparameter searches and testing routines on the custom dataset and the MNIST dataset.

Results: Achieved ~98% accuracy on the MNIST handwritten digit dataset, and around 80% accuracy on a custom breast cancer classification dataset (malignant vs. benign). These results demonstrate the effectiveness of the implementation despite using no high-level frameworks.
