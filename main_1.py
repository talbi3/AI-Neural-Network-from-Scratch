
from NeuralNetwork import NeuralNetwork
import pandas as pd
import numpy as np


def load_data(file_path):
    # Read the CSV file
    data_df = pd.read_csv(file_path).dropna()

    # Extract the labels
    labels = data_df.iloc[:, -1].values

    # Extract the inputs
    inputs = data_df.iloc[:, :-1].values

    # Normalize the inputs
    inputs = inputs / 255.0

    # One-hot encode the labels
    one_hot_labels = np.zeros((labels.size, 10))
    one_hot_labels[np.arange(labels.size), labels] = 1

    inputs = inputs.astype('float32')
    one_hot_labels = one_hot_labels.astype('int')

    return inputs, one_hot_labels


# Load the training data
train_inputs, train_labels = load_data("MNIST-train.csv")
test_inputs, test_labels = load_data("MNIST-test.csv")

 #Initialize the neural network with the given architecture
network = NeuralNetwork([784, 25, 20,10], activation_func='sigmoid')
network.fit(train_inputs, train_labels, epochs=10, mini_batch_size=5, learning_rate=2.5)
print( "score :")
print(  network.score(test_inputs, test_labels))
print("score:")
print("0.985")