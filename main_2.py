from NeuralNetwork import NeuralNetwork
import pandas as pd
import numpy as np


def load_data(file_path):
    # Read the CSV file
    data_df = pd.read_csv(file_path).dropna()

    # Extract the labels
    labels = data_df.iloc[:, 0].values

    # Extract the inputs
    inputs = data_df.iloc[:, 1:].values

    labels = np.array([(label.find('Pt_Ctrl_') + 1) for label in labels])

    one_hot_labels = np.zeros((labels.size, 2))
    one_hot_labels[np.arange(labels.size), labels] = 1

    return inputs, one_hot_labels


def create_dicts_for_labels(labels):
    labels_set = set(labels)

    index_to_labels_arr = np.array([])
    label_to_index_dict = dict()
    index = 0

    for label in labels_set:
        label_to_index_dict[label] = index
        index_to_labels_arr = np.append(index_to_labels_arr, label)
        index += 1

    return label_to_index_dict, index_to_labels_arr


def convert_labels_to_indexes(labels, label_dict):
    one_hot_labels = np.zeros((labels.size, 2))
    one_hot_labels[np.arange(labels.size), [label_dict[label] for label in labels]] = 1

    return one_hot_labels

def square_root(num):
    return num ** 0.5

# Load the training data
train_inputs, train_labels = load_data("MB_data_train.csv")
# label_dict, index_arr = create_dicts_for_labels(train_labels)
# train_labels = convert_labels_to_indexes(train_labels, label_dict)

print(train_labels[:5])

# max_log_train_inputs = np.log2(train_inputs).max()
train_inputs = np.log2(train_inputs + 1)
#train_inputs= square_root(train_inputs)

print(train_inputs.max(), train_inputs.min())

split_ratio = 0.8

test_inputs, test_labels = train_inputs[int(len(train_inputs) * split_ratio):], train_labels[
                                                                                int(len(train_labels) * split_ratio):]
train_inputs, train_labels = train_inputs[:int(len(train_inputs) * split_ratio)], train_labels[
                                                                                  :int(len(train_labels) * split_ratio)]

# Initialize the neural network with the given architecture
network = NeuralNetwork([1620, 512, 64, 2], activation_func='tanh')
network.fit(train_inputs, train_labels, epochs=10, mini_batch_size=5, learning_rate=1e-3)
print( "score :")
print(network.score(test_inputs, test_labels))

