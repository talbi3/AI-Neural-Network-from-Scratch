from NeuralNetwork import NeuralNetwork
import pandas as pd
import numpy as np
import itertools

def load_data(file_path):
    # Read the CSV file
    data_df = pd.read_csv(file_path).dropna()
    
    # Extract the labels
    labels = data_df.iloc[:, -1].values
    
    # Extract the inputs
    inputs = data_df.iloc[:, :-1].values
    
    # One-hot encode the labels
    one_hot_labels = np.zeros((labels.size, 10))
    one_hot_labels[np.arange(labels.size), labels] = 1
    
    inputs = inputs.astype('float32')
    one_hot_labels = one_hot_labels.astype('int')
    
    return inputs, one_hot_labels

def calculate_score_with_params(data, arcitecture, activation_func, params):

    train_inputs, train_labels, test_inputs, test_labels = data[0], data[1], data[2], data[3]
    
    network = NeuralNetwork(layers_sizes= arcitecture,activation_func= activation_func)
    network.fit(train_inputs, train_labels,epochs= params[0],mini_batch_size= params[1],learning_rate= params[2])
    
    print(arcitecture, activation_func, params)
    return network.score(test_inputs, test_labels)

# Load the training data
train_inputs, train_labels = load_data("MNIST-train.csv")
test_inputs, test_labels = load_data("MNIST-test.csv")

split_ratio = 0.8

validtion_inputs, validtion_labels = train_inputs[int(len(train_inputs) * split_ratio):], train_labels[int(len(train_labels) * split_ratio):]
train_inputs, train_labels =train_inputs[:int(len(train_inputs) * split_ratio)], train_labels[:int(len(train_labels) * split_ratio)]

arcitecture = [[784,128,64,10,10], [784,32,16,10,10], [784,25,20,10]]
activation_func = ['leaky_relu', 'sigmoid', 'tanh']
epochs = [10,15,20]
mini_batch_sizes = [5,10]
learning_rates = [2.5 ,1e-1, 1e-3]

scores_params = list()

for arci, active_func, epoch, mini_batch_size, learning_rate in itertools.product(arcitecture, activation_func, epochs, mini_batch_sizes, learning_rates):
    scores_params.append((arci,active_func,epoch,mini_batch_size,learning_rate,calculate_score_with_params([train_inputs,train_labels,validtion_inputs,validtion_labels],arcitecture=arci,activation_func=activation_func,params= [epoch,mini_batch_size,learning_rate])))

df = pd.DataFrame(columns= ['arcitecture','active_func', 'epoch', 'mini_batch_size', 'learning_rate', 'score'], data= scores_params)

df.to_csv('MNISTScoreParams.csv')
