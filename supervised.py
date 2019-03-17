import random
import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt

def sigmoid(x, derivative=False):
  return 1 / (1 + np.exp(-x))

def input_to_hidden(input_x, index_choose, input_hidden_bias, input_weight_matrix): ## activation(wx + b) = hidden
    input_matrix = pd.DataFrame(input_x[index_choose])
    hidden = sigmoid(input_weight_matrix.dot(input_matrix) + input_hidden_bias)
    return hidden
    
def hidden_to_output(hidden, output_weight_matrix, hidden_output_bias): #
    output_temp = sigmoid(output_weight_matrix.dot(hidden) + hidden_output_bias)
    return output_temp

def feedforward(input_x, index_choose, input_hidden_bias, hidden_output_bias, input_weight_matrix, output_weight_matrix):
    hidden = input_to_hidden(input_x, index_choose, input_hidden_bias, input_weight_matrix)
    output_temp = hidden_to_output(hidden, output_weight_matrix, hidden_output_bias)
    return hidden, output_temp

def backforward(output_weight_matrix, output_temp, targets):
    delta_output = ((output_temp - targets[0]).dot(output_temp)).dot((1 - output_temp))
    delta_hidden = (((output_weight_matrix.T.dot(delta_output))).dot(output_temp)).dot((1 - output_temp))
    return delta_output, delta_hidden

def update_weight_bias(learning_rate, delta_output, delta_hidden, hidden, output_weight_matrix, input_weight_matrix, hidden_output_bias, input_hidden_bias, input_x, index_choose):
    input_matrix = pd.DataFrame(input_x[index_choose])
    output_weight_matrix -= (learning_rate * hidden.dot(delta_output)).T
    input_weight_matrix -= (learning_rate * delta_hidden.dot(input_matrix.T))
    hidden_output_bias -= (learning_rate * delta_output)
    input_hidden_bias -= (learning_rate * np.sum(delta_hidden))
    return input_weight_matrix, output_weight_matrix, input_hidden_bias, hidden_output_bias

def loss(index_choose, targets, output_temp, round_index):
    target = targets[index_choose]
    square_result = (target - output_temp[0][0]) * (target - output_temp[0][0])
    plt.plot([round_index],[square_result], 'ro')
    

# def predict():

def training(input_x, input_hidden_bias, hidden_output_bias, input_weight_matrix, output_weight_matrix, learning_rate, targets):
    for round_index in range(100000):
        print(round_index)
        index_choose = random.randint(0,3)
        hidden, output_temp = feedforward(input_x, index_choose, input_hidden_bias, hidden_output_bias, input_weight_matrix, output_weight_matrix)
        loss(index_choose, targets, output_temp, round_index)
        delta_output, delta_hidden = backforward(output_weight_matrix, output_temp, targets)
        input_weight_matrix, output_weight_matrix, input_hidden_bias, hidden_output_bias = update_weight_bias(learning_rate, delta_output, delta_hidden, hidden, output_weight_matrix, input_weight_matrix, hidden_output_bias, input_hidden_bias, input_x, index_choose)
    plt.savefig("fff.png")

    return "finish"


if __name__ == "__main__":
    input_weight_matrix = pd.DataFrame(np.random.rand(2, 2))
    output_weight_matrix = pd.DataFrame(np.random.rand(1, 2))
    print("----------")
    input_hidden_bias = [0] 
    hidden_output_bias = [0]
    input_x = [[0,0], [0,1], [1,0], [1,1]]
    targets = [0, 0, 0, 1]
    learning_rate = 0.1
    training(input_x, input_hidden_bias, hidden_output_bias, input_weight_matrix, output_weight_matrix, learning_rate, targets)
    