import random
import numpy as np
import math
import matplotlib.pyplot as plt

np.random.seed(42)

input_weight_matrix = np.array([[1.0, 1.0], [1.0, 1.0]])
output_weight_matrix = np.array([1.0, 1.0]).reshape(-1,1)
input_hidden_bias = [0] 
hidden_output_bias = [0]
input_x = [[0,0], [0,1], [1,0], [1,1]]
targets = [0, 0, 0, 1]
learning_rate = 0.1
loss_result = []
hidden = np.matrix([0,0]).T

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def feedforward(input_matrix):
    global input_weight_matrix, input_hidden_bias, output_weight_matrix, hidden_output_bias, hidden
    hidden = sigmoid(np.dot(input_weight_matrix, input_matrix) + input_hidden_bias) #input_to_hidden
    output_temp = sigmoid(np.dot(output_weight_matrix.T, hidden) + hidden_output_bias) #hidden_to output
    return output_temp

def backforward(input_matrix, output_temp, index_choose):
    global output_weight_matrix, input_weight_matrix, hidden_output_bias, input_hidden_bias, hidden
    delta_output = (output_temp - targets[index_choose]) * output_temp * (1.0 - output_temp)
    delta_hidden = output_weight_matrix * delta_output * output_temp * (1.0 - output_temp)
    #update
    output_weight_matrix -= (learning_rate * np.dot(hidden, delta_output))
    input_weight_matrix -= (learning_rate * np.dot(delta_hidden, input_matrix.T))
    hidden_output_bias -= np.sum(learning_rate * delta_output)
    input_hidden_bias -= np.sum(learning_rate * delta_hidden)
    print(hidden_output_bias)
    print(input_hidden_bias)
    print(output_weight_matrix)
    print(input_weight_matrix)
    return delta_output, delta_hidden

def loss(index_choose, targets, output_temp):
    global loss_result
    target = targets[index_choose]
    square_result = (target - output_temp.item(0, 0)) **2
    loss_result.append(square_result)
    

def predict():
    for x in input_x:
        print(feedforward(x))
    # print(output_temp.item(0,0))

def training():
    for round_index in range(4):
        index_choose = random.randint(0,3)
        index_choose = round_index % 4
        input_matrix = np.matrix(input_x[index_choose]).T
        output_temp = feedforward(input_matrix)
        loss(index_choose, targets, output_temp)
        backforward(input_matrix, output_temp, index_choose)
        # if(round_index % 1000 ==0):
            # print(round_index)
    predict()


if __name__ == "__main__":
    training()
    