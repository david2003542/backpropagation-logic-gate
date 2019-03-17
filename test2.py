import numpy as np
import matplotlib.pyplot as plt
import random

# Controlling values
num_layers = 2
ninput_layer = 2
nhidden_layer = 2
noutput_layer = 1
learning_rate = 0.1

hidden_layer = np.zeros(nhidden_layer).reshape(-1, 1)
print(type(hidden_layer))

input_layer = np.zeros(ninput_layer).reshape(-1, 1)
output_layer = np.zeros(noutput_layer).reshape(-1, 1)

hidden_delta = np.zeros((noutput_layer, nhidden_layer))
output_delta = np.zeros(noutput_layer).reshape(-1, 1)

# Initialize weights and bias
weights = 2 * np.random.random((ninput_layer, nhidden_layer)) - 1
outweights = (2 * np.random.random(nhidden_layer) - 1).reshape(-1, 1)

bias_input = 0
bias_hidden = 0
loss = []
inputw1_changes = []
inputw2_changes = []
inputw3_changes = []
inputw4_changes = []
outputw1_changes = []
outputw2_changes = []

def Sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def ForwardPropa(input_data):
    global hidden_layer, output_layer, input_layer
    input_layer = input_data
    hidden_layer = Sigmoid(np.dot(weights, input_layer) + bias_input).reshape(-1, 1)
    output_layer = Sigmoid(np.dot(outweights.T, hidden_layer) + bias_hidden)

def LossFunc(output_data, index):
    tmp = 0.0
    for i in range(0, noutput_layer):
        tmp = ((1.0 / (2.0 * noutput_layer)) * (output_layer[i] - output_data[index]) ** 2)
    loss.append(tmp)

def BackwardPropa(output_data, index):
    global output_delta, hidden_delta

    output_delta = 1.0 / noutput_layer * (output_layer - output_data[index]) * output_layer * (1.0 - output_layer)

    #hidden_delta = hidden_layer * (1.0 - hidden_layer)
    #hidden_delta = hidden_delta * (1.0 - hidden_layer) * outweights * output_delta
    #hidden_delta = hidden_layer * ((1.0 - hidden_layer) ** 2) * outweights * output_delta
    hidden_delta = outweights * output_delta * output_layer * (1.0 - output_layer)

def UpdateWeights():
    global bias_input, bias_hidden, weights, outweights, input_layer, output_layer

    bias_input -= sum(learning_rate * hidden_delta)
    weights -= learning_rate * np.dot(hidden_delta, input_layer.reshape(-1,1).T)

    bias_hidden -= sum(learning_rate * output_delta)
    outweights -= learning_rate * hidden_layer * output_delta

def Training(epoch, input_data, output_data):
    global weights, outweights
    for i in range(0, epoch):
        data_index = random.randint(0,3)
        ForwardPropa(input_data[data_index])
        LossFunc(output_data, data_index)
        BackwardPropa(output_data, data_index)
        UpdateWeights()
        # print(hidden_layer)
        if i % 10000 == 0:
            print(i)
        #     print(weights)
        #     print(outweights)
            inputw1_changes.append(weights[0][0])
            inputw2_changes.append(weights[0][1])
            inputw3_changes.append(weights[1][0])
            inputw4_changes.append(weights[1][1])
            outputw1_changes.append(outweights[0])
            outputw2_changes.append(outweights[1])


def Predict(input_data):
    size = len(input_data)
    for index in range(0, size):
        ForwardPropa(input_data[index])
        print(input_data[index][0])
        print(input_data[index][1])
        print("-------------")
        print(output_layer[0])
        print("-------------")


x = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([0, 1, 1, 1])
Training(1, x, y)
# Predict(x)

plt.figure()
plt.plot(loss)

plt.figure()
plt.plot(inputw1_changes)
plt.plot(inputw2_changes, 'r')
plt.figure()
plt.plot(inputw3_changes, 'b--')
plt.plot(inputw4_changes, 'r--')

plt.figure()
plt.plot(outputw1_changes)
#plt.plot(outputw2_changes, 'r')
# plt.show()