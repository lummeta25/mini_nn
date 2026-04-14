import numpy as np
from mini_nn.loss import MSE
import sys

class Dense:
    def __init__(self, n_neurons, n_inputs, weights = None, bias = None):
        if weights is None and bias is None:
            self.weights = np.random.randn(n_inputs, n_neurons) * 0.01
            self.bias = np.random.randn(n_neurons)
            print("done")
        elif weights is None or bias is None:
            print(weights)
            print(bias)
            print("Error: both weights and bias should be given values. Or leave both None to generate random weights and bias.")
            sys.exit()
        else:
            if weights.shape != (n_inputs, n_neurons):
                print("Error: The dimesions of the given weights should match the dimensions of [n_neurons, n_inputs]" + str(weights.shape) + str([n_neurons, n_inputs]))
                sys.exit()
            elif len(bias) != n_neurons:
                print("Error: The dimesions of the given bias should match with n_neurons. Was given:" + str(len(bias)) + str(n_neurons))
                sys.exit()
            self.weights = weights
            self.bias = bias
                
    def forward(self, input):
        self.input = input
        return input @ self.weights + self.bias
    
    def backward(self, dL, learning_rate):
        dl_dw = self.input.reshape(-1, 1) @ dL.reshape(1, -1)
        dl_db = np.sum(dL)
        dl_di = dL @ self.weights.T

        self.weights = self.weights - learning_rate * dl_dw
        self.bias = self.bias - learning_rate * dl_db

        return dl_di
# dL/dweights = self.input.T @ dL
# dL/dbias    = np.sum(dL)
# dL/dinput   = dL @ self.weights.T