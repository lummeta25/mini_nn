import numpy as np
from mini_nn.loss import MSE
import sys

class Dense:
    def __init__(self, n_neurons, n_inputs, weights = None, bias = None):
        """
            Initialize a fully connected (dense) layer.

            This layer performs a linear transformation of the input:
                output = input · weights + bias

            If `weights` and `bias` are not provided, they are initialized randomly.

            Parameters
            ----------
            n_neurons : int
                Number of neurons (output units) in the layer.
            n_inputs : int
                Number of input features to the layer.
            weights : np.ndarray, optional
                Predefined weight matrix of shape (n_inputs, n_neurons).
                If None, weights are initialized randomly.
            bias : np.ndarray, optional
                Predefined bias vector of shape (n_neurons,).
                If None, biases are initialized randomly.

            Raises
            ------
            SystemExit
                If only one of `weights` or `bias` is provided.
                If `weights` shape does not match (n_inputs, n_neurons).
                If `bias` length does not match `n_neurons`.

            Notes
            -----
            - Random weights are initialized using a small normal distribution.
            - Biases are initialized using a standard normal distribution.
        """
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
        """
            Perform the forward pass of the dense layer.

            Computes the linear transformation of the input using the layer's
            weights and bias:
                output = input · weights + bias

            Parameters
            ----------
            input : np.ndarray
                Input data of shape (batch_size, n_inputs).

            Returns
            -------
            np.ndarray
                Transformed output of shape (batch_size, n_neurons).

            Notes
            -----
            - The input is stored in `self.input` for potential use during backpropagation.
            - Matrix multiplication is performed using the `@` operator.
        """
        self.input = input
        return input @ self.weights + self.bias
    
    def backward(self, dL, learning_rate):
        """
            Perform a backward pass of the dense layer updating weights and biases.

            Parameters
            ----------
            dL : no.ndarray
                The output from running backward pass on the previous layer or loss function of the model
            learning_rate : float
                The learnign rate(alpha) used to train the network
            
            Returns
            -------
            np.ndarray
                New layer weights
            
        """
        dl_dw = self.input.reshape(-1, 1) @ dL.reshape(1, -1)
        dl_db = dL
        dl_di = dL @ self.weights.T

        self.weights = self.weights - learning_rate * dl_dw
        self.bias = self.bias - learning_rate * dl_db

        return dl_di
    
    def replace_weights(self, new_values):
        """
        Replace the weights and bias of this layer.

        Parameters
        ----------
        new_values : tuple
        A tuple of (weights, bias) where weights is an np.ndarray
        of shape (n_inputs, n_neurons) and bias is an np.ndarray
        of shape (n_neurons,).
    """
        self.weights, self.bias = new_values[0], new_values[1]
