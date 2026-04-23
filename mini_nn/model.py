from mini_nn.layers import Dense
from mini_nn.loss import MSE
import numpy as np
import pickle

class NeuralNetwork:
    """
    A sequential neural network composed of Dense layers and activations.

    Layers are applied in the order they are provided during both
    forward and backward passes.

    Parameters
    ----------
    layers : list
        Ordered list of Dense layers and activation instances.
    """
    def __init__(self, layers):
        self.layers = layers
        
        
        
    def forward(self, input):
        """
        Run a forward pass through every layer in sequence.

        Parameters
        ----------
        input : np.ndarray
            Input sample or batch.

        Returns
        -------
        np.ndarray
            Final output after all layers.
        """
        for layer in self.layers:
            input = layer.forward(input)
        # print(input)
        self.input = input
        return input

    # def print(self):
    #     print(self.input)

    def backward(self, learning_rate, dL):
        """
        Run a backward pass through all layers in reverse order.

        Dense layers update their weights; activation layers
        pass the gradient through their derivative.

        Parameters
        ----------
        learning_rate : float
            Step size for weight updates.
        dL : np.ndarray
            Gradient from the loss function.

        Returns
        -------
        np.ndarray
            Gradient with respect to the network's input.
        """
        layers = reversed(self.layers)

        for layer in layers:
            if isinstance(layer, Dense):
                dL = layer.backward(dL, learning_rate)
            else:
                dL = layer.backward(dL)
        self.dL = dL
        return dL
    
    def train(self, input, labels, epochs, learning_rate, patience = 50):
        """
        Train the network using stochastic gradient descent with
        learning rate decay and early stopping.

        Parameters
        ----------
        input : np.ndarray
            Training samples.
        labels : np.ndarray
            Corresponding ground truth targets.
        epochs : int
            Maximum number of training epochs.
        learning_rate : float
            Initial learning rate (decays over time).
        patience : int, optional
            Number of epochs without improvement before early stopping.
            Default is 50.
        """
        mse = MSE()
        best_loss = float('inf')
        best_weights = np.array([])
        no_improve = 0
        for epoch in range(epochs):
            lr = learning_rate * (1 / (1 + 0.001 * epoch))
            for sample in range(0, len(input)):
                result = self.forward(input[sample])
                loss = mse.forward(labels[sample], result)
                grad = mse.backward()
                self.backward(lr, grad)
            
            if best_loss > loss:
                best_loss = loss
                no_improve = 0
                best_weights = [(layer.weights.copy(), layer.bias.copy()) 
                    for layer in self.layers 
                    if isinstance(layer, Dense)]
            else:
                no_improve += 1
            if no_improve == patience:
                print("Early stopping beacause the model ran out of patience(no loss improvement for " + str(no_improve) + " epochs).")
                print("Weights and bais were restored to the best performing version.")
                break
            print("Epoch: " + str(epoch) + " | loss: " + str(loss) + " | lr: " + str(lr) + " | " + "no_imp: " + str(no_improve))
        self.update_weights(best_weights)


    def update_weights(self, weights):
        """
        Restore saved weights and biases to all Dense layers in the network.

        Iterates through the network's layers in order, assigning each
        Dense layer the next set of weights from the provided list.

        Parameters
        ----------
        weights : list of tuple
        A list of (weights, bias) tuples, one per Dense layer,
        in the same order as they appear in the network.
    """
        for layer in self.layers:
            if isinstance(layer, Dense):
                layer.replace_weights(weights[0])
                weights = weights[1:]

    def save_model(self, file_name):
        """
        Save the model's layers to disk using pickle.
        METHOD NOT WORKING

        Parameters
        ----------
        file_name : str
            Path to the file where the model will be saved.
        """
        model = {}
        for layer in range(len(self.layers)):
            model[layer] = self.layers[layer]
        with open(file_name, 'wb') as f:
            pickle.dump(model, f)

    def load_model(self, file_name):
        """
        Load a previously saved model from disk and restore its layers.
        METHOD NOT WORKING

        Parameters
        ----------
        file_name : str
            Path to the pickle file to load.
        """
        with open(file_name, 'rb') as f:
            model = pickle.load(f)