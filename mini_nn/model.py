from mini_nn.layers import Dense
from mini_nn.loss import MSE
import numpy as np
import pickle

class NeuralNetwork:
    def __init__(self, layers):
        self.layers = layers
        
        
        
    def forward(self, input):
        for layer in self.layers:
            input = layer.forward(input)
        # print(input)
        self.input = input
        return input

    # def print(self):
    #     print(self.input)

    def backward(self, learning_rate, dL):
        layers = reversed(self.layers)

        for layer in layers:
            if isinstance(layer, Dense):
                dL = layer.backward(dL, learning_rate)
            else:
                dL = layer.backward(dL)
        self.dL = dL
        return dL
    
    def train(self, input, lables, epochs, learning_rate, patience = 50):
        mse = MSE()
        best_loss = float('inf')
        best_wights = np.array([])
        no_imporve = 0
        for epoch in range(epochs):
            lr = learning_rate * (1 / (1 + 0.001 * epoch))
            for sample in range(0, len(input)):
                result = self.forward(input[sample])
                loss = mse.forward(lables[sample], result)
                grad = mse.backward()
                self.backward(lr, grad)
            
            if best_loss > loss:
                best_loss = loss
                no_imporve = 0
                best_wights = [(layer.weights.copy(), layer.bias.copy()) 
                    for layer in self.layers 
                    if isinstance(layer, Dense)]
            else:
                no_imporve += 1
            if no_imporve == patience:
                print("Early stopping beacause the model ran out of patience(no loss improvement for " + str(no_imporve) + " epochs).")
                print("Weights and bais were restored to the best performing version.")
                self.update_weights(best_wights)
                break
            print("Epoch: " + str(epoch) + " | loss: " + str(loss) + " | lr: " + str(lr) + " | " + "no_imp: " + str(no_imporve))
        self.update_weights(best_wights)


    def update_weights(self, weights):
        for layer in self.layers:
            if isinstance(layer, Dense):
                layer.replace_weights(weights[0])
                weights = weights[1:]

    def save_model(self, file_name):
        model = {}
        for layer in range(len(self.layers)):
            model[layer] = self.layers[layer]
        with open(file_name, 'wb') as f:
            pickle.dump(model, f)

    def load_model(self, file_name):
        with open(file_name, 'rb') as f:
            model = pickle.load(f)