from mini_nn.layers import Dense
from mini_nn.loss import MSE
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

    def print(self):
        print(self.input)

    def backward(self, learning_rate, dL):
        layers = reversed(self.layers)

        for layer in layers:
            if isinstance(layer, Dense):
                dL = layer.backward(dL, learning_rate)
            else:
                dL = layer.backward(dL)
        self.dL = dL
        return dL
    
    def train(self, input, lables, epochs, learning_rate):
        mse = MSE()
        for epoch in range(epochs):
            lr = learning_rate * (1 / (1 + 0.001 * epoch))
            for sample in range(0, len(input)):
                result = self.forward(input[sample])
                loss = mse.forward(lables[sample], result)
                grad = mse.backward()
                self.backward(lr, grad)
            print("Epoch: " + str(epoch) + " | loss: " + str(loss) + " | lr: " + str(lr))

    def save_model(self, file_name):
        model = {}
        for layer in range(len(self.layers)):
            model[layer] = self.layers[layer]
        with open(file_name, 'wb') as f:
            pickle.dump(model, f)

    def load_model(self, file_name):
        with open(file_name, 'rb') as f:
            model = pickle.load(f)