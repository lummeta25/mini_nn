import numpy as np
from mini_nn.loss import MSE

class relu:
    def forward(self, input):
        self.input = input
        return np.maximum(0, input)
    
    def backward(self, input):
        return input * (self.input > 0)
    
class leaky_relu:
    def __init__(self, alpha = 0.01):
        self.alpha = alpha

    def forward(self, input):
        self.input = input
        self.output = np.where(input > 0, input, self.alpha * input)
        return self.output
        
    def backward(self, dL):
        return dL * np.where(self.input > 0, 1, self.alpha)
    

class sigmoid:
    def forward(self, input):
        self.input = input
        self.output = 1 / (1 + np.exp(-input))
        return self.output
    
    def backward(self, dL):
        return dL * self.output * (1 - self.output)
    
class tanh:
    def forward(self, input):
        self.input = input
        self.output = np.tanh(input)
        return self.output
    def backward(self, dL):
        return dL * (1 - (np.square(self.output)))