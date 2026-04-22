from mini_nn.layers import Dense
from mini_nn.activations import *
from mini_nn.loss import *
from mini_nn.model import NeuralNetwork
from sklearn.datasets import load_iris
from sklearn.preprocessing import normalize
import numpy as np
import numpy as np

# Load and normalize
iris = load_iris()
X = normalize(iris.data)        # shape (150, 4)
y = iris.target.astype(float)   # 0, 1, or 2

# Build model
model = NeuralNetwork([
    Dense(8, 4),
    relu(),
    Dense(4, 8),
    relu(),
    Dense(1, 4),
])

model.train(X, y, epochs=10, learning_rate=0.01)