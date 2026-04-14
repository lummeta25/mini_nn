# mini_nn

A minimal neural network library built from scratch using NumPy. No deep learning frameworks — just pure Python and NumPy.

Built as a learning project to understand what happens under the hood of libraries like PyTorch and Keras.

## Installation

No installation required. Just clone the repo and import the modules directly.

```bash
git clone https://github.com/lummeta25/mini_nn
```

**Requirements:** Python 3.x, NumPy

```bash
pip install numpy
```

## Usage

```python
import numpy as np
from mini_nn.layers import Dense
from mini_nn.activations import relu
from mini_nn.model import NeuralNetwork

# XOR dataset
X = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
])
y = np.array([0, 1, 1, 0])

# Build model
model = NeuralNetwork([
    Dense(4, 2),
    relu(),
    Dense(1, 4),
])

# Train
model.train(X, y, epochs=1000, learning_rate=0.001)

# Predict
for sample in X:
    print(model.forward(sample))
```

## Project Structure

```
mini_nn/
├── nn/
│   ├── __init__.py
│   ├── layers.py        # Dense layer with forward and backward pass
│   ├── activations.py   # ReLU, Leaky ReLU, Sigmoid, Tanh
│   ├── loss.py          # MSE loss
│   └── model.py         # NeuralNetwork class, training loop
```

## Supported Features

- **Layers:** Dense (fully connected)
- **Activations:** ReLU, Leaky ReLU, Sigmoid, Tanh
- **Loss:** Mean Squared Error (MSE)
- **Training:** Stochastic gradient descent with learning rate decay

## Roadmap

- [ ] Early stopping
- [ ] Model checkpointing (save best weights)
- [ ] Mini-batch gradient descent
- [ ] Multi-class output support (Softmax + Cross-entropy loss)
- [ ] Additional loss functions
- [ ] Docstrings and full documentation

## How It Works

Each layer implements a `forward` and `backward` method. During training:

1. **Forward pass** — input flows through each layer to produce a prediction
2. **Loss calculation** — MSE measures how wrong the prediction is
3. **Backward pass** — gradients flow back through each layer using the chain rule
4. **Weight update** — weights and biases are adjusted using gradient descent

## License

MIT
