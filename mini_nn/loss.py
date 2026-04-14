import numpy as np

class MSE:
    def forward(self, actual, predicted):
        self.actual = actual
        self.predicted = predicted
        return np.mean(np.square(predicted - actual))

    def backward(self):
        return 2 * (self.predicted - self.actual)