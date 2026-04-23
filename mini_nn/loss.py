import numpy as np

class MSE:
    """
    Mean Squared Error loss function.

    Used to measure the average squared difference between
    predicted and actual values during training.
    """
    def forward(self, actual, predicted):
        """
        Compute the MSE loss.

        Parameters
        ----------
        actual : np.ndarray
            Ground truth target values.
        predicted : np.ndarray
            Model predictions.

        Returns
        -------
        float
            The mean squared error between actual and predicted.
        """
        self.actual = actual
        self.predicted = predicted
        return np.mean(np.square(predicted - actual))

    def backward(self):
        """
        Compute the gradient of MSE loss with respect to predictions.

        Returns
        -------
        np.ndarray
            Gradient of the loss: 2 * (predicted - actual).
        """
        return 2 * (self.predicted - self.actual)