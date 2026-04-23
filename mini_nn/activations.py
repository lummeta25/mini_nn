import numpy as np
from mini_nn.loss import MSE

class relu:
    """
    Rectified Linear Unit (ReLU) activation function.

    Passes positive values unchanged and zeros out negatives,
    introducing non-linearity into the network.
    """
    def forward(self, input):
        """
        Apply ReLU: max(0, x).

        Parameters
        ----------
        input : np.ndarray
            Pre-activation values from the previous layer.

        Returns
        -------
        np.ndarray
            Input with negative values zeroed out.
        """
        self.input = input
        return np.maximum(0, input)
    
    def backward(self, input):
        """
        Backpropagate through ReLU.

        Passes the gradient where the forward input was positive,
        zeros it where it was negative.

        Parameters
        ----------
        dL : np.ndarray
            Gradient flowing in from the next layer.

        Returns
        -------
        np.ndarray
            Gradient after applying the ReLU derivative.
        """
        return input * (self.input > 0)
    
class leaky_relu:
    """
    Leaky ReLU activation function.

    Like ReLU but allows a small gradient (alpha) for negative inputs,
    helping avoid the dying ReLU problem.

    Parameters
    ----------
    alpha : float, optional
        Slope for negative values. Default is 0.01.
    """
    def __init__(self, alpha = 0.01):
        self.alpha = alpha

    def forward(self, input):
        """
        Apply Leaky ReLU: x if x > 0, else alpha * x.

        Parameters
        ----------
        input : np.ndarray
            Pre-activation values from the previous layer.

        Returns
        -------
        np.ndarray
            Activated output.
        """
        self.input = input
        self.output = np.where(input > 0, input, self.alpha * input)
        return self.output
        
    def backward(self, dL):
        """
        Backpropagate through Leaky ReLU.

        Parameters
        ----------
        dL : np.ndarray
            Gradient flowing in from the next layer.

        Returns
        -------
        np.ndarray
            Gradient after applying the Leaky ReLU derivative.
        """
        return dL * np.where(self.input > 0, 1, self.alpha)
    

class sigmoid:
    """
    Sigmoid activation function.

    Squashes input values to the range (0, 1). Commonly used
    in binary classification output layers.
    """
    def forward(self, input):
        """
        Apply sigmoid: 1 / (1 + exp(-x)).

        Parameters
        ----------
        input : np.ndarray
            Pre-activation values from the previous layer.

        Returns
        -------
        np.ndarray
            Values in the range (0, 1).
        """
        self.input = input
        self.output = 1 / (1 + np.exp(-input))
        return self.output
    
    def backward(self, dL):
        """
        Backpropagate through sigmoid.

        Uses the stored forward output: sigmoid'(x) = sigmoid(x) * (1 - sigmoid(x)).

        Parameters
        ----------
        dL : np.ndarray
            Gradient flowing in from the next layer.

        Returns
        -------
        np.ndarray
            Gradient after applying the sigmoid derivative.
        """
        return dL * self.output * (1 - self.output)
    
class tanh:
    """
    Hyperbolic tangent activation function.

    Squashes input values to the range (-1, 1). Zero-centered,
    which can make learning faster than sigmoid in some cases.
    """
    def forward(self, input):
        """
        Apply tanh.

        Parameters
        ----------
        input : np.ndarray
            Pre-activation values from the previous layer.

        Returns
        -------
        np.ndarray
            Values in the range (-1, 1).
        """
        self.input = input
        self.output = np.tanh(input)
        return self.output
    def backward(self, dL):
        """
        Backpropagate through tanh.

        Uses tanh'(x) = 1 - tanh(x)^2.

        Parameters
        ----------
        dL : np.ndarray
            Gradient flowing in from the next layer.

        Returns
        -------
        np.ndarray
            Gradient after applying the tanh derivative.
        """
        return dL * (1 - (np.square(self.output)))