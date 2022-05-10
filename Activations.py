import numpy as np


class Activation:
    """Activation methods for neural network layers"""
    def __init__(self):
        self.name = None

    def forward(self, x):
        """The function used during forward pass of information"""
        raise NotImplementedError("This activation function has not been implemented")

    def backward(self, x):
        """The derivative of the function used during back propogation"""
        raise NotImplementedError("The derivative of this activation function has not been implemented")


class Relu(Activation):
    """The REctified Linear Unit activation function"""

    def __init__(self):
        super().__init__()

    def forward(self, x, layer):
        """Is linear when greater than 0 and 0 when less than 0
        x must be a numpy array"""
        return np.maximum(0, x)

    def backward(self, x, layer):
        """1 when x is greater than 0 and 0 when less than or equal to 0
        x must be a numpy array"""
        return x > 0


class Sigmoid(Activation):
    """The sigmoid function that has an output constrained in [0, 1]"""

    def __init__(self):
        super().__init__()

    def forward(self, x):
        """applies 1/(1+e^(-x)) element wise on the passed in array.
        normalizes the array before applying the above equation so that exponential doesn't explode"""
        x -= np.mean(x, axis=0)
        x /= np.std(x, axis=0)
        return 1 / (1 + np.exp(-x))

    def backward(self, x):
        """forward(x)*(1-forward(x))"""
        return self.forward(x) * (1 - self.forward(x))


class Softmax(Activation):
    """The softmax equation of a summed group being normed so the sum totals to 1"""

    def __init__(self):
        super().__init__()

    def forward(self, x, layer):
        """applies e^(x)/ sum(e^(x)) element wise on the passed in array.
        normalizes the array before applying the above equation so that exponential don't explode."""

        try:
            x -= np.mean(x)
        except Exception:
            print("Found the break on np.mean in softmax forward")
            print(x)

        try:
            x /= np.std(x)
        except Exception:
            print("Found the break on np.std in softmax forward")
            print(x)

        return np.exp(x) / np.sum(np.exp(x))

    def backward(self, x, layer):
        """forward(x)*(1-forward(x))"""
        return self.forward(x, layer) * (1 - self.forward(x, layer))
