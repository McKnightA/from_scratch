import numpy as np


class Layer:
    # TODO
    def __init__(self):
        self.output = None


class InputLayer(Layer):

    def __init__(self):
        super().__init__()
        self.output_layers = []

    def gather_input(self, input_vector):
        self.output = np.array(input_vector)


class DenseLayer(Layer):

    def __init__(self, units, activation, inital_w_std=0.1):
        super().__init__()
        self.weights = None
        self.i_w_std = inital_w_std
        self.input_layers = None
        self.bias = np.zeros(units)
        self.summ = None
        self.output = np.random.rand(units) - 0.5
        self.error = None
        self.output_layers = []

        self.activation = activation

    def gather_input(self, input_layers=None):
        if input_layers is None:
            input_layers = []
        self.input_layers = input_layers
        for layer in input_layers:
            layer.output_layers.append(self)

        self.weights = (np.random.normal(0, self.i_w_std, (len(self.bias), len(input_layers[0].output))))
        self.summ = np.zeros(len(self.bias))

    def calc_output(self):
        self.summ = self.weights.dot(self.input_layers[0].output)
        z = self.summ + self.bias
        self.output = self.activation.forward(z, self)

        return self.output

    def calc_error(self, og_loss=None):
        if len(self.output_layers) > 0:
            self.error = self.output_layers[0].weights.T.dot(self.output_layers[0].error) \
                         * self.activation.backward(self.summ, self)
        else:
            self.error = self.activation.backward(self.summ, self) * og_loss

    def update_w_and_b(self, learning_rate, norm_cost=0):
        w_grad = np.expand_dims(self.error, axis=1).dot(np.expand_dims(self.input_layers[0].output, axis=0)) \
                 + norm_cost * self.weights
        b_grad = self.error

        self.weights -= learning_rate * w_grad
        self.bias -= learning_rate * b_grad
