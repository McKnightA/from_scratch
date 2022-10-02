import numpy as np


class Layer:

    pass
    """def update_in(self, input_layers):
        raise NotImplementedError("Stop not writing bits")

    def update_out(self, layer, option):
        raise NotImplementedError("Stop not writing bits")

    # Forward pass
    def forward(self, training):
        raise NotImplementedError("Stop not writing bits")

    def backward(self):
        raise NotImplementedError("Stop not writing bits")"""


# Dense layer
class GraphDense(Layer):

    # Layer initialization
    def __init__(self, n_neurons, activation,
                 weight_regularizer_l1=0, weight_regularizer_l2=0,
                 bias_regularizer_l1=0, bias_regularizer_l2=0):
        # Initialize biases and output
        self.biases = np.zeros((1, n_neurons))
        self.dBiases = np.zeros((1, n_neurons))
        self.output = np.zeros((1, n_neurons))
        # Set regularization strength
        self.weight_regularizer_l1 = weight_regularizer_l1
        self.weight_regularizer_l2 = weight_regularizer_l2
        self.bias_regularizer_l1 = bias_regularizer_l1
        self.bias_regularizer_l2 = bias_regularizer_l2
        # Set layer activation
        self.activation = activation
        # Setting everything else to empty
        self.input_layers = set([])
        self.output_layers = set([])
        self.weights = {}
        self.dWeights = {}
        self.dInputs = {}

    def update_in(self, input_layers):
        if len(input_layers) == 0:
            raise ValueError("dont pass in an empty iterable to update_in()")
            # potentially remove this check
            # maybe the ability to have a neron have no input and basically be a constant could be helpful

        if len(self.weights) == 0:
            self.input_layers = set(input_layers)
            for layer in input_layers:
                self.weights[layer] = np.random.normal(scale=0.001,
                                                       size=(layer.output.shape[-1], self.biases.shape[-1]))
                self.dWeights[layer] = np.random.normal(scale=0.001,
                                                        size=(layer.output.shape[-1], self.biases.shape[-1]))
                self.dInputs[layer] = np.zeros_like(layer.output)
                layer.update_out(self, '+')

        else:
            # need to remove any connections that aren't there anymore and add any new connections
            for layer in self.input_layers:
                if layer not in input_layers:
                    self.input_layers.remove(layer)
                    layer.update_out(self, '-')
                    del self.weights[layer]
                    del self.dWeights[layer]
                    del self.dInputs[layer]

            for layer in input_layers:
                if layer not in self.input_layers:
                    self.input_layers.add(layer)
                    self.weights[layer] = np.random.normal(scale=0.001,
                                                           size=(layer.output.shape[-1], self.biases.shape[-1]))
                    self.dWeights[layer] = np.random.normal(scale=0.001,
                                                            size=(layer.output.shape[-1], self.biases.shape[-1]))
                    self.dInputs[layer] = np.zeros_like(layer.output)
                    layer.update_out(self, '+')

    def update_out(self, layer, option):
        if option == '+':
            self.output_layers.add(layer)
        elif option == '-':
            self.output_layers.remove(layer)

    # Forward pass
    def forward(self, training):
        # Calculate output values from inputs, weights
        summ = 0
        for layer in self.input_layers:
            # (batch, input feats) . (input feats, self feats)
            summ = summ + np.dot(layer.output, self.weights[layer])

        # add biases
        self.output = summ + self.biases
        self.activation.forward(self.output, training)
        self.output = self.activation.output

    # Backward pass
    def backward(self, lossdInputs=None):
        # what is dvalues? error of each neuron for each in batch
        # where do dvalues come from? the sum of the self.output_layers.dInputs[self]
        # Gradients on parameters
        if lossdInputs is None:
            dValues = 0
        else:
            dValues = lossdInputs

        for layer in self.output_layers:
            if isinstance(layer, GraphDense):
                dValues = dValues + layer.dInputs[self]
            else:  # does this support dropout layer?
                dValues += layer.dInputs

        self.activation.backward(dValues)
        dValues = self.activation.dInputs

        for layer in self.input_layers:
            self.dWeights[layer] = np.dot(layer.output.T, dValues)  # (input feats, batch) . (batch, self feats)
        self.dBiases = np.sum(dValues, axis=0, keepdims=True)  # (1, self feats)

        # Gradients on regularization
        # L1 on weights
        for layer in self.weights:
            if self.weight_regularizer_l1 > 0:
                dL1 = np.ones_like(self.weights[layer])
                dL1[self.weights[layer] < 0] = -1
                self.dWeights[layer] += self.weight_regularizer_l1 * dL1
            # L2 on weights
            if self.weight_regularizer_l2 > 0:
                self.dWeights[layer] += 2 * self.weight_regularizer_l2 * self.weights[layer]
        # L1 on biases
        if self.bias_regularizer_l1 > 0:
            dL1 = np.ones_like(self.biases)
            dL1[self.biases < 0] = -1
            self.dBiases += self.bias_regularizer_l1 * dL1
        # L2 on biases
        if self.bias_regularizer_l2 > 0:
            self.dBiases += 2 * self.bias_regularizer_l2 * self.biases

        # Gradient on values
        for layer in self.input_layers:
            self.dInputs[layer] = np.dot(dValues, self.weights[layer].T)

    def predictions(self):
        return self.activation.predictions(self.output)


# Input "layer"
class Input(Layer):

    def __init__(self, n_features):
        self.output = np.zeros((1, n_features))

    # Forward pass
    def forward(self, inputs, training):
        self.output = inputs

    def backward(self, ):
        pass

    def update_out(self, layer, option):
        pass

    def update_in(self, input_layers):
        pass


# TODO integrate into dense layer
"""
# Dropout
class Dropout(Layer):

    # Init
    def __init__(self, rate):
        # Store rate, we invert it as for example for dropout
        # of 0.1 we need success rate of 0.9
        self.rate = 1 - rate
        self.inputs = None
        self.output = None
        self.binary_mask = None
        self.dInputs = None

    # Forward pass
    def forward(self, inputs, training):
        # Save input values
        self.inputs = inputs

        # If not in the training mode - return values
        if not training:
            self.output = inputs.copy()
            return

        # Generate and save scaled mask
        self.binary_mask = np.random.binomial(1, self.rate,
                                              size=inputs.shape) / self.rate
        # Apply mask to output values
        self.output = inputs * self.binary_mask

    # Backward pass
    def backward(self, dvalues):
        # Gradient on values
        self.dInputs = dvalues * self.binary_mask
"""


