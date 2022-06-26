import numpy as np
from Layers import Input
from Losses import *
from Activations import *


# Model class
class Model:

    def __init__(self):
        # Create a list of network objects
        self.layers = []
        self.loss = None
        self.optimizer = None
        self.accuracy = None
        self.input_layer = Input(2)  # WARNING: MAGIC NUMBER IN USE
        self.trainable_layers = []
        self.output_layer_activation = None

        # TODO: remove need for this
        # Softmax classifier's output object
        self.softmax_classifier_output = None

    # Add objects to the model
    def add(self, layer):
        if hasattr(layer, '__iter__'):
            self.layers.extend(layer)
        else:
            self.layers.append(layer)

    # TODO: maybe change self.accuracy to self.metrics
    # Set loss, optimizer and accuracy
    def set(self, *, loss, optimizer, accuracy):
        self.loss = loss
        self.optimizer = optimizer
        self.accuracy = accuracy

    # Finalize the model
    def finalize(self):

        # Count all the objects
        layer_count = len(self.layers)

        # Iterate the objects
        for i in range(layer_count):
            # If layer contains an attribute called "weights",
            # it's a trainable layer -
            # add it to the list of trainable layers
            # We don't need to check for biases -
            # checking for weights is enough
            if hasattr(self.layers[i], 'weights'):
                self.trainable_layers.append(self.layers[i])

        # Also let's save aside the reference to the last object
        # whose output is the model's output
        self.output_layer_activation = self.layers[-1]

        # Update loss object with trainable layers
        self.loss.remember_trainable_layers(self.trainable_layers)

        # TODO: I don't like this soooo
        # If output activation is Softmax and
        # loss function is Categorical Cross-Entropy
        # create an object of combined activation
        # and loss function containing
        # faster gradient calculation
        if isinstance(self.layers[-1], Softmax) and \
                isinstance(self.loss, CategoricalCrossentropy):
            # Create an object of combined activation
            # and loss functions
            self.softmax_classifier_output = \
                Activation_Softmax_Loss_CategoricalCrossentropy()

    # Train the model
    def train(self, X, y, *, epochs=1, print_every=1, validation_data=None):

        # Initialize accuracy object
        self.accuracy.init(y)

        # Main training loop
        for epoch in range(1, epochs + 1):

            # Perform the forward pass
            output = self.forward(X, training=True)

            # Calculate loss
            data_loss, regularization_loss = \
                self.loss.calculate(output, y,
                                    include_regularization=True)
            loss = data_loss + regularization_loss

            # Get predictions and calculate an accuracy
            predictions = self.output_layer_activation.predictions(
                output)
            accuracy = self.accuracy.calculate(predictions, y)

            # Perform backward pass
            self.backward(output, y)

            # Optimize (update parameters)
            self.optimizer.pre_update_params()
            for layer in self.trainable_layers:
                self.optimizer.update_params(layer)
            self.optimizer.post_update_params()

            # Print a summary
            if not epoch % print_every:
                print(f'epoch: {epoch}, ' +
                      f'acc: {accuracy:.3f}, ' +
                      f'loss: {loss:.3f} (' +
                      f'data_loss: {data_loss:.3f}, ' +
                      f'reg_loss: {regularization_loss:.3f}), ' +
                      f'lr: {self.optimizer.current_learning_rate}')

        # If there is the validation data
        if validation_data is not None:
            # For better readability
            X_val, y_val = validation_data

            # Perform the forward pass
            output = self.forward(X_val, training=False)

            # Calculate the loss
            loss = self.loss.calculate(output, y_val)

            # Get predictions and calculate an accuracy
            predictions = self.output_layer_activation.predictions(
                output)
            accuracy = self.accuracy.calculate(predictions, y_val)

            # Print a summary
            print(f'validation, ' +
                  f'acc: {accuracy:.3f}, ' +
                  f'loss: {loss:.3f}')

    # Performs forward pass
    def forward(self, X, training):

        # Call forward method on the input layer
        # this will set the output property that
        # the first layer in the list is expecting
        self.input_layer.forward(X, training)

        # Call forward method of every object in a chain
        # Pass output of the previous object as a parameter
        for i in range(len(self.layers)):
            if i == 0:
                self.layers[i].forward(self.input_layer.output, training)
            else:
                self.layers[i].forward(self.layers[i-1].output, training)

        # return the last object from the list's output
        return self.layers[-1].output

    # Performs backward pass
    def backward(self, output, y):

        # TODO: is the part that requires softmax_cl... I'm not a fan
        # If softmax classifier
        if self.softmax_classifier_output is not None:
            # First call backward method
            # on the combined activation/loss
            # this will set dinputs property
            self.softmax_classifier_output.backward(output, y)

            # Since we'll not call backward method of the last layer
            # which is Softmax activation
            # as we used combined activation/loss
            # object, let's set dinputs in this object
            self.layers[-1].dInputs = \
                self.softmax_classifier_output.dInputs

            # Call backward method going through
            # all the objects but last
            # in reversed order passing dinputs as a parameter
            for i in range(len(self.layers) - 2, -1, -1):
                if i == len(self.layers) - 1:
                    self.layers[i].backward(self.loss.dInputs)
                else:
                    self.layers[i].backward(self.layers[i + 1].dInputs)

            return

        # First call backward method on the loss
        # this will set dInputs property that the last
        # layer will try to access shortly
        self.loss.backward(output, y)

        # Call backward method going through all the objects
        # in reversed order passing dInputs as a parameter
        for i in range(len(self.layers) - 1, -1, -1):
            if i == len(self.layers) - 1:
                self.layers[i].backward(self.loss.dInputs)
            else:
                self.layers[i].backward(self.layers[i+1].dInputs)


# Model class
class GraphModel:

    def __init__(self):
        # Create a list of network objects
        self.layers = {}
        self.input = None
        self.output = None
        self.loss = None
        self.optimizer = None
        self.metric = None
        self.input_layer = None
        self.trainable_layers = []

    # TODO: (short) check that all values in graph are layers
    # TODO: (long) handle receiving a list of input and/or output layers
    # Add objects to the model
    # All values must be lists (I guess sets would actually be good since no order is needed)
    # All elements of those lists must be layers
    def add(self, graph, input=None, output=None, num_features=0):
        g = graph.copy()
        self.input_layer = Input(num_features)

        # TODO (short) make any layer with an empty array as a value an input
        if (input is None and self.input is None) or (output is None and self.output is None):
            raise ValueError("if the model does not already have defined input and output layers it needs them"
                             "defined")

        if (input not in graph and input not in self.layers) or (output not in graph and output not in self.layers):
            raise ValueError("input and output layers must be layers in the graph")

        for layer_outs in graph.values():
            for layer in layer_outs:
                if layer not in self.layers.keys() and layer not in graph.keys():
                    raise ValueError("all values must be a key value in either the existing layer graph "
                                     "or the provided updates")

        if input is not None:
            self.input = input
            g[self.input].append(self.input_layer)
        if output is not None:
            self.output = output

        for layer in g.keys():
            self.layers[layer] = g[layer]
            layer.update_in(g[layer])

    # Set loss, optimizer and accuracy
    def set(self, *, loss, optimizer, accuracy):
        self.loss = loss
        self.optimizer = optimizer
        self.metric = accuracy

    # Finalize the model
    def finalize(self):
        self.trainable_layers = []

        # Iterate the objects
        for layer in self.layers:
            # If layer contains an attribute called "weights",
            # it's a trainable layer -
            # add it to the list of trainable layers
            # We don't need to check for biases -
            # checking for weights is enough
            if hasattr(layer, 'weights'):
                self.trainable_layers.append(layer)

        # Update loss object with trainable layers
        self.loss.remember_trainable_layers(self.trainable_layers)

    # TODO: add a batch size parameter
    # Train the model
    def train(self, X, y, *, epochs=1, print_every=1, validation_data=None):

        # Initialize accuracy object
        self.metric.init(y)

        # Main training loop
        for epoch in range(1, epochs + 1):
            # Perform the forward pass
            output = self.forward(X, training=True)

            # Calculate loss
            data_loss, regularization_loss = self.loss.calculate(output, y, include_regularization=True)
            loss = data_loss + regularization_loss

            # Get predictions and calculate an accuracy
            predictions = self.output.predictions()
            metric = self.metric.calculate(predictions, y)

            # Perform backward pass
            self.backward(output, y)

            # Optimize (update parameters)
            self.optimizer.pre_update_params()
            for layer in self.trainable_layers:
                self.optimizer.update_params(layer)
            self.optimizer.post_update_params()

            self.zero_grads()
            self.zero_outs()

            # Print a summary
            if not epoch % print_every:
                print(f'epoch: {epoch}, ' +
                      f'acc: {metric:.3f}, ' +
                      f'loss: {loss:.3f} (' +
                      f'data_loss: {data_loss:.3f}, ' +
                      f'reg_loss: {regularization_loss:.3f}), ' +
                      f'lr: {self.optimizer.current_learning_rate}')

        # If there is the validation data
        if validation_data is not None:
            # For better readability
            X_val, y_val = validation_data

            # Perform the forward pass
            output = self.forward(X_val, training=False)

            # Calculate the loss
            loss = self.loss.calculate(output, y_val)

            # Get predictions and calculate an accuracy
            predictions = self.output.predictions()
            metric = self.metric.calculate(predictions, y_val)

            # Print a summary
            print(f'validation, ' +
                  f'acc: {metric:.3f}, ' +
                  f'loss: {loss:.3f}')

    # TODO: (long) update by DF by argument
    # Performs forward pass
    def forward(self, X, training):

        # Call forward method on the input layer
        # this will set the output property that
        # the first layer in the list is expecting
        self.input_layer.forward(X, training)

        visited = []
        frontier = [self.input]

        # if method == 'BF':
        # Do a breadth first update of the layer graph
        while len(frontier) > 0:
            current = frontier.pop(0)
            if current not in visited:
                current.forward(training)
                visited.append(current)
                # so layers should hold their activations
                for next_layer in current.output_layers:
                    if next_layer not in visited:
                        frontier.append(next_layer)

        # return the last object from the list's output
        return self.output.output

    # TODO: (long) update by DF by argument
    # Performs backward pass
    def backward(self, output, y):
        # self.output.update_out(self.loss, '+')
        # First call backward method on the loss
        # this will set dInputs property that the last
        # layer will try to access shortly
        self.loss.backward(output, y)

        # Call backward method going through all the objects
        # in reversed order passing dInputs as a parameter

        visited = []
        frontier = [self.output]

        # if method == 'BF':
        # Do a breadth first update of the layer graph
        while len(frontier) > 0:
            current = frontier.pop(0)
            if current not in visited and not isinstance(current, Input):
                if current == self.output:
                    current.backward(self.loss.dInputs)
                else:
                    current.backward()
                visited.append(current)
                # so layers should hold their activations
                for next_layer in current.input_layers:
                    if next_layer not in visited:
                        frontier.append(next_layer)

        return

    def zero_grads(self):
        for layer in self.layers:
            for laer in layer.dInputs:
                layer.dInputs[laer] = np.zeros_like(layer.dInputs[laer])

            if hasattr(layer, 'dWeights'):
                for laer in layer.dWeights:
                    layer.dWeights[laer] = np.zeros_like(layer.dWeights[laer])
            if hasattr(layer, 'dBiases'):
                layer.dBiases = np.zeros_like(layer.dBiases)

    def zero_outs(self):
        for layer in self.layers.keys():
            layer.output = np.zeros(shape=(1, layer.output.shape[-1]))

