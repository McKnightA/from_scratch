import numpy as np
from Layers import *
from Losses import *
from Activations import *


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

    # Add objects to the model
    # All values must be lists (I guess sets would actually be good since no order is needed)
    # All elements of those lists must be layers
    def add(self, graph, input=None, output=None, num_features=0):

        if self.input_layer is None:
            if num_features < 1:
                raise ValueError("number of input features can't be less than 1 on first add() call")

            self.input_layer = Input(num_features)

        if (input is None and self.input is None) or (output is None and self.output is None):
            raise ValueError("if the model does not already have defined input and output layers it needs them"
                             "defined")
        if input is not None:
            for layer in input:
                if layer not in graph and layer not in self.layers:
                    raise ValueError("input layers must be layers in the graph")

        if output is not None:
            for layer in output:
                if layer not in graph and layer not in self.layers:
                    raise ValueError("output layers must be layers in the graph")

        # TODO: make it check for Layers more generally
        for layer in graph.keys():
            if type(layer) is not GraphDense:
                raise ValueError("Each key in graph must be a Layer")

        # TODO: make it check for Layers more generally
        for layers_in in graph.values():  # lists of layers that each layer points to
            for layer in layers_in:
                if type(layer) is not GraphDense and type(layer) is not Input:
                    raise ValueError("each object in the adjecency lists must be a Layer")

        for layers_in in graph.values():  # lists of layers that each layer points to
            for layer in layers_in:
                if layer not in self.layers.keys() and layer not in graph.keys() and type(layer) is not Input:
                    raise ValueError("all values must be a key value in either the existing layer graph "
                                     "or the provided updates")

        if self.input_layer is not None and num_features != 0:
            if num_features != len(self.input_layer.output):
                self.input_layer = Input(num_features)

        g = graph.copy()

        # gotta check for an updated list of input layers
        if input is not None:
            self.input = input

            # don't need the input layer inputing to any layer not on the list
            for layer in self.layers.keys():
                if self.input_layer in g[layer]:
                    g[layer].remove(self.input_layer)
                    # Why only remove from g and not self.layers?

            # gotta add the input layer to the list of inputs for all layers on the list
            for layer in input:
                if self.input_layer not in g[layer]:
                    g[layer].append(self.input_layer)

        if output is not None:
            self.output = output

        # need to remove layers that no longer exist
        for layer in self.layers:
            if layer not in g.keys():
                del self.layers[layer]

        # update the layer connections after adding any new layers
        for layer in g.keys():
            self.layers[layer] = g[layer]
            layer.update_in(g[layer])

        # self.zero_outs()
        # self.zero_grads()

    # Set loss, optimizer and accuracy
    def set(self, *, loss, optimizer, accuracy):
        if loss is Loss:
            self.loss = [loss]
        else:  # elif loss is a list
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

    def loop(self, X, y, epoch, batch_size, time_series, print_every):
        # Perform the forward pass
        for i in range(batch_size, len(X), batch_size):
            output = self.forward(X[i - batch_size:i], training=True)

            # Calculate loss
            data_loss, regularization_loss = self.loss.calculate(output, y[i - batch_size:i],
                                                                 include_regularization=True)
            loss = data_loss + regularization_loss

            # Get predictions and calculate an accuracy
            predictions = np.concatenate([out.predictions() for out in self.output], axis=-1)
            metric = self.metric.calculate(predictions, y[i - batch_size:i])

            # Perform backward pass
            self.backward(output, y[i - batch_size:i])

            # Optimize (update parameters)
            for layer in self.trainable_layers:
                self.optimizer.update_params(layer)

            if not time_series:
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

    # Train the model
    def train(self, data, labels, *, epochs=1, batch_size=64, time_series=False, print_every=1, validation_data=None):
        if len(data) != len(labels):
            raise ValueError("data and labels must have the same length")

        # Best practice to not modify the information given, so operate on a copy to be safe
        if time_series:
            for dat, lab in zip(data, labels):
                if len(dat) != len(lab):
                    raise ValueError("data and labels must have the same length")

            x = []
            Y = []
            for dat, lab in zip(data, labels):
                x.append(dat.copy())
                Y.append(lab.copy())

        else:
            x = data.copy()
            Y = labels.copy()

        # Initialize accuracy object
        self.metric.init(Y)

        # Main training loop
        for epoch in range(1, epochs + 1):
            self.optimizer.pre_update_params()

            self.zero_grads()
            self.zero_outs()

            if time_series:
                for i in range(len(x)):
                    if len(x[i]) < batch_size:
                        x[i] = np.pad(x[i], ((0, batch_size - len(x[i]) + 1), (0, 0)))
                        Y[i] = np.pad(Y[i], ((0, batch_size - len(Y[i]) + 1), (0, 0)))

                    self.loop(x[i], Y[i], epoch, min(batch_size, len(x[i])-1), time_series, print_every)
                    self.zero_grads()
                    self.zero_outs()

            else:
                self.loop(x, Y, epoch, min(batch_size, len(x)-1), time_series, print_every)

            self.optimizer.post_update_params()

        # If there is the validation data
        if validation_data is not None:
            self.zero_grads()
            self.zero_outs()

            # For better readability
            X_val, y_val = validation_data

            if time_series:
                metrics = []
                losses = []

                for i in range(len(X_val)):
                    # Perform the forward pass
                    output = self.forward(X_val[i], training=False)

                    # Calculate the loss
                    loss = self.loss.calculate(output, y_val[i])
                    losses.append(loss)

                    # Get predictions and calculate an accuracy
                    predictions = np.concatenate([out.predictions() for out in self.output], axis=-1)
                    metric = self.metric.calculate(predictions, y_val[i])
                    metrics.append(metric)

                    self.zero_grads()
                    self.zero_outs()

                # Print a summary
                print(f'validation, ' +
                      f'acc: {sum(metrics)/len(metrics):.3f}, ' +
                      f'loss: {sum(losses)/len(losses):.3f}')

            else:
                # Perform the forward pass
                output = self.forward(X_val, training=False)

                # Calculate the loss
                loss = self.loss.calculate(output, y_val)

                # Get predictions and calculate an accuracy
                predictions = np.concatenate([out.predictions() for out in self.output], axis=-1)
                metric = self.metric.calculate(predictions, y_val)

                # Print a summary
                print(f'validation, ' +
                      f'acc: {metric:.3f}, ' +
                      f'loss: {loss:.3f}')

                self.zero_grads()
                self.zero_outs()

    # TODO: (long) update by DF by argument
    # TODO: (longer) update by priority first with priority determined by number of edges
    # Performs forward pass
    def forward(self, X, training):

        # Call forward method on the input layer
        # this will set the output property that
        # the first layer in the list is expecting
        self.input_layer.forward(X, training)

        visited = []
        frontier = self.input.copy()

        # if method == 'BF':
        # Do a breadth first update of the layer graph
        while len(frontier) > 0:
            current = frontier.pop(0)
            if current not in visited:
                current.forward(training)
                visited.append(current)
                
                for next_layer in current.output_layers:
                    if next_layer not in visited:
                        frontier.append(next_layer)

        # return the last object from the list's output
        return np.concatenate([layer.output for layer in self.output], axis=-1)

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
        frontier = self.output.copy()

        # if method == 'BF':
        # Do a breadth first update of the layer graph
        while len(frontier) > 0:
            current = frontier.pop(0)
            if current not in visited and not isinstance(current, Input):
                if current in self.output:
                    current.backward(self.loss.dInputs)
                else:
                    current.backward()
                visited.append(current)
                # so layers should hold their activations
                for next_layer in current.input_layers:
                    if next_layer not in visited:
                        frontier.append(next_layer)

        return

    def copy(self):
        # create a new model object to return
        cop = GraphModel()

        # create new layers with the same parameters as all the layers in this model
        old_layers = list(self.layers.keys())
        new_layers = []
        for layer in old_layers:
            l = GraphDense(layer.biases.shape[-1],
                           layer.activation,
                           layer.weight_regularizer_l1,
                           layer.weight_regularizer_l2,
                           layer.bias_regularizer_l1,
                           layer.bias_regularizer_l2)

            new_layers.append(l)

        # copy the connections of each layer
        new_graph = {}
        for i, layer in enumerate(old_layers):
            ls = self.layers[layer]

            new_ls = []
            for layr in ls:
                # get its index in old_layer then add that index of new_layers to new_ls
                for old, new in zip(old_layers, new_layers):
                    if layr == old:
                        new_ls.append(new)
                        break

            new_graph[new_layers[i]] = new_ls

        # copy input and output
        inp = []
        for layer in self.input:
            for old, new in zip(old_layers, new_layers):
                if layer == old:
                    inp.append(new)

        oup = []
        for layer in self.output:
            for old, new in zip(old_layers, new_layers):
                if layer == old:
                    oup.append(new)

        # copy the number of features
        feat = self.input_layer.output.shape[-1]

        # add info to the new model object
        cop.add(new_graph, inp, oup, feat)

        # update the weights for each layer
        for old, new in zip(old_layers, new_layers):
            for layer in old.weights.keys():
                for o, n in zip(old_layers, new_layers):
                    if layer == o:
                        new.weights[n] = old.weights[o].copy()
                        break

        # set loss and what not
        cop.set(loss=self.loss, optimizer=self.optimizer.copy(), accuracy=self.metric.copy())

        # finish and clean
        cop.finalize()
        cop.zero_outs()
        cop.zero_grads()

        # return the copy
        return cop

    def predict(self, list_of_input):
        self.zero_outs()

        predictions = []
        for inp in list_of_input:
            out = self.forward(np.array([inp]), False)
            predictions.append(out)

        return np.array(predictions)

    def foresight(self, list_of_input, distance):
        predictions = []
        for inp in list_of_input:
            out = self.forward(np.array([inp]), False)
            predictions.append(out)

        for i in range(distance):
            out = self.forward(np.array(predictions[-1]), False)
            predictions.append(out)

        return np.array(predictions)

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

