import numpy as np

# TODO (sooner rather than later): make an optimizer interface
# TODO (eventually): make all versions graph compliant
"""
# SGD optimizer
class SGD:

    # Initialize optimizer - set settings,
    # learning rate of 1. is default for this optimizer
    def __init__(self, learning_rate=1., decay=0., momentum=0.):
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.iterations = 0
        self.momentum = momentum

    # Call once before any parameter updates
    def pre_update_params(self):
        if self.decay:
            self.current_learning_rate = self.learning_rate * \
                                         (1. / (1. + self.decay * self.iterations))

    # Update parameters
    def update_params(self, layer):

        # TODO: make work with a dictionary of weights
        # If we use momentum
        if self.momentum:

            # If layer does not contain momentum arrays, create them
            # filled with zeros
            if not hasattr(layer, 'weight_momentums'):
                layer.weight_momentums = np.zeros_like(layer.weights)
                # If there is no momentum array for weights
                # The array doesn't exist for biases yet either.
                layer.bias_momentums = np.zeros_like(layer.biases)
            # Build weight updates with momentum - take previous
            # updates multiplied by retain factor and update with
            # current gradients
            weight_updates = \
                self.momentum * layer.weight_momentums - \
                self.current_learning_rate * layer.dWeights
            layer.weight_momentums = weight_updates

            # Build bias updates
            bias_updates = \
                self.momentum * layer.bias_momentums - \
                self.current_learning_rate * layer.dBiases
            layer.bias_momentums = bias_updates

        # Vanilla SGD updates (as before momentum update)
        else:
            weight_updates = -self.current_learning_rate * layer.dWeights
            bias_updates = -self.current_learning_rate * layer.dBiases

        # Update weights and biases using either
        # vanilla or momentum updates
        layer.weights += weight_updates
        layer.biases += bias_updates

    # Call once after any parameter updates
    def post_update_params(self):
        self.iterations += 1


# Adagrad optimizer
class Adagrad:

    # Initialize optimizer - set settings
    def __init__(self, learning_rate=1., decay=0., epsilon=1e-7):
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.iterations = 0
        self.epsilon = epsilon

    # Call once before any parameter updates
    def pre_update_params(self):
        if self.decay:
            self.current_learning_rate = self.learning_rate * \
                                         (1. / (1. + self.decay * self.iterations))

    # Update parameters
    def update_params(self, layer):

        # If layer does not contain cache arrays,
        # create them filled with zeros
        if not hasattr(layer, 'weight_cache'):
            layer.weight_cache = np.zeros_like(layer.weights)
            layer.bias_cache = np.zeros_like(layer.biases)

        # Update cache with squared current gradients
        layer.weight_cache += layer.dWeights ** 2
        layer.bias_cache += layer.dBiases ** 2

        # Vanilla SGD parameter update + normalization
        # with square rooted cache
        layer.weights += -self.current_learning_rate * \
                         layer.dWeights / \
                         (np.sqrt(layer.weight_cache) + self.epsilon)
        layer.biases += -self.current_learning_rate * \
                        layer.dBiases / \
                        (np.sqrt(layer.bias_cache) + self.epsilon)

    # Call once after any parameter updates
    def post_update_params(self):
        self.iterations += 1


# RMSprop optimizer
class RMSprop:

    # Initialize optimizer - set settings
    def __init__(self, learning_rate=0.001, decay=0., epsilon=1e-7,
                 rho=0.9):
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.iterations = 0
        self.epsilon = epsilon
        self.rho = rho

    # Call once before any parameter updates
    def pre_update_params(self):
        if self.decay:
            self.current_learning_rate = self.learning_rate * \
                                         (1. / (1. + self.decay * self.iterations))

    # Update parameters
    def update_params(self, layer):

        # If layer does not contain cache arrays,
        # create them filled with zeros
        if not hasattr(layer, 'weight_cache'):
            layer.weight_cache = np.zeros_like(layer.weights)
            layer.bias_cache = np.zeros_like(layer.biases)

        # Update cache with squared current gradients
        layer.weight_cache = self.rho * layer.weight_cache + \
                             (1 - self.rho) * layer.dWeights ** 2
        layer.bias_cache = self.rho * layer.bias_cache + \
                           (1 - self.rho) * layer.dBiases ** 2

        # Vanilla SGD parameter update + normalization
        # with square rooted cache
        layer.weights += -self.current_learning_rate * \
                         layer.dWeights / \
                         (np.sqrt(layer.weight_cache) + self.epsilon)
        layer.biases += -self.current_learning_rate * \
                        layer.dBiases / \
                        (np.sqrt(layer.bias_cache) + self.epsilon)

    # Call once after any parameter updates
    def post_update_params(self):
        self.iterations += 1


# Adam optimizer
class Adam:

    # Initialize optimizer - set settings
    def __init__(self, learning_rate=0.001, decay=0., epsilon=1e-7,
                 beta_1=0.9, beta_2=0.999):
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.iterations = 0
        self.epsilon = epsilon
        self.beta_1 = beta_1
        self.beta_2 = beta_2

    # Call once before any parameter updates
    def pre_update_params(self):
        if self.decay:
            self.current_learning_rate = self.learning_rate * \
                                         (1. / (1. + self.decay * self.iterations))

    # Update parameters
    def update_params(self, layer):

        # If layer does not contain cache arrays,
        # create them filled with zeros
        if not hasattr(layer, 'weight_cache'):
            layer.weight_momentums = np.zeros_like(layer.weights)
            layer.weight_cache = np.zeros_like(layer.weights)
            layer.bias_momentums = np.zeros_like(layer.biases)
            layer.bias_cache = np.zeros_like(layer.biases)

        # Update momentum  with current gradients
        layer.weight_momentums = self.beta_1 * \
                                 layer.weight_momentums + \
                                 (1 - self.beta_1) * layer.dweights
        layer.bias_momentums = self.beta_1 * \
                               layer.bias_momentums + \
                               (1 - self.beta_1) * layer.dbiases
        # Get corrected momentum
        # self.iteration is 0 at first pass
        # and we need to start with 1 here
        weight_momentums_corrected = layer.weight_momentums / \
                                     (1 - self.beta_1 ** (self.iterations + 1))
        bias_momentums_corrected = layer.bias_momentums / \
                                   (1 - self.beta_1 ** (self.iterations + 1))
        # Update cache with squared current gradients
        layer.weight_cache = self.beta_2 * layer.weight_cache + \
                             (1 - self.beta_2) * layer.dweights ** 2
        layer.bias_cache = self.beta_2 * layer.bias_cache + \
                           (1 - self.beta_2) * layer.dbiases ** 2
        # Get corrected cache
        weight_cache_corrected = layer.weight_cache / \
                                 (1 - self.beta_2 ** (self.iterations + 1))
        bias_cache_corrected = layer.bias_cache / \
                               (1 - self.beta_2 ** (self.iterations + 1))

        # Vanilla SGD parameter update + normalization
        # with square rooted cache
        layer.weights += -self.current_learning_rate * \
                         weight_momentums_corrected / \
                         (np.sqrt(weight_cache_corrected) +
                          self.epsilon)
        layer.biases += -self.current_learning_rate * \
                        bias_momentums_corrected / \
                        (np.sqrt(bias_cache_corrected) +
                         self.epsilon)
"""


# Adam optimizer
class GAdam:

    # Initialize optimizer - set settings
    def __init__(self, time_series=False, learning_rate=0.001, decay=0., epsilon=1e-7,
                 beta_1=0.9, beta_2=0.999):
        self.time_series = time_series
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.iterations = 0
        self.epsilon = epsilon
        self.beta_1 = beta_1
        self.beta_2 = beta_2

    # Call once before any parameter updates
    def pre_update_params(self):
        if self.decay:
            self.current_learning_rate = self.learning_rate * \
                                         (1. / (1. + self.decay * self.iterations))

    # Update parameters
    def update_params(self, layer):

        # If layer does not contain cache arrays,
        # create them filled with zeros
        if not hasattr(layer, 'weight_cache'):
            layer.weight_momentums = {}
            layer.weight_cache = {}
            for laer in layer.weights:
                layer.weight_momentums[laer] = np.zeros_like(layer.weights[laer])
                layer.weight_cache[laer] = np.zeros_like(layer.weights[laer])

            layer.bias_momentums = np.zeros_like(layer.biases)
            layer.bias_cache = np.zeros_like(layer.biases)
        else:
            # so the weights and biases already have their momentums and their caches
            # need to make sure that the weight momentums and caches are connected to the correct layers
            # incase the graph got updated

            # starting with removing any connection that no longer exists
            for laer in layer.weight_momentums.keys():
                if laer not in layer.weights.keys():
                    del layer.weight_momentums[laer]
                    del layer.weight_cache[laer]

            # then add any new connections that weren't there before
            for laer in layer.weights.keys():
                if laer not in layer.weight_momentums.keys():
                    layer.weight_momentums[laer] = np.zeros_like(layer.weights[laer])
                    layer.weight_cache[laer] = np.zeros_like(layer.weights[laer])

        # Update momentum with current gradients
        for laer in layer.weight_momentums:
            layer.weight_momentums[laer] = self.beta_1 * layer.weight_momentums[laer] + \
                                           (1 - self.beta_1) * layer.dWeights[laer]

        layer.bias_momentums = self.beta_1 * layer.bias_momentums + (1 - self.beta_1) * layer.dBiases

        # Get corrected momentum
        # self.iteration is 0 at first pass
        # and we need to start with 1 here
        weight_momentums_corrected = {}
        for laer in layer.weight_momentums:
            weight_momentums_corrected[laer] = layer.weight_momentums[laer] / \
                                               (1 - self.beta_1 ** (self.iterations + 1))
            if self.time_series:
                # without a and b parameters for time series
                weight_momentums_corrected[laer] *= min(1, np.tanh(1 / (self.iterations + 1))
                                                        / abs(np.sum(weight_momentums_corrected[laer])))

        bias_momentums_corrected = layer.bias_momentums / \
                                   (1 - self.beta_1 ** (self.iterations + 1))

        # Update cache with squared current gradients
        for laer in layer.weight_cache:
            layer.weight_cache[laer] = self.beta_2 * layer.weight_cache[laer] + \
                                       (1 - self.beta_2) * layer.dWeights[laer] ** 2
        layer.bias_cache = self.beta_2 * layer.bias_cache + \
                           (1 - self.beta_2) * layer.dBiases ** 2

        # Get corrected cache
        weight_cache_corrected = {}
        for laer in layer.weight_momentums:
            weight_cache_corrected[laer] = layer.weight_cache[laer] / \
                                           (1 - self.beta_2 ** (self.iterations + 1))
            if self.time_series:
                # without a and b parameters for time series
                weight_cache_corrected[laer] *= min(1, np.tanh(1 / (self.iterations + 1))
                                                    / abs(np.sum(weight_cache_corrected[laer])))

        bias_cache_corrected = layer.bias_cache / \
                               (1 - self.beta_2 ** (self.iterations + 1))

        # Vanilla SGD parameter update + normalization
        # with square rooted cache
        for laer in layer.weights:
            layer.weights[laer] += -self.current_learning_rate * weight_momentums_corrected[laer] / \
                                   (np.sqrt(weight_cache_corrected[laer]) + self.epsilon)
        layer.biases += -self.current_learning_rate * bias_momentums_corrected / \
                        (np.sqrt(bias_cache_corrected) + self.epsilon)

    # Call once after any parameter updates
    def post_update_params(self):
        self.iterations += 1

    def copy(self):
        return GAdam(self.time_series, self.learning_rate, self.decay, self.epsilon, self.beta_1, self.beta_2)


# maybe this helps
class GAidan:

    # Initialize optimizer - set settings
    def __init__(self, learning_rate=0.001, decay=0., epsilon=1e-7,
                 beta_1=0.9, beta_2=0.999, alpha_1=100, alpha_2=10):
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.iterations = 0
        self.epsilon = epsilon
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.alpha_1 = alpha_1
        self.alpha_2 = alpha_2

    # Call once before any parameter updates
    def pre_update_params(self):
        if self.decay:
            self.current_learning_rate = self.learning_rate * \
                                         (1. / (1. + self.decay * self.iterations))

    # Update parameters
    def update_params(self, layer):

        # If layer does not contain cache arrays,
        # create them filled with zeros
        if not hasattr(layer, 'weight_cache'):
            layer.weight_momentums = {}
            layer.weight_cache = {}
            for laer in layer.weights:
                layer.weight_momentums[laer] = np.zeros_like(layer.weights[laer])
                layer.weight_cache[laer] = np.zeros_like(layer.weights[laer])

            layer.bias_momentums = np.zeros_like(layer.biases)
            layer.bias_cache = np.zeros_like(layer.biases)
        else:
            # so the weights and biases already have their momentums and their caches
            # need to make sure that the weight momentums and caches are connected to the correct layers
            # incase the graph got updated

            # starting with removing any connection that no longer exists
            for laer in layer.weight_momentums.keys():
                if laer not in layer.weights.keys():
                    del layer.weight_momentums[laer]
                    del layer.weight_cache[laer]

            # then add any new connections that weren't there before
            for laer in layer.weights.keys():
                if laer not in layer.weight_momentums.keys():
                    layer.weight_momentums[laer] = np.zeros_like(layer.weights[laer])
                    layer.weight_cache[laer] = np.zeros_like(layer.weights[laer])

        # Update momentum with current gradients
        for laer in layer.weight_momentums:
            layer.weight_momentums[laer] = self.beta_1 * layer.weight_momentums[laer] + \
                                           (1 - self.beta_1) * layer.dWeights[laer]

        layer.bias_momentums = self.beta_1 * layer.bias_momentums + (1 - self.beta_1) * layer.dBiases

        # Get corrected momentum
        # self.iteration is 0 at first pass
        # and we need to start with 1 here
        weight_momentums_corrected = {}
        for laer in layer.weight_momentums:
            weight_momentums_corrected[laer] = layer.weight_momentums[laer] / \
                                               (1 - self.beta_1 ** (self.iterations + 1))
            # with a and b parameters for time series
            weight_momentums_corrected[laer] *= min(1, np.tanh(1 / (self.iterations % self.alpha_1 + self.alpha_2))
                                                    / abs(np.sum(weight_momentums_corrected[laer])))

        bias_momentums_corrected = layer.bias_momentums / \
                                   (1 - self.beta_1 ** (self.iterations + 1))

        # Update cache with squared current gradients
        for laer in layer.weight_cache:
            layer.weight_cache[laer] = self.beta_2 * layer.weight_cache[laer] + \
                                       (1 - self.beta_2) * layer.dWeights[laer] ** 2
        layer.bias_cache = self.beta_2 * layer.bias_cache + \
                           (1 - self.beta_2) * layer.dBiases ** 2

        # Get corrected cache
        weight_cache_corrected = {}
        for laer in layer.weight_momentums:
            weight_cache_corrected[laer] = layer.weight_cache[laer] / \
                                           (1 - self.beta_2 ** (self.iterations + 1))
            # with a and b parameters for time series
            weight_cache_corrected[laer] *= min(1, np.tanh(1 / (self.iterations % self.alpha_1 + self.alpha_2))
                                                / abs(np.sum(weight_cache_corrected[laer])))

        bias_cache_corrected = layer.bias_cache / \
                               (1 - self.beta_2 ** (self.iterations + 1))

        # Vanilla SGD parameter update + normalization
        # with square rooted cache
        for laer in layer.weights:
            layer.weights[laer] += -self.current_learning_rate * weight_momentums_corrected[laer] / \
                                   (np.sqrt(weight_cache_corrected[laer]) + self.epsilon)
        layer.biases += -self.current_learning_rate * bias_momentums_corrected / \
                        (np.sqrt(bias_cache_corrected) + self.epsilon)

    # Call once after any parameter updates
    def post_update_params(self):
        self.iterations += 1

    def copy(self):
        return GAidan(self.learning_rate, self.decay, self.epsilon,
                      self.beta_1, self.beta_2, self.alpha_1, self.alpha_2)
