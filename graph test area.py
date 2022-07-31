import numpy as np
import nnfs
from nnfs.datasets import spiral_data
from Layers import *
from Activations import *
from Losses import *
from Optimizers import *
from Models import *
from Metrics import *

nnfs.init()

# Create dataset
X, y = spiral_data(samples=1000, classes=3)
X_test, y_test = spiral_data(samples=100, classes=3)

# Shuffle the dataset so mini batches aren't all a single class
ind = np.arange(len(X))
np.random.shuffle(ind)
X = X[ind]
y = y[ind]


# Instantiate the model
model = GraphModel()

# Add layers
l1 = GraphDense(128, Relu())  # , weight_regularizer_l2=5e-4, bias_regularizer_l2=5e-4)
l2 = GraphDense(128, Relu())
l3 = GraphDense(3, Softmax())
l4 = GraphDense(3, Softmax())
g = {
    l1: [],
    l2: [l1],
    l3: [l1],
    l4: [l2, l3]
}
model.add(g, [l1], [l4], 2)

# Set loss, optimizer and accuracy objects
model.set(
    loss=CategoricalCrossentropy(),
    optimizer=GraphAdam(learning_rate=0.05, decay=5e-5),
    accuracy=Accuracy_Categorical()
)

# Finalize the model
model.finalize()

# Train the model
model.train(X, y,
            validation_data=(X_test, y_test),
            epochs=200,
            time_series=False,
            print_every=10)
