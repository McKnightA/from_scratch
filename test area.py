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

# Instantiate the model
model = Model()

# Add layers
model.add([Dense(2, 512), Relu()])
model.add(Dense(512, 3))
model.add(Softmax())

# Set loss, optimizer and accuracy objects
model.set(
    loss=CategoricalCrossentropy(),
    optimizer=Adam(learning_rate=0.05, decay=5e-5),
    accuracy=Accuracy_Categorical()
)

# Finalize the model
model.finalize()

# Train the model
model.train(X, y, validation_data=(X_test, y_test),
            epochs=1000, print_every=100)
