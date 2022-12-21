# import numpy as np
from Layers import *
from Activations import *
from Losses import *
from Optimizers import *
from Models import *
from Metrics import *
from Datasets import Goog
import matplotlib.pyplot as plt

dataset = Goog()
data = dataset.get_data()

# Instantiate the model
model = GraphModel()

# ### Add layers
l1 = GraphDense(16, Sigmoid())  # , weight_regularizer_l2=5e-4, bias_regularizer_l2=5e-4)
l2 = GraphDense(4, Sigmoid())
l3 = GraphDense(1, Linear())
g = {
    l1: [],
    l2: [l1, l2],
    l3: [l2]
}
model.add(g, [l1], [l3], dataset.f_set_size) # give datasets a feature count field

# Set loss, optimizer and accuracy objects
model.set(
    loss=MeanSquaredError(),
    optimizer=GAidan(learning_rate=0.02, decay=5e-5),
    accuracy=Accuracy_Regression()
)

# Finalize the model
model.finalize()

# Train the model
model.train(data["trn"]["X"], data["trn"]["Y"],
            validation_data=(data["val"]["X"], data["val"]["Y"]),
            epochs=2,
            time_series=True,
            print_every=1)

cop = model.copy()

cop.train(data["trn"]["X"], data["trn"]["Y"],
          validation_data=(data["val"]["X"], data["val"]["Y"]),
          epochs=2,  # batch_size=1,
          time_series=True,
          print_every=1)

model.train(data["trn"]["X"], data["trn"]["Y"],
            validation_data=(data["val"]["X"], data["val"]["Y"]),
            epochs=2,  # batch_size=1,
            time_series=True,
            print_every=1)

l4 = GraphDense(12, Relu())
g[l4] = [l1]
g[l3] = [l2, l4]
model.add(g)

model.train(data["trn"]["X"], data["trn"]["Y"],
            validation_data=(data["val"]["X"], data["val"]["Y"]),
            epochs=2,  # batch_size=1,
            time_series=True,
            print_every=1)


