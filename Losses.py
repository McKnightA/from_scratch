import numpy as np


class MSE:
    def __init__(self):
        self.name = "MSE"

    def forward(self, pred, label):
        return 1/2 * np.square(pred - label)

    def backward(self, pred, label):
        return pred - label


class BCE:
    def __init__(self):
        self.name = "BCE"

    def forward(self, pred, label):
        return -1 * (label * np.log(pred) + (1-label) * np.log(1 - pred))

    def backward(self, pred, label):
        return -1 * (label * (1 - pred) - (1 - label) * pred)


class MCCE:
    def __init__(self):
        self.name = "MCCE"

    def forward(self, pred, label):
        return -1 * (label * np.log(pred)).sum()

    def backward(self, pred, label):
        return -1 * (label * (1 - pred)).sum()
