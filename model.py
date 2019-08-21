import numpy as np


class Model:

    def __init__(self, W, b):
        # Capital letter variables will mark vectors or matrices in our code
        self.W = W
        self.b = b

    # E2.5
    def predict(self, X):
        return np.matmul(X, self.W) + self.b


