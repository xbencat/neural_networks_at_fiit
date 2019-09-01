import numpy as np


class MLP:

    ...


a = MLP(2, 100, 2)
a.sgd(np.random.rand(50, 2) - 0.5, np.random.rand(50, 2))
