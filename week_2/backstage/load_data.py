import types

import numpy as np


def load_data(filename):
    data = np.genfromtxt(f'data/{filename}', delimiter=',')
    output = types.SimpleNamespace()
    output.x = data[:, :-1]
    output.y = np.squeeze(data[:, -1])
    return output