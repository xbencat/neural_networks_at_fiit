import itertools

import numpy as np
import matplotlib.pyplot as plt


def derivatives_plot():
    xlist = np.linspace(-3.0, 3.0, 20)
    ylist = np.linspace(-3.0, 3.0, 20)
    X, Y = np.meshgrid(xlist, ylist)
    Z = np.sin(X) * np.cos(Y)

    fig, axes = plt.subplots(1, 3, squeeze=True)

    for i, ax in enumerate(axes):
        ax.axis([-3, 3, -3, 3])
        if i == 1: ax.set_title(r'$f(x, y) = \sin(x) \cos(y)$')
        ax.set_xlabel('$x$')
        if i == 0: ax.set_ylabel('$y$')
        ax.contourf(X, Y, Z, levels=[i/5 - 1 for i in range(10)], extend='both')
        for x, y in itertools.product(xlist, ylist):
            ax.arrow(x, y, (i != 0) * np.cos(x)*np.cos(y) / 6, (i != 1) * -np.sin(x)*np.sin(y) / 6,
                color='white',
                head_width=0.06,
                head_length=0.04,
                width=0.0005)

    plt.show()
