import itertools
from time import sleep

from ipywidgets import *
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d, Axes3D


def one_d_plot():
    fig = plt.figure()
    X = np.linspace(-3.0, 3.0, 2)

    def show_plot(
            w1=widgets.FloatSlider(min=-10, max=10, step=0.1, value=1.5),
            b=(-5.0, 5.0, 0.1),
            use_bias=True):
        plt.clf()

        Y = w1 * X
        if use_bias:
            Y += b

        ax = fig.add_subplot(1, 1, 1)
        ax.plot(X, Y, scalex=False, scaley=False)
        ax.axis([-3, 3, -3, 3])
        ax.grid()
        ax.set_xlabel('$x_1$')
        ax.set_ylabel('$\hat{y}$')
        ax.set_title('$f(x) = \hat{y} = w_1x + b$')

        fig.canvas.draw_idle()

    interact(show_plot)


def two_d_plot():
    xlist = np.linspace(-3.0, 3.0, 10)
    ylist = np.linspace(-3.0, 3.0, 10)
    X, Y = np.meshgrid(xlist, ylist)

    fig = plt.figure()

    def show_plot(
            w1=widgets.FloatSlider(min=-3, max=3, step=0.1, value=1.5),
            w2=widgets.FloatSlider(min=-3, max=3, step=0.1, value=1.5),
            b=widgets.FloatSlider(min=-3, max=3, step=0.1, value=0),):
        plt.clf()

        Z = w1 * X + w2 * Y + b

        ax = fig.add_subplot(1, 2, 1)
        ax.axis([-3, 3, -3, 3])
        ax.set_title(r'$f(x) = \hat{y} = w_1x_1 + w_2x_2 + b$') # TODO: make the x bold
        ax.set_xlabel('$x_1$')
        ax.set_ylabel('$x_2$')
        cf = ax.contourf(X, Y, Z, levels=[i - 10 for i in range(20)], extend='both')
        cbar = fig.colorbar(cf)
        cbar.ax.set_ylabel('$\hat{y}$', rotation=270)

        ax = fig.add_subplot(1, 2, 2,
            projection='3d')
        ax.plot_surface(X, Y, Z,
            cmap='viridis',
            linewidth=0,
            antialiased=False,
            vmin=-10,
            vmax=10)
        ax.axis([-3, 3, -3, 3])
        ax.set_xlabel('$x_1$')
        ax.set_ylabel('$x_2$')
        ax.set_zlabel('$\hat{y}$')
        ax.set_xlim3d(-3, 3)
        ax.set_ylim3d(-3, 3)
        ax.set_zlim3d(-10, 10)
        fig.canvas.draw_idle()

    interact(show_plot)

def manual_fit_plot(show_loss=False):
    fig = plt.figure()

    true_w = -0.35
    true_b = 1
    scatter_X = np.array([-2, -0.2, 0, 0.9, 2.6])
    noise = np.array([0.1, 0.1, -0.1, -0.1, 0.12])
    scatter_Y = scatter_X * true_w + true_b + noise

    def show_plot(
            w1=widgets.FloatSlider(min=-10, max=10, step=0.1, value=1.5),
            b=(-5.0, 5.0, 0.1)):
        plt.clf()

        X = np.linspace(-3.0, 3.0, 2)
        Y = w1 * X + b

        ax = fig.add_subplot(1, 1, 1)
        ax.plot(X, Y, scalex=False, scaley=False)
        ax.scatter(scatter_X, scatter_Y, color='red')
        ax.axis([-3, 3, -3, 3])
        ax.grid()
        ax.set_xlabel('$x_1$')
        ax.set_ylabel('$\hat{y}$')
        ax.set_title('$f(x) = \hat{y} = w_1x + b$')
        if show_loss:
            pred_Y = scatter_X * w1 + b
            errors = (pred_Y - scatter_Y)**2
            loss = np.mean(errors)
            ax.text(1, -2.9, f'Current loss: {loss:.4f}')
            for x, y, error in zip(scatter_X, scatter_Y, errors):
                ax.text(x + 0.1, y + 0.1, f'{error:.4f}')

        fig.canvas.draw_idle()

    interact(show_plot)


def derivatives_plot():
    fig = plt.figure()

    xlist = np.linspace(-3.0, 3.0, 20)
    ylist = np.linspace(-3.0, 3.0, 20)
    X, Y = np.meshgrid(xlist, ylist)
    Z = np.sin(X) * np.cos(Y)

    ax = fig.add_subplot(1, 1, 1)
    ax.axis([-3, 3, -3, 3])
    ax.set_title(r'$f(x, y) = \sin(x) \cos(y)$')  # TODO: make the x bold
    ax.set_xlabel('$x$')
    ax.set_ylabel('$y$')
    cf = ax.contourf(X, Y, Z, levels=[i/5 - 1 for i in range(10)], extend='both')
    cbar = fig.colorbar(cf)
    cbar.ax.set_ylabel('$f(x,y)$', rotation=270)
    for x, y in itertools.product(xlist, ylist):
        ax.arrow(x, y, -np.cos(x)*np.cos(y) / 5, np.sin(x)*np.sin(y) / 5, color='black', head_width=0.05)


def gd_plot():

    fig = plt.figure()

    xlist = np.linspace(-3.0, 3.0, 20)
    ylist = np.linspace(-3.0, 3.0, 20)
    X, Y = np.meshgrid(xlist, ylist)
    a = 1.737
    b = 1.232
    Z = a*X**2 + b*Y**2

    def show_background():
        plt.clf()
        ax = fig.add_subplot(1, 1, 1)
        ax.axis([-2, 2, -2, 2])
        ax.set_title(r'$f(x, y) = \sin(x) \cos(y)$')  # TODO: make the x bold
        ax.set_xlabel('$x$')
        ax.set_ylabel('$y$')
        cf = ax.contourf(X, Y, Z, levels=[i / 2 for i in range(20)], extend='both')
        cbar = fig.colorbar(cf)
        cbar.ax.set_ylabel('$f(x,y)$', rotation=270)
        return ax

    def show_plot(alpha=0.3):
        ax = show_background()
        p = np.array([0.87, 1.23])
        for _ in range(10):
            d = -alpha*2*p
            d[0] *= a
            d[1] *= b
            ax.arrow(*p, *d, color='black')
            p = p + d
            fig.canvas.draw()
            sleep(0.5)

    show_background()
    interact_manual(show_plot)