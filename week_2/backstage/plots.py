import itertools
import types
from time import sleep

from ipywidgets import *
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d, Axes3D

from week_2.backstage.load_data import load_data
from solutions.week_2.model import LinearRegressionModel


def one_d_plot():
    fig = plt.figure()
    X = np.linspace(-3.0, 3.0, 2)

    def show_plot(
            w1=widgets.FloatSlider(min=-10, max=10, step=0.1, value=1.5),
            b=(-5.0, 5.0, 0.1),
            use_bias=True):
        print(repr(fig))
        fig.clf()

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

    fig = plt.figure(figsize=(9, 5))

    def show_plot(
            w1=widgets.FloatSlider(min=-3, max=3, step=0.1, value=1.5),
            w2=widgets.FloatSlider(min=-3, max=3, step=0.1, value=1.5),
            b=widgets.FloatSlider(min=-3, max=3, step=0.1, value=0),):
        fig.clf()

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

    data = load_data("toy_1.csv")
    scatter_X, scatter_Y = data.x, data.y

    def show_plot(
            w1=widgets.FloatSlider(min=-10, max=10, step=0.1, value=1.5),
            b=(-5.0, 5.0, 0.1)):
        fig.clf()

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
            pred_Y = np.dot(np.squeeze(scatter_X), w1) + b
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
    a, b = 1, 1.2
    Z = a*X**2 + b*Y**2

    def show_background():
        fig.clf()
        ax = fig.add_subplot(1, 1, 1)
        ax.axis([-2, 2, -2, 2])
        ax.set_title(r'$f(x,y) = x^2 + 1.2y^2$')  # TODO: make the x bold
        ax.set_xlabel('$x$')
        ax.set_ylabel('$y$')
        cf = ax.contourf(X, Y, Z, levels=[i / 2 for i in range(20)], extend='both')
        cbar = fig.colorbar(cf)
        cbar.ax.set_ylabel('$f(x,y)$', rotation=270)
        p = np.array([0.87, 0.52])
        ax.plot(*p, 'w.')
        return ax, p

    def show_plot(alpha=[0.01, 0.3, 0.7, 1.2, 'custom'], custom=0.3):
        if alpha == 'custom':
            alpha = custom
        ax, p = show_background()
        txt = ax.text(1, -1.75, '', color='white')
        num_steps = 10
        for i in range(num_steps):
            d = -alpha*2*p
            d[0] *= a
            d[1] *= b
            ax.arrow(*p, *d, color='white')
            p = p + d
            txt.set_text(f'Step: {i+1}/10' if i + 1 < num_steps else 'Done')
            fig.canvas.draw()
            sleep(0.5)

    show_background()
    interact.options(manual=True, manual_name='Run gradient descent')(show_plot)


def add_magic(model, speed=2):

    model._m = types.SimpleNamespace()
    ends = np.array([3, -3])

    def wrap_step(model_step):
        def new_step(self, xs, ys):
            self._m.ax1.plot(self.w, self.b, 'w.')
            self._m.ax2_line.set_ydata([self.predict(end) for end in ends])
            self._m.txt.set_text(f'w: {self.w[0]:.2f}\nb: {self.b:.2f}\nloss: {self.loss(*self._m.data):.3f}')

            model_step(xs, ys)
            self._m.fig.canvas.draw()
            sleep(1/speed)

        return new_step
    model.step = types.MethodType(wrap_step(model.step), model)

    def wrap_gd(model_gd):
        def new_gd(self, xs, ys, **kwargs):

            self._m.fig = plt.figure(figsize=(9, 4))

            self._m.ax1 = self._m.fig.add_subplot(1, 2, 1)
            self._m.ax1.axis([-3, 3, -3, 3])
            self._m.ax1.set_title(r'$L$')
            self._m.ax1.set_xlabel('$w_1$')
            self._m.ax1.set_ylabel('$b$')
            self._m.txt = self._m.ax1.text(-2.8, 2.0, '', color='white')
            self._m.data = (xs, ys)

            xlist = np.linspace(-3.0, 3.0, 10)
            ylist = np.linspace(-3.0, 3.0, 10)
            X, Y = np.meshgrid(xlist, ylist)
            Z = np.zeros_like(X)
            w, b = model.w, model.b
            for i, j in itertools.product(range(Z.shape[0]), range(Z.shape[1])):
                model.w = xlist[i]
                model.b = ylist[j]
                Z[j, i] = self.loss(xs, ys)
            model.w, model.b = w, b
            self._m.ax1.contourf(X, Y, Z, levels=[i for i in range(0, int(0.8*np.max(Z)))], extend='both')

            self._m.ax2 = self._m.fig.add_subplot(1, 2, 2)

            self._m.ax2.scatter(xs, ys, color='red')
            self._m.ax2.axis([-3, 3, -3, 3])
            self._m.ax2.grid()
            self._m.ax2.set_xlabel('$x_1$')
            self._m.ax2.set_ylabel('$y$')
            self._m.ax2.set_title('$\hat{y} = w_1x_1 + b$')
            self._m.ax2_line, = self._m.ax2.plot(ends, [self.predict(end) for end in ends], scalex=False, scaley=False)
            self._m.fig.canvas.draw()

            model_gd(xs, ys, **kwargs)
        return new_gd
    model.gradient_descent = types.MethodType(wrap_gd(model.gradient_descent), model)
    if hasattr(model, 'stochastic_gradient_descent'):
        model.stochastic_gradient_descent = types.MethodType(wrap_gd(model.stochastic_gradient_descent), model)

    
def stochastic_plot():

    data = load_data('toy_1.csv')
    xs, ys = data.x, data.y

    fig, axes = plt.subplots(2, 3, sharex=True, sharey=True)
    fig.subplots_adjust(wspace=0.2)
    fig.subplots_adjust(hspace=0.5)
    axes = np.reshape(axes, (6,))
    model = LinearRegressionModel(input_dim=1, w=[2], b=2)
    xlabels=['', '', '', '$w_1$', '$w_1$', '$w_1$', ]
    ylabels=['b', '', '', 'b', '', '', ]
    titles = [f'$L^{{({i})}}$' for i in range(1,6)]
    titles.append(r'$L = \frac{1}{5}\sum_{i=1}^5L^{(i)}$')
    for ax_id, ax in enumerate(axes):
        xlist = np.linspace(-3.0, 3.0, 10)
        ylist = np.linspace(-3.0, 3.0, 10)
        X, Y = np.meshgrid(xlist, ylist)
        Z = np.zeros_like(X)
        ax.set_xlabel(xlabels[ax_id])
        ax.set_ylabel(ylabels[ax_id])
        ax.set_title(titles[ax_id])
        if ax_id == len(data.y):
            ax_xs, ax_ys = data.x, data.y
        else:
            ax_xs, ax_ys = xs[ax_id, :], [ys[ax_id]]

        w, b = model.w, model.b
        for i, j in itertools.product(range(Z.shape[0]), range(Z.shape[1])):
            model.b = ylist[j]
            model.w = xlist[i]
            Z[j, i] = model.loss(ax_xs, ax_ys)
        model.w, model.b = [w], b

        ax.contourf(X, Y, Z, levels=[i for i in range(0, 35)], extend='both')

    arrows = []

    def show_arrows(
        w1=widgets.FloatSlider(min=-3, max=3, step=0.1, value=2),
        b=widgets.FloatSlider(min=-3, max=3, step=0.1, value=-2)):
        for arr in arrows:
            arr.remove()
        arrows.clear()
        for ax_id, ax in enumerate(axes):
            if ax_id == len(data.y):
                ax_xs, ax_ys = data.x, data.y
            else:
                ax_xs, ax_ys = xs[ax_id, :], [ys[ax_id]]

            model.w, model.b = [w1], b
            dw, db = model.compute_gradients(ax_xs, ax_ys)
            arrows.append(ax.arrow(model.w, model.b, -dw[0] * 0.1, -db * 0.1, color='white', head_width=0.2))

    interact(show_arrows)
