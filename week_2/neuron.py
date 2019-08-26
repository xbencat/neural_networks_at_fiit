import numpy as np

from solutions.week_2.model import LinearRegressionModel


class Neuron(LinearRegressionModel):

    def sigma(self, x):
        e = np.exp(x)
        return e / (e + 1)

    def dsigma(self, x):
        sigma = self.sigma(x)
        return sigma * (1 - sigma)

    def predict(self, x):
        return self.sigma(LinearRegressionModel.predict(self, x))
        ...  # FIXME: 2.14

    def compute_gradients(self, xs, ys):
        dw = np.mean([-2 * (y - self.predict(x)) * x * self.dsigma(self.predict(x)) for x, y in zip(xs, ys)], axis=0)  # FIXME: See 2.9.2
        db = np.mean([-2 * (y - self.predict(x)) * self.dsigma(self.predict(x)) for x, y in zip(xs, ys)])
        return dw, db
        ...  # FIXME: 2.14


x = [-2.7, -2.3, -1.9, -1.2, -0.2, 1.0, 1.5, 2.1]
for x_ in x:
    def sigma(a):
        e = np.exp(a)
        return e / (e + 1)
    print(
        x_,
        ",",
        sigma(2.5 * x_ + 2) + (np.random.rand() / 10))

