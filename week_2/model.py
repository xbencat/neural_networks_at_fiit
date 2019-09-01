import numpy as np


class LinearRegressionModel:
    """
    Class for Linear Regression model

    Attributes
    ----------
    w : np.array dim=(input_dim)
        An array of weights.
    b : float
        Bias parameter.
    input_dim : int
        Number of features in samples
    lr : float
        Learning rate for gradient descent algorithm
    """

    def __init__(self, input_dim, learning_rate=0.03, w=None, b=.0):
        self.input_dim = input_dim
        self.lr = learning_rate
        self.w = np.zeros(input_dim) if w is None else w
        self.b = b

    def predict(self, x):
        """
        Runs the linear regression model over input vector x and computes the predicted value y^

        :param x: np.array dim=(input_dim)
        :return: float
        """
        return ...  # FIXME: 2.9.1

    def gradients(self, xs, ys):
        """
        Computes the derivatives of loss function L w.r.t parameters w and b. Note that the attributes xs and ys are
        not one sample, but multiple samples. xs is a matrix, where i-th row is i-th sample feature vector and
        ys is a vector, where i-th element is a true score for i-th sample.

        xs and ys will have the same intrepretation for following methods as well.

        :param xs:  2D np.array dim=(num_samples, input_dim)
        :param ys:  np.array dim=(num_samples)
        :return:    np.array dim=(input_dim), float
        """
        dw = ...  # FIXME: 2.9.2
        db = ...
        return dw, db

    def gradient_descent(self, xs, ys, num_steps=100):
        """
        Performs the gradient descent algorithm for num_steps steps. Returns the final parameters and
        loss function value.

        :param      num_steps: int
        :return:    np.array dim=(input_dim), float, float
        """
        for _ in range(num_steps):
            self.step(xs, ys)

    def step(self, xs, ys):
        """
        Performs one gradient descent step and updates the parameters accordingly.
        """
        dw, db = self.compute_gradients(xs, ys)
        self.w = ...  # FIXME: 2.9.3
        self.b = ...

    def loss(self, xs, ys):
        """
        Calculates the loss L with current parameters for given data.

        :return: float
        """
        return ...  # FIXME: 2.9.4


if __name__ == '__main__':
    import week_2.tests.hub as tests
    tests.model_test()
    # tests.sgd_test() # FIXME: Uncomment this line for programming assignment 2.13














