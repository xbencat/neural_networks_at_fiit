import types
import unittest

import numpy as np

from solutions.week_2.model import LinearRegressionModel


class TestSGD(unittest.TestCase):

    def setUp(self):

        self.model = LinearRegressionModel(input_dim=3)
        self.data = (
            np.random.rand(9, 3),
            np.array([i for i in range(9)])
        )
        self.data_dict = {y: x for x, y in zip(*self.data)}
        self.model.batch_log = []

        def wrap_step(func):
            def new_step(self, xs, ys):
                func(xs, ys)
                self.batch_log.append((xs, ys))
            return new_step

        self.model.step = types.MethodType(wrap_step(self.model.step), self.model)

    def test_batch_count(self):
        """
        Tests if expected number of training steps is performed
        """
        self.model.stochastic_gradient_descent(*self.data, num_epochs=3, batch_size=2)
        self.assertEqual(
            len(self.model.batch_log),
            15)

        self.model.batch_log = []
        self.model.stochastic_gradient_descent(*self.data, num_epochs=5, batch_size=3)
        self.assertEqual(
            len(self.model.batch_log),
            15)

    def test_sample_coupling(self):
        """
        Tests if  sample features and labels stay together
        """
        self.model.stochastic_gradient_descent(*self.data, num_epochs=3, batch_size=2)
        for batch in self.model.batch_log:
            for x, y in zip(*batch):
                self.assertListEqual(
                    list(x),
                    list(self.data_dict[y]))

    def test_shuffling(self):
        """
        Tests whether the shuffling is done correctly
        """
        self.model.stochastic_gradient_descent(*self.data, num_epochs=5, batch_size=2)
        ys = np.concatenate([
            y for _, y in self.model.batch_log
        ])
        for i in range(5):
            self.assertListEqual(
                sorted(ys[i * 9: i * 9 + 9]),
                [j for j in range(9)])
        self.assertFalse(all(
            np.array_equal(ys[:9], ys[i * 9: i * 9 + 9])
            for i in range(1, 5)
        ))
