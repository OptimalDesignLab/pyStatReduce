# Test stochastic collocation module

import sys
sys.path.insert(0, '/Users/kinshuk/Documents/ODL/pyStatReduce/src')

import unittest
import numpy as np
import chaospy as cp

from StochasticCollocation import StochasticCollocation
from QuantityOfInterest import QuantityOfInterest
import examples

class StochasticCollocationTest(unittest.TestCase):

    def test_stochasticCollocation2D(self):
        systemsize = 2
        x = np.zeros(systemsize)
        theta = 0
        sigma = np.array([0.2, 0.1])
        tuple = (theta,)

        # Create a Stochastic collocation object
        collocation = StochasticCollocation(3, "Normal")

        # Create a QoI object using ellpsoid
        QoI = examples.Paraboloid2D(tuple)

        # Compute the expected value
        mu_j = collocation.normal(x, sigma, QoI, collocation)

        # Test against nested loops
        mu_j_hat = 0.0

        sqrt2 = np.sqrt(2)
        for i in xrange(0, collocation.q.size):
            for j in xrange(0, collocation.q.size):
                f_val = QoI.eval_QoI(x, sqrt2*sigma*[collocation.q[i], collocation.q[j]])
                mu_j_hat += collocation.w[i]*collocation.w[j]*f_val

        mu_j_hat = mu_j_hat/(np.sqrt(np.pi)**systemsize)

        diff = abs(mu_j - mu_j_hat)
        self.assertTrue(diff < 1.e-15)


if __name__ == "__main__":
    unittest.main()
