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

    def test_stochasticCollocation3D(self):
        systemsize = 3
        x = np.zeros(systemsize)
        sigma = 0.2*np.ones(systemsize)

        # Create a Stochastic collocation object
        collocation = StochasticCollocation(3, "Normal")

        # Create a QoI object using ellpsoid
        QoI = examples.Paraboloid3D(systemsize)

        # Compute the expected value
        mu_j = collocation.normal(x, sigma, QoI, collocation)

        # Test against nested loops
        mu_j_hat = 0.0
        sqrt2 = np.sqrt(2)
        for i in xrange(0, collocation.q.size):
            for j in xrange(0, collocation.q.size):
                for k in xrange(0, collocation.q.size):
                    f_val = QoI.eval_QoI(x, sqrt2*sigma*[collocation.q[i],
                            collocation.q[j], collocation.q[k]])
                    mu_j_hat += collocation.w[i]*collocation.w[j]* \
                                collocation.w[k]*f_val

        mu_j_hat = mu_j_hat/(np.sqrt(np.pi)**systemsize)

        diff = abs(mu_j - mu_j_hat)
        self.assertTrue(diff < 1.e-15)

    def test_stochasticCollocation5D(self):
        systemsize = 5
        x = np.random.rand(systemsize)
        sigma = 0.2*np.ones(systemsize)

        # Create a Stochastic collocation object
        collocation = StochasticCollocation(3, "Normal")

        # Create a QoI object using ellpsoid
        QoI = examples.Paraboloid3D(systemsize)

        # Compute the expected value
        mu_j = collocation.normal(x, sigma, QoI, collocation)

        # Test against nested loops
        mu_j_hat = 0.0
        sqrt2 = np.sqrt(2)
        q = collocation.q
        w = collocation.w
        for i in xrange(0, collocation.q.size):
            for j in xrange(0, collocation.q.size):
                for k in xrange(0, collocation.q.size):
                    for l in xrange(0, collocation.q.size):
                        for m in xrange(0, collocation.q.size):
                            fval = QoI.eval_QoI(x, sqrt2*sigma*[q[i], q[j], q[k],
                                   q[l], q[m]])
                            mu_j_hat += w[i]*w[j]*w[k]*w[l]*w[m]*fval

        mu_j_hat = mu_j_hat/(np.sqrt(np.pi)**systemsize)

        diff = abs(mu_j - mu_j_hat)
        self.assertTrue(diff < 1.e-15)

if __name__ == "__main__":
    unittest.main()
