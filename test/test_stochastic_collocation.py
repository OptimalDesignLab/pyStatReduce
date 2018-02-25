# Test stochastic collocation module

import sys
import os

# Get the directory of this file
TEST_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = TEST_DIR + '/../src'
sys.path.insert(0, SRC_DIR)

import unittest
import numpy as np
import chaospy as cp

from stochastic_collocation import StochasticCollocation
from quantity_of_interest import QuantityOfInterest
import examples

class StochasticCollocationTest(unittest.TestCase):

    def test_stochasticCollocation2D(self):
        systemsize = 2
        x = np.zeros(systemsize)
        theta = 0
        sigma = np.array([0.2, 0.1])
        tuple = (theta,)
        jdist = cp.MvNormal(x, np.diag(sigma))

        # Create a Stochastic collocation object
        collocation = StochasticCollocation(3, "Normal")

        # Create a QoI object using ellpsoid
        QoI = examples.Paraboloid2D(systemsize, tuple)

        # Compute the expected value
        mu_j = collocation.normal.mean(x, sigma, QoI)

        # Test against nested loops
        mu_j_hat = 0.0
        sqrt2 = np.sqrt(2)
        for i in xrange(0, collocation.normal.q.size):
            for j in xrange(0, collocation.normal.q.size):
                f_val = QoI.eval_QoI(x, sqrt2*sigma*[collocation.normal.q[i],
                        collocation.normal.q[j]])
                mu_j_hat += collocation.normal.w[i]*collocation.normal.w[j]*f_val

        mu_j_hat = mu_j_hat/(np.sqrt(np.pi)**systemsize)

        diff = abs(mu_j - mu_j_hat)
        self.assertTrue(diff < 1.e-15)

        # Test Check for the variance
        # - Analytical Value
        cov_mat = cp.Cov(jdist)
        mat1 = np.matmul(QoI.quadratic_matrix, cov_mat)
        mat2 = np.matmul(QoI.quadratic_matrix, np.matmul(cov_mat,
                         QoI.quadratic_matrix.dot(x)))
        variance_analytical = 2*np.trace(np.matmul(mat1, mat1)) + 4*x.dot(mat2)

        # - Numerical Value
        sigma_j = collocation.normal.variance(QoI, jdist, mu_j)
        diff = abs(variance_analytical - sigma_j)
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
        mu_j = collocation.normal.mean(x, sigma, QoI)

        # Test against nested loops
        mu_j_hat = 0.0
        sqrt2 = np.sqrt(2)
        for i in xrange(0, collocation.normal.q.size):
            for j in xrange(0, collocation.normal.q.size):
                for k in xrange(0, collocation.normal.q.size):
                    f_val = QoI.eval_QoI(x, sqrt2*sigma*[collocation.normal.q[i],
                            collocation.normal.q[j], collocation.normal.q[k]])
                    mu_j_hat += collocation.normal.w[i]*collocation.normal.w[j]* \
                                collocation.normal.w[k]*f_val

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
        mu_j = collocation.normal.mean(x, sigma, QoI)

        # Test against nested loops
        mu_j_hat = 0.0
        sqrt2 = np.sqrt(2)
        q = collocation.normal.q
        w = collocation.normal.w
        for i in xrange(0, q.size):
            for j in xrange(0, q.size):
                for k in xrange(0, q.size):
                    for l in xrange(0, q.size):
                        for m in xrange(0, q.size):
                            fval = QoI.eval_QoI(x, sqrt2*sigma*[q[i], q[j], q[k],
                                   q[l], q[m]])
                            mu_j_hat += w[i]*w[j]*w[k]*w[l]*w[m]*fval

        mu_j_hat = mu_j_hat/(np.sqrt(np.pi)**systemsize)

        diff = abs(mu_j - mu_j_hat)
        self.assertTrue(diff < 1.e-15)

if __name__ == "__main__":
    unittest.main()
