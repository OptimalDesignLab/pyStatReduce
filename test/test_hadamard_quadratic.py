import sys
import os

# Get the directory of this file
TEST_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = TEST_DIR + '/../src'
sys.path.insert(0, SRC_DIR)

import unittest
import numpy as np
import chaospy as cp
import numdifftools as nd

from stochastic_collocation import StochasticCollocation
from quantity_of_interest import QuantityOfInterest
import examples

class HadamardQuadraticTest(unittest.TestCase):

    def test_syntheticEigenValues(self):
        systemsize = 2
        eigen_decayrate = 0.5
        # Create Hadmard Quadratic object
        QoI = examples.HadamardQuadratic(systemsize, eigen_decayrate)

        true_value = np.array([1, 1/np.sqrt(2)])
        self.assertTrue((QoI.eigen_vals == true_value).all())

    def test_syntheticEigenVectors(self):
        systemsize = 4
        eigen_decayrate = 0.5
        QoI = examples.HadamardQuadratic(systemsize, eigen_decayrate)

        true_value = np.array([[0.5, 0.5, 0.5, 0.5],
                               [0.5, -0.5, 0.5, -0.5],
                               [0.5, 0.5, -0.5, -0.5],
                               [0.5, -0.5, -0.5, 0.5]])
        error_vals = QoI.eigen_vectors - true_value
        self.assertTrue((error_vals < 1.e-15).all())

    def test_evalQoI(self):
        systemsize = 4
        eigen_decayrate = 2.0
        QoI = examples.HadamardQuadratic(systemsize, eigen_decayrate)
        mu = np.ones(systemsize)
        xi = np.ones(systemsize)
        fval = QoI.eval_QoI(mu, xi)

        true_value = 16.0
        err = abs(fval - true_value)
        self.assertTrue(err < 1.e-14)

    def test_eval_analytical_QoI_mean(self):
        systemsize = 4
        eigen_decayrate = 2.0
        QoI = examples.HadamardQuadratic(systemsize, eigen_decayrate)
        mu = np.ones(systemsize)
        Sigma = 0.2*np.eye(systemsize) # Covariance matrix
        fval = QoI.eval_analytical_QoI_mean(mu, Sigma)

        # Check against stochastic collocation
        collocation = StochasticCollocation(3, "Normal")
        sigma = np.diagonal(np.sqrt(Sigma)) # Standard deviation vector
        mu_j = collocation.normal.mean(mu, sigma, QoI)
        err = abs(fval - mu_j)
        self.assertTrue(err < 1.e-14)

    def test_evalQoIGradient(self):
        systemsize = 4
        eigen_decayrate = 2.0
        QoI = examples.HadamardQuadratic(systemsize, eigen_decayrate)
        mu = np.random.rand(systemsize)
        xi = np.random.rand(systemsize)

        grad_val = QoI.eval_QoIGradient(mu, xi)

        # Check against finite difference
        def func(x):
            deviation = x - mu
            return QoI.eval_QoI(mu, deviation)

        grad_fd = nd.Gradient(func)(mu + xi)
        error_vals = abs(grad_val - grad_fd)
        self.assertTrue((error_vals < 1.e-10).all())


    def test_evalQoIHessian(self):
        systemsize = 4
        eigen_decayrate = 2.0
        QoI = examples.HadamardQuadratic(systemsize, eigen_decayrate)
        mu = np.zeros(systemsize)
        xi = np.zeros(systemsize)

        Hessian = QoI.eval_QoIHessian(mu, xi)

        # Check against finite difference
        def func(x):
            deviation = x - mu
            return QoI.eval_QoI(mu, deviation)

        Hess_fd = nd.Hessian(func)(mu + xi)
        
        error_vals = abs(Hessian - Hess_fd)
        self.assertTrue((error_vals < 1.e-10).all())

if __name__ == "__main__":
    unittest.main()
