import sys
sys.path.insert(0, '/Users/kinshuk/Documents/ODL/pyStatReduce/src')

import unittest
import numpy as np
import chaospy as cp

from StochasticCollocation import StochasticCollocation
from QuantityOfInterest import QuantityOfInterest
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
        mu_j = collocation.normal(mu, sigma, QoI, collocation)
        err = abs(fval - mu_j)
        self.assertTrue(err < 1.e-14)

    def test_evalQoIHessian(self):
        systemsize = 4
        eigen_decayrate = 2.0
        QoI = examples.HadamardQuadratic(systemsize, eigen_decayrate)
        mu = np.zeros(systemsize)
        xi = np.zeros(systemsize)

        Hessian = QoI.eval_QoIHessian(mu, xi)
        true_value = np.array([[ 0.35590278,  0.19965278,  0.26909722,  0.17534722],
                               [ 0.19965278,  0.35590278,  0.17534722,  0.26909722],
                               [ 0.26909722,  0.17534722,  0.35590278,  0.19965278],
                               [ 0.17534722,  0.26909722,  0.19965278,  0.35590278]])

        error_vals = abs(Hessian - true_value)
        self.assertTrue((error_vals < 1.e-7).all())


if __name__ == "__main__":
    unittest.main()