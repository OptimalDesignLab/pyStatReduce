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
        print(fval)





if __name__ == "__main__":
    unittest.main()
