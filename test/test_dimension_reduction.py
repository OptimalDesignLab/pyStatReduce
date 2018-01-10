# Test dimension reduction
import sys
sys.path.insert(0, '/Users/kinshuk/Documents/ODL/pyStatReduce/src')

import unittest
import numpy as np
import chaospy as cp

from StochasticCollocation import StochasticCollocation
from QuantityOfInterest import QuantityOfInterest
from dimension_reduction import DimensionReduction
import examples

class DimensionReductionTest(unittest.TestCase):

    def test_dimensionReduction(self):

        systemsize = 4
        eigen_decayrate = 2.0

        # Create Hadmard Quadratic object
        QoI = examples.HadamardQuadratic(systemsize, eigen_decayrate)

        # Create stochastic collocation object
        collocation = StochasticCollocation(3, "Normal")

        # Create dimension reduction object
        threshold_factor = 0.9
        dominant_space = DimensionReduction(threshold_factor)

        # Initialize chaospy distribution
        std_dev = 0.2*np.ones(QoI.systemsize)
        x = np.ones(QoI.systemsize)
        jdist = cp.MvNormal(x, np.diag(std_dev))

        # Get the eigenmodes of the Hessian product and the dominant indices
        eigen_vals, eigen_vectors, indices = \
                                dominant_space.getDominantDirections(QoI, jdist)
        true_eigen_vals = np.array([ 0.04, 0.01, 0.0025, 0.004444444444444])
        err_eigen_vals = abs(eigen_vals - true_eigen_vals)

        self.assertTrue((err_eigen_vals < 1.e-15).all())
        self.assertItemsEqual(indices, [0,1])


if __name__ == "__main__":
    unittest.main()
