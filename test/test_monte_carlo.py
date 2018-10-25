# Test the monte carlo method
import sys
import os

# Get the directory of this file
TEST_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = TEST_DIR + '/../src'
sys.path.insert(0, SRC_DIR)

import unittest
import numpy as np
import chaospy as cp

from monte_carlo import MonteCarlo
from stochastic_collocation import StochasticCollocation
from quantity_of_interest import QuantityOfInterest
from dimension_reduction import DimensionReduction

import examples

class MonteCarloTest(unittest.TestCase):

    def test_scalarQoI(self):
        systemsize = 3
        mu = np.random.rand(systemsize)
        std_dev = np.diag(np.random.rand(systemsize))
        jdist = cp.MvNormal(mu, std_dev)
        # Create QoI Object
        QoI = examples.Paraboloid3D(systemsize)

        # Create the Monte Carlo object
        QoI_dict = {'paraboloid' : {'QoI_func' : QoI.eval_QoI,
                                    'output_dimensions' : 1,}
                    }
        nsample = 1000000
        mc_obj = MonteCarlo(nsample, QoI_dict)
        mc_obj.getSamples(jdist)
        # Get the mean and variance using Monte Carlo
        mu_js = mc_obj.mean(jdist, of=['paraboloid'])
        var_js = mc_obj.variance(jdist, of=['paraboloid'])

        # Analytical mean
        mu_j_analytical = QoI.eval_QoI_analyticalmean(mu, cp.Cov(jdist))
        err = abs((mu_js['paraboloid'] - mu_j_analytical)/ mu_j_analytical)
        self.assertTrue(err < 1e-3)
        # Analytical variance
        var_j_analytical = QoI.eval_QoI_analyticalvariance(mu, cp.Cov(jdist))
        err = abs((var_js['paraboloid'] - var_j_analytical) / var_j_analytical)
        self.assertTrue(err < 1e-2)

if __name__ == "__main__":
    unittest.main()
