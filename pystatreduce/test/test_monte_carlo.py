# Test the monte carlo method
import sys
import os

# # Get the directory of this file
# TEST_DIR = os.path.dirname(os.path.abspath(__file__))
# SRC_DIR = TEST_DIR + '/../src'
# sys.path.insert(0, SRC_DIR)

import unittest
import numpy as np
import chaospy as cp

from pystatreduce.monte_carlo import MonteCarlo
from pystatreduce.stochastic_collocation import StochasticCollocation
from pystatreduce.new_stochastic_collocation import StochasticCollocation2
from pystatreduce.quantity_of_interest import QuantityOfInterest
from pystatreduce.dimension_reduction import DimensionReduction

from pystatreduce.examples import Paraboloid3D, HadamardQuadratic

class MonteCarloTest(unittest.TestCase):

    def test_scalarQoI(self):
        systemsize = 3
        mu = np.random.rand(systemsize)
        std_dev = np.diag(np.random.rand(systemsize))
        jdist = cp.MvNormal(mu, std_dev)
        # Create QoI Object
        QoI = Paraboloid3D(systemsize)

        # Create the Monte Carlo object
        QoI_dict = {'paraboloid' : {'QoI_func' : QoI.eval_QoI,
                                    'output_dimensions' : 1,}
                    }
        nsample = 1000000
        mc_obj = MonteCarlo(nsample, jdist, QoI_dict)
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
        # print('var_js paraboloid = ', var_js['paraboloid'], '\n')
        self.assertTrue(err < 1e-2)

    def test_derivatives_scalarQoI(self):

        systemsize = 3
        mu = np.random.rand(systemsize)
        std_dev = np.diag(np.random.rand(systemsize))
        jdist = cp.MvNormal(mu, std_dev)
        # Create QoI Object
        QoI = Paraboloid3D(systemsize)

        # Create the Monte Carlo object
        deriv_dict = {'xi' : {'dQoI_func' : QoI.eval_QoIGradient,
                              'output_dimensions' : systemsize}
                     }
        QoI_dict = {'paraboloid' : {'QoI_func' : QoI.eval_QoI,
                                    'output_dimensions' : 1,
                                    'deriv_dict' : deriv_dict
                                    }
                    }

        nsample = 1000000
        mc_obj = MonteCarlo(nsample, jdist, QoI_dict, include_derivs=True)
        mc_obj.getSamples(jdist, include_derivs=True)
        dmu_j = mc_obj.dmean(jdist, of=['paraboloid'], wrt=['xi'])
        dvar_j = mc_obj.dvariance(jdist, of=['paraboloid'], wrt=['xi'])

        # Analytical dmu_j
        dmu_j_analytical = np.array([100*mu[0], 50*mu[1], 2*mu[2]])
        err = abs((dmu_j['paraboloid']['xi'] - dmu_j_analytical) / dmu_j_analytical)
        self.assertTrue((err < 0.01).all())

        # Analytical dvar_j
        rv_dev = cp.Std(jdist)
        dvar_j_analytical = np.array([(100*rv_dev[0])**2, (50*rv_dev[1])**2, (2*rv_dev[2])**2])
        err = abs((dvar_j['paraboloid']['xi'] - dvar_j_analytical) / dvar_j_analytical)
        # print("dvar_j = ", dvar_j['paraboloid']['xi'])
        # print("dvar_j_analytical = ", dvar_j_analytical)
        # print("err = ", err)

    def test_reduced_montecarlo(self):
        systemsize = 4
        eigen_decayrate = 2.0

        # Create Hadmard Quadratic object
        QoI = HadamardQuadratic(systemsize, eigen_decayrate)
        true_eigenvals = np.array([ 0.08, 0.02, 0.005, 0.00888888888888889])
        true_eigenvecs = np.array([[ 0.5,  0.5, -0.5, -0.5],
                                   [ 0.5, -0.5,  0.5, -0.5],
                                   [ 0.5,  0.5,  0.5,  0.5],
                                   [ 0.5, -0.5, -0.5,  0.5]])
        dominant_dir = true_eigenvecs[:,0:3]

        QoI_dict = {'hadamard4_2' : {'QoI_func' : QoI.eval_QoI,
                                     'output_dimensions' : 1,
                                    }
                    }

        # Create the distribution
        mu = np.ones(systemsize)
        std_dev = 0.2 * np.eye(systemsize)
        jdist = cp.MvNormal(mu, std_dev)
        # Create the Monte Carlo object
        nsample = 100000
        mc_obj = MonteCarlo(nsample, jdist, QoI_dict, reduced_collocation=True,
                            dominant_dir=dominant_dir, include_derivs=False)
        mc_obj.getSamples(jdist, include_derivs=False)
        mu_j_mc = mc_obj.mean(jdist, of=['hadamard4_2'])

        # Compare against a reduced stochastic collocation object
        sc_obj = StochasticCollocation2(jdist, 3, 'MvNormal', QoI_dict,
                                        reduced_collocation=True,
                                        dominant_dir=dominant_dir)
        sc_obj.evaluateQoIs(jdist)
        mu_j_sc = sc_obj.mean(of=['hadamard4_2'])

        rel_err = abs( (mu_j_mc['hadamard4_2'] - mu_j_sc['hadamard4_2']) / mu_j_sc['hadamard4_2'])
        self.assertTrue(rel_err[0] < 1.e-3)

if __name__ == "__main__":
    unittest.main()
