# test_oas_example.py
import sys
import os

# Get the directory of this file
TEST_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = TEST_DIR + '/../src'
sys.path.insert(0, SRC_DIR)

import unittest
import numpy as np
import chaospy as cp

# pyStatReduce specific imports
import numpy as np
import chaospy as cp
from stochastic_collocation import StochasticCollocation
from quantity_of_interest import QuantityOfInterest
from dimension_reduction import DimensionReduction
from stochastic_arnoldi.arnoldi_sample import ArnoldiSampling
import examples

#pyoptsparse sepecific imports
from scipy import sparse
import argparse
from pyoptsparse import Optimization, OPT, SNOPT

# Import the OpenMDAo shenanigans
from openmdao.api import IndepVarComp, Problem, Group, NewtonSolver, \
    ScipyIterativeSolver, LinearBlockGS, NonlinearBlockGS, \
    DirectSolver, LinearBlockGS, PetscKSP, SqliteRecorder

from openaerostruct.geometry.utils import generate_mesh
from openaerostruct.geometry.geometry_group import Geometry
from openaerostruct.aerodynamics.aero_groups import AeroPoint

class OASExample1Test(unittest.TestCase):
    """
    Base class that checks the wrapper for the first openaerostruct example.
    """
    def test_evaluation(self):
        uq_systemsize = 5
        mean_v = 248.136 # Mean value of input random variable
        mean_alpha = 5   #
        mean_Ma = 0.84
        mean_re = 1.e6
        mean_rho = 0.38
        mean_cg = np.zeros((3))

        std_dev = np.eye(uq_systemsize)
        mu_init = np.array([mean_v, mean_alpha, mean_Ma, mean_re, mean_rho])

        QoI = examples.OASAerodynamicWrapper(uq_systemsize)
        jdist = cp.MvNormal(mu_init, std_dev)
        fval = QoI.eval_QoI(mu_init, np.zeros(uq_systemsize))

        # Check the values
        expected_CD = 0.03721668965447261
        expected_CL = 0.5123231521985626
        expected_CM = np.array([0., -0.1793346481832254, 0.])
        self.assertAlmostEqual(QoI.p['oas_example1.aero_point_0.CD'], expected_CD, places=13)
        self.assertAlmostEqual(QoI.p['oas_example1.aero_point_0.CL'], expected_CL, places=13)
        np.testing.assert_array_almost_equal(QoI.p['oas_example1.aero_point_0.CM'], expected_CM, decimal=12)

        # Check the gradients
        grad = QoI.eval_QoIGradient(mu_init, np.zeros(uq_systemsize))
        expected_dCD = np.array([ 0., 0.00245012, 0.00086087, -0., 0.])
        expected_dCL = np.array([-0., 0.07634319, -0., -0., -0.])
        np.testing.assert_array_almost_equal(QoI.deriv['oas_example1.aero_point_0.CD', 'mu'][0], expected_dCD, decimal=7)
        np.testing.assert_array_almost_equal(QoI.deriv['oas_example1.aero_point_0.CL', 'mu'][0], expected_dCL, decimal=7)

    def test_optimization(self):
        pass

    def test_dominant_directions(self):
        uq_systemsize = 5
        mean_v = 248.136 # Mean value of input random variable
        mean_alpha = 5   #
        mean_Ma = 0.84
        mean_re = 1.e6
        mean_rho = 0.38
        mean_cg = np.zeros((3))

        std_dev = np.diag([1.0, 0.2, 0.01, 1.e2, 0.01]) # np.eye(uq_systemsize)
        mu_init = np.array([mean_v, mean_alpha, mean_Ma, mean_re, mean_rho])

        QoI = examples.OASAerodynamicWrapper(uq_systemsize)
        jdist = cp.MvNormal(mu_init, std_dev)
        dominant_space = DimensionReduction(n_arnoldi_sample=uq_systemsize+1,
                                            exact_Hessian=False)
        dominant_space.getDominantDirections(QoI, jdist, max_eigenmodes=3)
        print('iso_eigenvals = ', dominant_space.iso_eigenvals)
        print('iso_eigenvecs = ', dominant_space.iso_eigenvecs)
        print('dominant_indices = ', dominant_space.dominant_indices)


if __name__ == "__main__":
    unittest.main()
