import unittest
import numpy as np
import cmath
import chaospy as cp

from pystatreduce.stochastic_collocation import StochasticCollocation
from pystatreduce.quantity_of_interest import QuantityOfInterest
from pystatreduce.dimension_reduction import DimensionReduction
import pystatreduce.examples as examples

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
        rv_dict = {'v' : mean_v,
                   'alpha': mean_alpha,
                   'Mach_number' : mean_Ma,
                   're' : mean_re,
                   'rho' : mean_rho,
                  }

        QoI = examples.OASAerodynamicWrapper(uq_systemsize, rv_dict)
        jdist = cp.MvNormal(mu_init, std_dev)
        fval = QoI.eval_QoI(mu_init, np.zeros(uq_systemsize))

        # Check the values
        expected_CD = 0.03721668965447261
        expected_CL = 0.5123231521985626
        expected_CM = np.array([0., -0.1793346481832254, 0.])
        self.assertAlmostEqual(QoI.p['oas_example1.aero_point_0.CD'], expected_CD, places=13)
        self.assertAlmostEqual(QoI.p['oas_example1.aero_point_0.CL'], expected_CL, places=13)
        # np.testing.assert_array_almost_equal(QoI.p['oas_example1.aero_point_0.CM'][1], expected_CM, decimal=12)

        # Check the gradients
        grad = QoI.eval_QoIGradient(mu_init, np.zeros(uq_systemsize))
        expected_dCD = np.array([ 0., 0.00245012, 0.00086087, -0., 0.])
        expected_dCL = np.array([-0., 0.07634319, -0., -0., -0.])
        np.testing.assert_array_almost_equal(QoI.deriv['oas_example1.aero_point_0.CD', 'mu'][0], expected_dCD, decimal=7)
        np.testing.assert_array_almost_equal(QoI.deriv['oas_example1.aero_point_0.CL', 'mu'][0], expected_dCL, decimal=7)

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
        rv_dict = {'v' : mean_v,
                   'alpha': mean_alpha,
                   'Mach_number' : mean_Ma,
                   're' : mean_re,
                   'rho' : mean_rho,
                  }

        QoI = examples.OASAerodynamicWrapper(uq_systemsize, rv_dict)
        jdist = cp.MvNormal(mu_init, std_dev)
        dominant_space = DimensionReduction(n_arnoldi_sample=uq_systemsize+1,
                                            exact_Hessian=False)
        dominant_space.getDominantDirections(QoI, jdist, max_eigenmodes=3)

        expected_eigenvals = np.array([ 0.00001345, -0.00000043, 0., 0., -0.])
        expected_eigenvecs = np.array([[-0.00000003, 0.00000002],
                                       [-1., 0.00000001],
                                       [-0.00000001, -0.99999992],
                                       [ 0., -0.00039715],
                                       [ 0.00000004, 0.]])

        # We will only test the first 2 eigenvectors since the remaining 4 eigenvalues
        # are 0.
        np.testing.assert_array_almost_equal(dominant_space.iso_eigenvals, expected_eigenvals, decimal=6)
        np.testing.assert_array_almost_equal(dominant_space.iso_eigenvecs[:,0:2], expected_eigenvecs, decimal=6)



if __name__ == "__main__":
    unittest.main()
