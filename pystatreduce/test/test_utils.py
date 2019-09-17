# Test stochastic collocation module
import unittest
import numpy as np
import cmath
import chaospy as cp

from pystatreduce.new_stochastic_collocation import StochasticCollocation2
from pystatreduce.stochastic_collocation3 import StochasticCollocation3
from pystatreduce.quantity_of_interest import QuantityOfInterest
from pystatreduce.dimension_reduction import DimensionReduction
import pystatreduce.examples as examples
import pystatreduce.utils as utils

class TestUtils(unittest.TestCase):
    """
    This fie contains unittests of all the generic utitlity functions in
    pystatreduce.utils, i.e., utils.py
    """
    def test_subspace_angles(self):
        S1 = np.array([[1,2],[3,4],[5,6]])
        S2 = np.array([[1,5],[3,7],[5,-1]])
        val = utils.compute_subspace_angles(S1, S2)
        true_val = np.array([0., 0.5431221704536429])
        err = abs(val - true_val)
        self.assertTrue((err < 1.e-14).all())

        S3 = np.array([1,3,5])
        S4 = np.array([2,4,6])
        val2 = utils.compute_subspace_angles(S3, S1)
        true_val2 = np.array([0.11088375048225246])



    def test_central_difference(self):
        systemsize=3
        QoI = examples.Paraboloid3D(systemsize)

        def func(x):
            return QoI.eval_QoI(x, np.zeros(systemsize))

        x_location = np.random.randn(3)
        # Do the finite difference
        dfval = utils.central_fd(func, x_location, output_dimensions=1, fd_pert=1.e-6)

        # Get the analytical gradient
        dfval_analytical = QoI.eval_QoIGradient(x_location, np.zeros(3))

        err = abs(dfval - dfval_analytical)
        self.assertTrue((err < 1.e-6).all())



if __name__ == '__main__':
    unittest.main()
