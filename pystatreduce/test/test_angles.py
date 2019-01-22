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

np.set_printoptions(precision=16)

class TestDominantSubspaceDistance(unittest.TestCase):

    def test_subspace_angles(self):
        S1 = np.array([[1,2],[3,4],[5,6]])
        S2 = np.array([[1,5],[3,7],[5,-1]])
        val = utils.compute_subspace_angles(S1, S2)
        true_val = np.array([0., 0.5431221704536429])
        err = abs(val - true_val)
        self.assertTrue((err < 1.e-14).all())

    def test_angles_2QoI(self):
        systemsize = 2
        theta = np.pi/3
        mu = np.random.randn(systemsize)
        std_dev = np.eye(systemsize) # np.diag(np.random.rand(systemsize))
        jdist = cp.MvNormal(mu, std_dev)
        QoI1 = examples.Paraboloid2D(systemsize, (theta,))
        QoI2 = examples.PolyRVDV()
        QoI_dict = {'paraboloid2' : {'quadrature_degree' : 3,
                                     'reduced_collocation' : False,
                                     'QoI_func' : QoI1.eval_QoI,
                                     'output_dimensions' : 1,
                                     'include_derivs' : False,
                                    },
                    'PolyRVDV' : {'quadrature_degree' : 3,
                                  'reduced_collocation' : False,
                                  'QoI_func' : QoI2.eval_QoI,
                                  'output_dimensions' : 1,
                                  'include_derivs' : False,
                                  }
                    }
        # Create the dominant space for the 2 dominant directions
        threshold_factor = 0.8
        QoI1_dominant_space = DimensionReduction(threshold_factor=threshold_factor, exact_Hessian=True)
        QoI2_dominant_space = DimensionReduction(threshold_factor=threshold_factor, exact_Hessian=True)
        QoI1_dominant_space.getDominantDirections(QoI1, jdist)
        QoI2_dominant_space.getDominantDirections(QoI2, jdist)

        # Print the dominant dominant_indices
        # print("QoI1 dominant_indices = ", QoI1_dominant_space.iso_eigenvals)
        # print("QoI1 eigenvecs = \n", QoI1_dominant_space.iso_eigenvecs, '\n')
        # print("QoI2 eigenvecs = \n", QoI2_dominant_space.iso_eigenvecs)
        S1 = QoI1_dominant_space.iso_eigenvecs[:, QoI1_dominant_space.dominant_indices]
        S2 = QoI2_dominant_space.iso_eigenvecs[:, QoI2_dominant_space.dominant_indices]
        # print('S1 = \n', S1)
        # print('S2 = \n', S2)
        angles = utils.compute_subspace_angles(S1, S2)
        self.assertTrue(abs(angles[0] - theta) < 1.e-14)

if __name__ == "__main__":
    unittest.main()
