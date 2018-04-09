# Test dimension reduction
import sys
import os

# Get the directory of this file
TEST_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = TEST_DIR + '/../src'
sys.path.insert(0, SRC_DIR)

import unittest
import numpy as np
import chaospy as cp

from stochastic_collocation import StochasticCollocation
from quantity_of_interest import QuantityOfInterest
from dimension_reduction import DimensionReduction
from stochastic_arnoldi.arnoldi_sample import ArnoldiSampling
import examples

np.set_printoptions(linewidth=150)


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
        dominant_space = DimensionReduction(threshold_factor, exact_Hessian=True)

        # Initialize chaospy distribution
        std_dev = 0.2*np.ones(QoI.systemsize)
        x = np.ones(QoI.systemsize)
        jdist = cp.MvNormal(x, np.diag(std_dev))

        # Get the eigenmodes of the Hessian product and the dominant indices
        dominant_space.getDominantDirections(QoI, jdist)
        true_eigenvals = np.array([ 0.08, 0.02, 0.005, 0.00888888888888889])
        err_eigenvals = abs(dominant_space.iso_eigenvals - true_eigenvals)

        true_eigenvecs = np.array([[ 0.5,  0.5, -0.5, -0.5],
                                    [ 0.5, -0.5,  0.5, -0.5],
                                    [ 0.5,  0.5,  0.5,  0.5],
                                    [ 0.5, -0.5, -0.5,  0.5]])
        err_eigenvecs = abs(dominant_space.iso_eigenvecs - true_eigenvecs)

        self.assertTrue((err_eigenvals < 1.e-15).all())
        self.assertTrue((err_eigenvecs < 1.e-15).all())
        self.assertItemsEqual(dominant_space.dominant_indices, [0,1])

    def test_reducedCollocation(self):

        systemsize = 4
        eigen_decayrate = 2.0

        # Create Hadmard Quadratic object
        QoI = examples.HadamardQuadratic(systemsize, eigen_decayrate)

        # Create stochastic collocation object
        collocation = StochasticCollocation(3, "Normal")

        # Create dimension reduction object
        threshold_factor = 0.9
        dominant_space = DimensionReduction(threshold_factor, exact_Hessian=True)

        # Initialize chaospy distribution
        std_dev = 0.2*np.ones(QoI.systemsize)
        x = np.ones(QoI.systemsize)
        jdist = cp.MvNormal(x, np.diag(std_dev))

        # Get the eigenmodes of the Hessian product and the dominant indices
        dominant_space.getDominantDirections(QoI, jdist)

        mu_j = collocation.normal.reduced_mean(QoI, jdist, dominant_space)
        true_value_mu_j = 4.05
        err = abs(mu_j - true_value_mu_j)
        self.assertTrue(err < 1.e-15)

    def test_dimensionReduction_arnoldi_enlarge(self):
        systemsize = 16
        eigen_decayrate = 2.0
        # n_arnoldi_sample = 51

        # Create Hadmard Quadratic object
        QoI = examples.HadamardQuadratic(systemsize, eigen_decayrate)

        # Create stochastic collocation object
        collocation = StochasticCollocation(3, "Normal")

        # Create dimension reduction object
        threshold_factor = 0.9
        dominant_space_exactHess = DimensionReduction(threshold_factor, exact_Hessian=True)
        dominant_space_arnoldi = DimensionReduction(threshold_factor, exact_Hessian=False, n_arnoldi_sample=6)

        # Initialize chaospy distribution
        std_dev = np.random.rand(QoI.systemsize)
        x = np.random.rand(QoI.systemsize)
        jdist = cp.MvNormal(x, np.diag(std_dev))

        # Get the eigenmodes of the Hessian product and the dominant indices
        dominant_space_exactHess.getDominantDirections(QoI, jdist)
        discarded_energy = dominant_space_arnoldi.getDominantDirections(QoI, jdist)

        # Print iso_eigenvals
        sort_ind1 = dominant_space_exactHess.iso_eigenvals.argsort()[::-1]
        sort_ind2 = dominant_space_arnoldi.iso_eigenvals.argsort()[::-1]
        lambda_exact = dominant_space_exactHess.iso_eigenvals[sort_ind1]
        lambda_arnoldi = dominant_space_arnoldi.iso_eigenvals[sort_ind2]

        energy_exact = 1 - np.sum(lambda_exact[0:5])/np.sum(lambda_exact)

        print "energy_exact = ", energy_exact
        print "discarded_energy = ", discarded_energy

        # # Compare the eigenvalues
        # print "dominant_space_arnoldi.num_sample = ", dominant_space_arnoldi.num_sample
        # for i in xrange(0, dominant_space_arnoldi.num_sample-1):
        #     if i < 10:
        #         np.testing.assert_almost_equal(lambda_arnoldi[i], lambda_exact[i], 6, err_msg='at index (%d)' % i)
        #     else:
        #         np.testing.assert_almost_equal(lambda_arnoldi[i], lambda_exact[i], 1, err_msg='at index (%d)' % i)

        # # Compare the eigenvectors
        # V_exact = dominant_space_exactHess.iso_eigenvecs[:, sort_ind1]
        # V_arnoldi = dominant_space_arnoldi.iso_eigenvecs[:, sort_ind2]
        # for i in xrange(0, dominant_space_arnoldi.num_sample-1):
        #     # print "i = ", i
        #     product = abs(np.dot(V_exact[:,i], V_arnoldi[:,i]))
        #     if i < 10:
        #         self.assertAlmostEqual(product, 1.0, places=6)
        #     # else:
        #     #     self.assertAlmostEqual(product, 1.0, places=0)



if __name__ == "__main__":
    unittest.main()
