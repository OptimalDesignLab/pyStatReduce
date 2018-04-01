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


    def test_dimensionReduction_arnoldi(self):

        systemsize = 16
        eigen_decayrate = 2.0

        # Create Hadmard Quadratic object
        QoI = examples.HadamardQuadratic(systemsize, eigen_decayrate)

        # Create stochastic collocation object
        collocation = StochasticCollocation(3, "Normal")

        # Create dimension reduction object
        threshold_factor = 0.9
        dominant_space_exactHess = DimensionReduction(threshold_factor, exact_Hessian=True)
        dominant_space_arnoldi = DimensionReduction(threshold_factor, exact_Hessian=False)

        # Initialize chaospy distribution
        std_dev = np.random.rand(QoI.systemsize)
        x = np.random.rand(QoI.systemsize)
        jdist = cp.MvNormal(x, np.diag(std_dev))

        # Get the eigenmodes of the Hessian product and the dominant indices
        dominant_space_exactHess.getDominantDirections(QoI, jdist)
        dominant_space_arnoldi.getDominantDirections(QoI, jdist)

        # Print iso_eigenvals
        sort_ind1 = dominant_space_exactHess.iso_eigenvals.argsort()[::-1]
        sort_ind2 = dominant_space_arnoldi.iso_eigenvals.argsort()[::-1]
        # print '\n', "dominant_space_exactHess.iso_eigenvals = ", \
        #     dominant_space_exactHess.iso_eigenvals[sort_ind1]
        # print "dominant_space_arnoldi.iso_eigenvals = ", \
        #     dominant_space_arnoldi.iso_eigenvals[sort_ind2]

        # Print iso_eigenvecs
        # print '\n', "dominant_space_exactHess.iso_eigenvecs = ", '\n', \
        #     dominant_space_exactHess.iso_eigenvecs[:, sort_ind1]
        # print "dominant_space_arnoldi.iso_eigenvecs = ", '\n', \
        #     dominant_space_arnoldi.iso_eigenvecs[:, sort_ind2]

        # Compare the eigenvalues
        err_eigenvals = abs(dominant_space_exactHess.iso_eigenvals[sort_ind1] -
                            dominant_space_arnoldi.iso_eigenvals[sort_ind2])
        self.assertTrue((err_eigenvals < 1.e-7).all())

        # Compare the eigenvectors
        V_exact = dominant_space_exactHess.iso_eigenvecs[:, sort_ind1]
        V_arnoldi = dominant_space_arnoldi.iso_eigenvecs[:, sort_ind2]
        product = np.zeros(QoI.systemsize)
        for i in xrange(0, systemsize):
            product[i] = abs(np.dot(V_exact[:,i], V_arnoldi[:,i]))
            self.assertAlmostEqual(product[i], 1.0, places=7)


    def test_dimensionReduction_arnoldi_enlarge(self):
        systemsize = 32
        eigen_decayrate = 2.0
        n_arnoldi_sample = 21

        # Create Hadmard Quadratic object
        QoI = examples.HadamardQuadratic(systemsize, eigen_decayrate)

        # Create stochastic collocation object
        collocation = StochasticCollocation(3, "Normal")

        # Create dimension reduction object
        threshold_factor = 0.9
        dominant_space_exactHess = DimensionReduction(threshold_factor, exact_Hessian=True)
        dominant_space_arnoldi = DimensionReduction(threshold_factor, exact_Hessian=False)

        # Initialize chaospy distribution
        std_dev = np.random.rand(QoI.systemsize)
        x = np.random.rand(QoI.systemsize)
        jdist = cp.MvNormal(x, np.diag(std_dev))

        # Get the eigenmodes of the Hessian product and the dominant indices
        dominant_space_exactHess.getDominantDirections(QoI, jdist)
        dominant_space_arnoldi.getDominantDirections(QoI, jdist)

        # Print iso_eigenvals
        sort_ind1 = dominant_space_exactHess.iso_eigenvals.argsort()[::-1]
        sort_ind2 = dominant_space_arnoldi.iso_eigenvals.argsort()[::-1]
        lambda_exact = dominant_space_exactHess.iso_eigenvals[sort_ind1]
        lambda_arnoldi = dominant_space_arnoldi.iso_eigenvals[sort_ind2]

        # Compare the eigenvalues
        for i in xrange(0, n_arnoldi_sample-1):
            if i < 10:
                self.assertAlmostEqual(lambda_arnoldi[i], lambda_exact[i], places=6)
            else:
                print "lambda_exact[i] = ", lambda_exact[i], "lambda_arnoldi[i] = ", lambda_arnoldi[i]
                self.assertAlmostEqual(lambda_arnoldi[i], lambda_exact[i], places=1)

        # Compare the eigenvectors
        V_exact = dominant_space_exactHess.iso_eigenvecs[:, sort_ind1]
        V_arnoldi = dominant_space_arnoldi.iso_eigenvecs[:, sort_ind2]
        for i in xrange(0, n_arnoldi_sample-1):
            print "i = ", i
            product = abs(np.dot(V_exact[:,i], V_arnoldi[:,i]))
            if i < 10:
                self.assertAlmostEqual(product, 1.0, places=6)
            # else:
            #     self.assertAlmostEqual(product, 1.0, places=0)


if __name__ == "__main__":
    unittest.main()
