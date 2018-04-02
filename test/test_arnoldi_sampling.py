# test_arnoldi_sampling.py
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
from stochastic_arnoldi.arnoldi_sample import ArnoldiSampling
from dimension_reduction import DimensionReduction
import examples

class ArnoldiSamplingTest(unittest.TestCase):

    def test_modified_GramSchmidt_fullRank(self):

        # Generate a random set of vectors and use modGramSchmidt to
        # orthogonalize them

        systemsize = 10

        # Initialize ArnoldiSampling object
        alpha = 1.0
        num_sample = 4
        arnoldi = ArnoldiSampling(alpha, num_sample)

        # Create arrays for modified_GramSchmidt
        Z = np.random.rand(systemsize, num_sample)
        H = np.zeros([num_sample, num_sample-1])

        # Populate Z
        for i in xrange(-1, num_sample-1):
            arnoldi.modified_GramSchmidt(i, H, Z)
            # Check that the vectors are unit normal
            self.assertAlmostEqual(np.linalg.norm(Z[:,i+1]), 1, places=14)

        # Check that the vectors are orthogonal
        for i in xrange(0, num_sample):
            for j in xrange(i+1, num_sample):
                self.assertAlmostEqual(np.dot(Z[:,i], Z[:,j]), 0, places=14)

    def test_modified_GramSchmidt_RankDeficient(self):

        # Generate a random set of vectors, make one of them a linear combination
        # of the others, and use modGramSchmidt to orthogonalize them

        systemsize = 10

        # Initialize ArnoldiSampling object
        alpha = 1.0
        num_sample = 4
        arnoldi = ArnoldiSampling(alpha, num_sample)

        # Create arrays for modified_GramSchmidt
        Z = np.random.rand(systemsize, num_sample)
        Z[:,num_sample-1] = Z[:,0:num_sample-1].dot(np.random.rand(num_sample-1))
        H = np.zeros([num_sample, num_sample-1])

        for i in xrange(-1, num_sample-2):
            arnoldi.modified_GramSchmidt(i, H, Z)
            # Check that the vectors are unit normal
            self.assertAlmostEqual(np.linalg.norm(Z[:,i+1]), 1, places=14)

        # calling now should produce lin_depend flag
        lin_depend = arnoldi.modified_GramSchmidt(num_sample-2, H, Z)
        self.assertTrue(lin_depend)

    def test_arnoldiSample_og_positiveDefinite(self):

        # Use a synthetic quadratic function, and check that arnoldiSample recovers
        # its eigenvalues and eigenvectors

        systemsize = 10

        # Generate QuantityOfInterest
        QoI = examples.RandomQuadratic(systemsize, positive_definite=True)

        # generate data at initial point (1,1,1,...,1)^T
        xdata = np.zeros([systemsize, systemsize+1])
        fdata = np.zeros(systemsize+1)
        gdata = np.zeros([systemsize, systemsize+1])
        xdata[:,0] = np.ones(systemsize)
        fdata[0] = QoI.eval_QoI(xdata[:,0], np.zeros(systemsize))
        gdata[:,0] = QoI.eval_QoIGradient(xdata[:,0], np.zeros(systemsize))

        # Initialize ArnoldiSampling object
        alpha = 1.0
        num_sample = systemsize+1
        arnoldi = ArnoldiSampling(alpha, num_sample)

        # Generate sample
        eigenvals = np.zeros(systemsize)
        eigenvecs = np.zeros([systemsize, systemsize])
        grad_red = np.zeros(systemsize)
        arnoldi.arnoldiSample_og(QoI, xdata, fdata, gdata, eigenvals, eigenvecs,
                              grad_red)

        # Check that eigenvalues and eigenvectors agree
        for i in xrange(0, systemsize):
            self.assertAlmostEqual(eigenvals[i], QoI.E[i], places=7)
            self.assertAlmostEqual(abs(np.dot(eigenvecs[:,i], QoI.V[:,i])), 1.0, places=7)

    def test_arnoldiSample_og_positiveSemiDefinite(self):

        # Use a synthetic quadratic function, and check that arnoldiSample recovers
        # its eigenvalues and eigenvectors

        systemsize = 10

        # Generate QuantityOfInterest
        QoI = examples.RandomQuadratic(systemsize, positive_definite=False)

        # generate data at initial point (1,1,1,...,1)^T
        xdata = np.zeros([systemsize, systemsize+1])
        fdata = np.zeros(systemsize+1)
        gdata = np.zeros([systemsize, systemsize+1])
        xdata[:,0] = np.ones(systemsize)
        fdata[0] = QoI.eval_QoI(xdata[:,0], np.zeros(systemsize))
        gdata[:,0] = QoI.eval_QoIGradient(xdata[:,0], np.zeros(systemsize))

        # Initialize ArnoldiSampling object
        alpha = 1.0
        num_sample = systemsize+1
        arnoldi = ArnoldiSampling(alpha, num_sample)

        # Generate sample
        eigenvals = np.zeros(systemsize)
        eigenvecs = np.zeros([systemsize, systemsize])
        grad_red = np.zeros(systemsize)
        dim, error_estimate = arnoldi.arnoldiSample_og(QoI, xdata, fdata, gdata,
                                                    eigenvals, eigenvecs,
                                                    grad_red)

        self.assertAlmostEqual(error_estimate, 0.0, places=6)
        self.assertEqual(dim, systemsize-1)

        # Check that eigenvalues and eigenvectors agree
        for i in xrange(0, dim):
            self.assertAlmostEqual(eigenvals[i], QoI.E[i+1], places=7)
            self.assertAlmostEqual(abs(np.dot(eigenvecs[:,i], QoI.V[:,i+1])), 1.0, places=7)

    def test_arnoldiSample_og_reducedGradient(self):

        # use a nonlinear function and a small perturbation to test the reduced
        # gradient produced by arnoldiSample

        systemsize = 10

        # Generate QuantityOfInterest
        QoI = examples.ExponentialFunction(systemsize)

        # generate data at initial point (1,1,1,...,1)^T
        xdata = np.zeros([systemsize, systemsize+1])
        fdata = np.zeros(systemsize+1)
        gdata = np.zeros([systemsize, systemsize+1])
        xdata[:,0] = np.ones(systemsize)
        fdata[0] = QoI.eval_QoI(xdata[:,0], np.zeros(systemsize))
        gdata[:,0] = QoI.eval_QoIGradient(xdata[:,0], np.zeros(systemsize))

        # Initialize ArnoldiSampling object
        alpha = np.sqrt(np.finfo(float).eps)
        num_sample = systemsize+1
        arnoldi = ArnoldiSampling(alpha, num_sample)

        # Generate sample
        eigenvals = np.zeros(systemsize)
        eigenvecs = np.zeros([systemsize, systemsize])
        grad_red = np.zeros(systemsize)
        dim, error_estimate = arnoldi.arnoldiSample_og(QoI, xdata, fdata, gdata,
                                                    eigenvals, eigenvecs,
                                                    grad_red)

        # check reduced gradient;
        # rotate back out of eigenvector coordinates
        g = eigenvecs.dot(grad_red)
        err_g = abs(g - gdata[:,0])
        self.assertTrue((err_g < 1.e-6).all())

    def test_arnoldiSample_complete(self):

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


    def test_dimensionReduction_arnoldi_partial(self):
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
