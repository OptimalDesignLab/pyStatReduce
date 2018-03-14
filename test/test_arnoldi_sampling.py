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
import examples

class ArnoldiSamplingTest(unittest.TestCase):
    """
    def test_modified_GramSchmidt_fullRank(self):

        # Generate a random set of vectors and use modGramSchmidt to
        # orthogonalize them

        systemsize = 10

        # Initialize ArnoldSampling object
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

        # Initialize ArnoldSampling object
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
    """
    def test_arnoldiSample_positiveDefinite(self):
        """
        Use a synthetic quadratic function, and check that arnoldiSample recovers
        its eigenvalues and eigenvectors
        """
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

        # Initialize ArnoldSampling object
        alpha = 1.0
        num_sample = systemsize+1
        arnoldi = ArnoldiSampling(alpha, num_sample)

        # Generate sample
        eigenvals = np.zeros(systemsize)
        eigenvecs = np.zeros([systemsize, systemsize])
        grad_red = np.zeros(systemsize)
        arnoldi.arnoldiSample(QoI, xdata, fdata, gdata, eigenvals, eigenvecs,
                              grad_red)

        # Check that eigenvalues and eigenvectors agree
        for i in xrange(0, systemsize):
            print "i = ", i
            self.assertAlmostEqual(eigenvals[i], QoI.E[i], places=7)
            self.assertAlmostEqual(abs(np.dot(eigenvecs[:,i], QoI.V[:,i])), 1.0, places=7)

if __name__ == "__main__":
    unittest.main()
