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

    def test_modified_GramSchmidt(self):
        """
        generate a random set of vectors and use modGramSchmidt to
        orthogonalize them
        """
        systemsize = 10

        # Initialize ArnoldSampling object
        alpha = 1.e-6
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

if __name__ == "__main__":
    unittest.main()
