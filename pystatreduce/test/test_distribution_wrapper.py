import numpy as np
import chaospy as cp
import openturns as ot
import unittest

# PyStatReduce Imports
from pystatreduce.distribution_wrapper import Distribution

np.set_printoptions(linewidth=150)

class DistributionWrapperTest(unittest.TestCase):

    def setUp(self):
        # Univariate
        self.mu_scalar = 2.2
        self.std_dev_scalar = 0.6
        # Bivariate
        self.mu_vec2 = np.array([1.0,1.5])
        self.std_dev_vec2 = np.array([2.0, 4.0])

    def test_chaospy_univariate_normal(self):
        dist = Distribution(package="chaospy", distribution_type="Normal",
                            mean_val=self.mu_scalar,
                            standard_deviation=self.std_dev_scalar)
        mu = dist.E()
        variance = dist.Cov()
        std_dev = dist.Std()

        self.assertAlmostEqual(mu, self.mu_scalar, places=14)
        self.assertAlmostEqual(self.std_dev_scalar, std_dev, places=14)
        self.assertAlmostEqual(variance, self.std_dev_scalar**2)

    def test_openturns_univariate_normal(self):
        dist = Distribution(package="openturns", distribution_type="Normal",
                            mean_val=self.mu_scalar,
                            standard_deviation=self.std_dev_scalar)
        mu = dist.E()
        variance = dist.Cov()
        std_dev = dist.Std()

        self.assertAlmostEqual(mu, self.mu_scalar, places=14)
        self.assertAlmostEqual(self.std_dev_scalar, std_dev, places=14)
        self.assertAlmostEqual(variance, self.std_dev_scalar**2)

    def test_chaospy_bivariate_normal(self):
        dist = Distribution(package="chaospy", distribution_type="Normal",
                            mean_val=self.mu_vec2,
                            standard_deviation=self.std_dev_vec2)
        mu = dist.E()
        covariance = dist.Cov()
        std_dev = dist.Std()
        expected_covariance_mat = np.diag(self.std_dev_vec2**2)

        np.testing.assert_array_almost_equal(mu, self.mu_vec2, decimal=14)
        np.testing.assert_array_almost_equal(std_dev, self.std_dev_vec2, decimal=14)
        np.testing.assert_array_almost_equal(covariance, expected_covariance_mat, decimal=14)

    def test_openturns_bivariate_normal(self):
        dist = Distribution(package="openturns", distribution_type="Normal",
                            mean_val=self.mu_vec2,
                            standard_deviation=self.std_dev_vec2,
                            correleation_matrix=ot.CorrelationMatrix(2))
        mu = dist.E()
        covariance = dist.Cov()
        std_dev = dist.Std()
        expected_covariance_mat = np.diag(self.std_dev_vec2**2)

        np.testing.assert_array_almost_equal(mu, self.mu_vec2, decimal=14)
        np.testing.assert_array_almost_equal(std_dev, self.std_dev_vec2, decimal=14)
        np.testing.assert_array_almost_equal(covariance, expected_covariance_mat, decimal=14)

if __name__ == "__main__":
    unittest.main()
