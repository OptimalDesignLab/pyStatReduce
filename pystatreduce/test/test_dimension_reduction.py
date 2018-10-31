# Test dimension reduction
import unittest
import numpy as np
import chaospy as cp

from pystatreduce.stochastic_collocation import StochasticCollocation
from pystatreduce.quantity_of_interest import QuantityOfInterest
from pystatreduce.dimension_reduction import DimensionReduction
from pystatreduce.stochastic_arnoldi.arnoldi_sample import ArnoldiSampling
import pystatreduce.examples as examples

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
        dominant_space = DimensionReduction(threshold_factor=threshold_factor,
                                            exact_Hessian=True)

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
        self.assertEqual(dominant_space.dominant_indices, [0,1])

    def test_reducedCollocation(self):

        systemsize = 4
        eigen_decayrate = 2.0

        # Create Hadmard Quadratic object
        QoI = examples.HadamardQuadratic(systemsize, eigen_decayrate)

        # Create stochastic collocation object
        collocation = StochasticCollocation(3, "Normal")

        # Create dimension reduction object
        threshold_factor = 0.9
        dominant_space = DimensionReduction(threshold_factor=threshold_factor,
                                            exact_Hessian=True)

        # Initialize chaospy distribution
        std_dev = 0.2*np.ones(QoI.systemsize)
        x = np.ones(QoI.systemsize)
        jdist = cp.MvNormal(x, np.diag(std_dev))

        # Get the eigenmodes of the Hessian product and the dominant indices
        dominant_space.getDominantDirections(QoI, jdist)

        QoI_func = QoI.eval_QoI
        # Check the mean
        mu_j = collocation.normal.reduced_mean(QoI_func, jdist, dominant_space)
        true_value_mu_j = 4.05
        err = abs(mu_j - true_value_mu_j)
        self.assertTrue(err < 1.e-13)

        # Check the variance
        red_var_j = collocation.normal.reduced_variance(QoI_func, jdist, dominant_space, mu_j)
        # var_j = collocation.normal.variance(QoI_func, jdist, mu_j)
        analytical_var_j = QoI.eval_analytical_QoI_variance(x, cp.Cov(jdist))
        err = abs(analytical_var_j - red_var_j)
        self.assertTrue(err < 1.e-4)

    def test_dimensionReduction_arnoldi_enlarge(self):
        systemsize = 128
        eigen_decayrate = 1.0

        # Create Hadmard Quadratic object
        QoI = examples.HadamardQuadratic(systemsize, eigen_decayrate)

        # Create stochastic collocation object
        collocation = StochasticCollocation(3, "Normal")

        # Create dimension reduction object
        threshold_factor = 0.9
        dominant_space_exactHess = DimensionReduction(threshold_factor=threshold_factor,
                                                      exact_Hessian=True)
        dominant_space_arnoldi = DimensionReduction(exact_Hessian=False, n_arnoldi_sample=71)

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

        energy_err = np.linalg.norm(lambda_arnoldi[0:10] - lambda_exact[0:10]) / np.linalg.norm(lambda_exact[0:10])

        self.assertTrue(energy_err < 1.e-8)



if __name__ == "__main__":
    unittest.main()
