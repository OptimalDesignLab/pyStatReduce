# test_active_subspace.py
import os
from os import path
import unittest
import numpy as np
import chaospy as cp

from pystatreduce.new_stochastic_collocation import StochasticCollocation2
from pystatreduce.stochastic_collocation import StochasticCollocation
from pystatreduce.quantity_of_interest import QuantityOfInterest
from pystatreduce.dimension_reduction import DimensionReduction
from pystatreduce.active_subspace import ActiveSubspace
from pystatreduce.stochastic_arnoldi.arnoldi_sample import ArnoldiSampling
import pystatreduce.examples as examples
# import pystatreduce.optimize.OAS_ScanEagle.oas_scaneagle_opt as scaneagle_opt

class ActiveSubspaceTest(unittest.TestCase):

    def test_on_exponential(self):
        systemsize = 2
        QoI = examples.Exp_07xp03y(systemsize)

        jdist = cp.J(cp.Uniform(-1,1),
                     cp.Uniform(-1,1))

        active_subspace = ActiveSubspace(QoI, n_dominant_dimensions=1, n_monte_carlo_samples=10000)
        active_subspace.getDominantDirections(QoI, jdist)

        expected_W1 = QoI.a / np.linalg.norm(QoI.a)
        np.testing.assert_almost_equal(active_subspace.dominant_dir[:,0], expected_W1)


    def test_on_Hadamard_Quadratic(self):
        systemsize = 4
        eigen_decayrate = 2.0

        # Create Hadmard Quadratic object
        QoI = examples.HadamardQuadratic(systemsize, eigen_decayrate)

        # Create the joint distribution
        jdist = cp.J(cp.Uniform(-1,1),
                     cp.Uniform(-1,1),
                     cp.Uniform(-1,1),
                     cp.Uniform(-1,1))

        # Create the active subspace object
        active_subspace = ActiveSubspace(QoI, n_dominant_dimensions=2, n_monte_carlo_samples=100000)
        active_subspace.getDominantDirections(QoI, jdist)

        # Expected C
        Hessian = QoI.eval_QoIHessian(np.zeros(systemsize), np.zeros(systemsize))
        C = np.matmul(Hessian, Hessian) / 3
        err = C - active_subspace.C_tilde
        self.assertTrue((abs(err) < 1.e-2).all())

    def test_SVD_equivalence(self):
        systemsize = 4
        eigen_decayrate = 2.0

        # Create Hadmard Quadratic object
        QoI = examples.HadamardQuadratic(systemsize, eigen_decayrate)

        # Create the joint distribution
        jdist = cp.J(cp.Uniform(-1,1),
                     cp.Uniform(-1,1),
                     cp.Uniform(-1,1),
                     cp.Uniform(-1,1))

        active_subspace_eigen = ActiveSubspace(QoI, n_dominant_dimensions=1,
                                               n_monte_carlo_samples=10000,
                                               use_svd=False, read_rv_samples=False,
                                               write_rv_samples=True)
        active_subspace_eigen.getDominantDirections(QoI, jdist)

        active_subspace_svd = ActiveSubspace(QoI, n_dominant_dimensions=1,
                                             n_monte_carlo_samples=10000,
                                             use_svd=True, read_rv_samples=True,
                                             write_rv_samples=False)
        active_subspace_svd.getDominantDirections(QoI, jdist)

        # Check the iso_eigenvals
        np.testing.assert_almost_equal(active_subspace_eigen.iso_eigenvals, active_subspace_svd.iso_eigenvals)
        # check the iso_eigenvecs
        self.assertTrue(active_subspace_eigen.iso_eigenvecs.shape, active_subspace_svd.iso_eigenvecs.shape)
        for i in range(active_subspace_eigen.iso_eigenvecs.shape[1]):
            arr1 = active_subspace_eigen.iso_eigenvecs[:,i]
            arr2 = active_subspace_svd.iso_eigenvecs[:,i]
            if np.allclose(arr1, arr2) == False:
                np.testing.assert_almost_equal(arr1, -arr2)

    def test_gradient_read_active_subspace(self):

        if os.path.exists("rv_arr.txt"):
            os.remove("rv_arr.txt")

        systemsize = 4
        eigen_decayrate = 2.0

        # Create Hadmard Quadratic object
        QoI = examples.HadamardQuadratic(systemsize, eigen_decayrate)

        # Create the joint distribution
        jdist = cp.J(cp.Uniform(-1,1),
                     cp.Uniform(-1,1),
                     cp.Uniform(-1,1),
                     cp.Uniform(-1,1))

        n_active_subspace_samples = 10000
        active_subspace_reference = ActiveSubspace(QoI, n_dominant_dimensions=1,
                                               n_monte_carlo_samples=n_active_subspace_samples,
                                               use_svd=False, read_rv_samples=False,
                                               write_rv_samples=True)
        active_subspace_reference.getDominantDirections(QoI, jdist)

        rv_arr = np.loadtxt('rv_arr.txt')
        grad_arr = np.zeros([n_active_subspace_samples, systemsize])
        pert = np.zeros(systemsize)
        for i in range(0, n_active_subspace_samples):
            grad_arr[i,:] = QoI.eval_QoIGradient(rv_arr[:,i], pert)

        active_subspace_read = ActiveSubspace(QoI, n_dominant_dimensions=1,
                                               n_monte_carlo_samples=n_active_subspace_samples,
                                               read_gradient_samples=True,
                                               gradient_array=grad_arr)
        active_subspace_read.getDominantDirections(QoI, jdist)

        # Check the iso_eigenvals
        np.testing.assert_almost_equal(active_subspace_read.iso_eigenvals, active_subspace_reference.iso_eigenvals)
        # check the iso_eigenvecs
        self.assertTrue(active_subspace_read.iso_eigenvecs.shape, active_subspace_reference.iso_eigenvecs.shape)
        for i in range(active_subspace_read.iso_eigenvecs.shape[1]):
            arr1 = active_subspace_read.iso_eigenvecs[:,i]
            arr2 = active_subspace_reference.iso_eigenvecs[:,i]
            if np.allclose(arr1, arr2) == False:
                np.testing.assert_almost_equal(arr1, -arr2)

    def test_multifidelity_active_subspace(self):
        systemsize = 4
        eigen_decayrate = 2.0

        # Create Hadmard Quadratic object
        QoI = examples.HadamardQuadratic(systemsize, eigen_decayrate)

        # Create the joint distribution
        jdist = cp.J(cp.Uniform(-1,1),
                     cp.Uniform(-1,1),
                     cp.Uniform(-1,1),
                     cp.Uniform(-1,1))

    def demo_hadamard_accuracy(self):
        """
        Demo for how to use active subspace for reduced collocation
        """
        systemsize = 4
        eigen_decayrate = 2.0

        # Create Hadmard Quadratic object
        QoI = examples.HadamardQuadratic(systemsize, eigen_decayrate)

        mu = np.zeros(systemsize)
        std_dev = np.ones(systemsize)
        jdist = cp.J(cp.Normal(mu[0], std_dev[0]),
                     cp.Normal(mu[1], std_dev[1]),
                     cp.Normal(mu[2], std_dev[2]),
                     cp.Normal(mu[3], std_dev[3]),)
        # jdist = cp.MvNormal(mu, std_dev)

        active_subspace = ActiveSubspace(QoI, n_dominant_dimensions=4,
                                             n_monte_carlo_samples=10000,
                                             use_svd=True, read_rv_samples=False,
                                             write_rv_samples=False)
        active_subspace.getDominantDirections(QoI, jdist)

        mu_j_analytical = QoI.eval_analytical_QoI_mean(mu, cp.Cov(jdist))
        var_j_analytical = QoI.eval_analytical_QoI_variance(mu, cp.Cov(jdist))

        # Create reduced collocation object
        QoI_dict = {'Hadamard' : {'QoI_func' : QoI.eval_QoI,
                                  'output_dimensions' : 1,
                                  },
                    }
        sc_obj_active = StochasticCollocation2(jdist, 4, 'MvNormal', QoI_dict,
                                        include_derivs=False,
                                        reduced_collocation=True,
                                        dominant_dir=active_subspace.dominant_dir)
        sc_obj_active.evaluateQoIs(jdist)

        mu_j_active = sc_obj_active.mean(of=['Hadamard'])
        var_j_active = sc_obj_active.variance(of=['Hadamard'])


if __name__ == '__main__':
    unittest.main()
