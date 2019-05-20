# test_active_subspace.py
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
import pystatreduce.optimize.OAS_ScanEagle.oas_scaneagle_opt as scaneagle_opt

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
    """
    def util_func(arb):

        active_dominant_dir = active_subspace.dominant_dir
        print('Active Subspace')
        print('iso_eigenvals = ', active_subspace.iso_eigenvals)
        print('iso_eigenvecs = \n', active_subspace.iso_eigenvecs)

        # Create a dimension_reduction object
        dominant_space = DimensionReduction(threshold_factor=0.9,
                                            exact_Hessian=True)
        dominant_space.getDominantDirections(QoI, jdist)
        dominant_dir = dominant_space.dominant_dir
        print('\nDominant Directions.dominant_indices = ', dominant_space.dominant_indices)
        print('iso_eigenvals = ', dominant_space.iso_eigenvals)
        print('iso_eigenvecs = \n', dominant_space.iso_eigenvecs)

        # Compute the analytical value of the mean
        # mu_j_analytical = QoI.eval_analytical_QoI_mean(mu, cp.Cov(jdist))

        # Create the QoI Dictionary
        QoI_dict = {'Hadamard' : {'QoI_func' : QoI.eval_QoI,
                                  'output_dimensions' : 1,
                                  },
                    'exponential' : {'QoI_func' : QoI.eval_QoI,
                                              'output_dimensions' : 1,
                                    },
                    }

        # Create a stochastic collocation object
        sc_obj_active = StochasticCollocation2(jdist, 4, 'MvNormal', QoI_dict,
                                        include_derivs=False,
                                        reduced_collocation=True,
                                        dominant_dir=active_dominant_dir)
        sc_obj_active.evaluateQoIs(jdist)

        sc_obj = StochasticCollocation2(jdist, 4, 'MvNormal', QoI_dict,
                                        include_derivs=False,
                                        reduced_collocation=True,
                                        dominant_dir=dominant_dir)
        sc_obj.evaluateQoIs(jdist)

        sc_obj_full = StochasticCollocation2(jdist, 4, 'MvNormal', QoI_dict,
                                             include_derivs=False,
                                             reduced_collocation=False)
        sc_obj_full.evaluateQoIs(jdist)



        mu_j_active = sc_obj_active.mean(of=['exponential'])
        mu_j = sc_obj.mean(of=['exponential'])
        mu_j_full = sc_obj_full.mean(of=['exponential'])
        print('\nmu_j = ', mu_j['exponential'])
        print('mu_j_active = ', mu_j_active['exponential'])
        print('mu_j_full = ', mu_j_full['exponential'])
        # print('mu_j_analytical = ', mu_j_analytical)
    """
if __name__ == '__main__':
    unittest.main()
