# test_stochastic_collocation3.py
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

np.set_printoptions(precision=16)

# Declare some global variables that will be used for testing. This is done to
# prevent possible discrepancies while adding more test or using a different
# computing platform.
mean_2dim = np.random.randn(2)
mean_3dim = np.random.randn(3)
std_dev_2dim = abs(np.diag(np.random.randn(2)))
std_dev_3dim = abs(np.diag(np.random.randn(3)))

class StochasticCollocation3Test(unittest.TestCase):
    def test_normalStochasticCollocation3D(self):
        systemsize = 3
        mu = mean_3dim # np.random.randn(systemsize)
        std_dev = std_dev_3dim # np.diag(np.random.rand(systemsize))
        jdist = cp.MvNormal(mu, std_dev)
        # Create QoI Object
        QoI = examples.Paraboloid3D(systemsize)

        # Create the Stochastic Collocation object
        deriv_dict = {'xi' : {'dQoI_func' : QoI.eval_QoIGradient,
                              'output_dimensions' : systemsize}
                     }
        QoI_dict = {'paraboloid' : {'quadrature_degree' : 3,
                                    'reduced_collocation' : False,
                                    'QoI_func' : QoI.eval_QoI,
                                    'output_dimensions' : 1,
                                    'include_derivs' : False,
                                    'deriv_dict' : deriv_dict
                                    }
                    }
        sc_obj = StochasticCollocation3(jdist, 'MvNormal', QoI_dict)
        sc_obj.evaluateQoIs(jdist)
        mu_js = sc_obj.mean(of=['paraboloid'])
        var_js = sc_obj.variance(of=['paraboloid'])

        # Analytical mean
        mu_j_analytical = QoI.eval_QoI_analyticalmean(mu, cp.Cov(jdist))
        err = abs((mu_js['paraboloid'][0] - mu_j_analytical)/ mu_j_analytical)
        self.assertTrue(err < 1.e-15)

        # Analytical variance
        var_j_analytical = QoI.eval_QoI_analyticalvariance(mu, cp.Cov(jdist))
        err = abs((var_js['paraboloid'][0,0] - var_j_analytical) / var_j_analytical)
        self.assertTrue(err < 1.e-15)

    def test_multipleQoI(self):

        # This tests for multiple QoIs. We only compute the mean in this test,
        # because it is only checking if it can do multiple loops

        systemsize = 2
        theta = 0
        mu = mean_2dim # np.random.randn(systemsize)
        std_dev = std_dev_2dim # np.diag(np.random.rand(systemsize))
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
        sc_obj = StochasticCollocation3(jdist, 'MvNormal', QoI_dict)
        sc_obj.evaluateQoIs(jdist)
        mu_js = sc_obj.mean(of=['paraboloid2', 'PolyRVDV'])

        # Compare against known values
        # 1. Paraboloid2D, we use nested loops
        mu_j1_analytical = QoI1.eval_QoI_analyticalmean(mu, cp.Cov(jdist))
        err = abs((mu_js['paraboloid2'][0] - mu_j1_analytical)/ mu_j1_analytical)
        self.assertTrue(err < 1.e-15)

    def test_reduced_normalStochasticCollocation3D(self):

        # This is not a very good test because we are comparing the reduced collocation
        # against the analytical expected value. The only hting it tells us is that
        # the solution is within the ball park of actual value. We still need to
        # come up with a better test.

        systemsize = 3
        mu = mean_3dim # np.random.randn(systemsize)
        std_dev = std_dev_3dim # abs(np.diag(np.random.randn(systemsize)))
        jdist = cp.MvNormal(mu, std_dev)
        # Create QoI Object
        QoI = examples.Paraboloid3D(systemsize)
        dominant_dir = np.array([[1.0, 0.0],[0.0, 1.0],[0.0, 0.0]], dtype=np.float)
        # Create the Stochastic Collocation object
        deriv_dict = {'xi' : {'dQoI_func' : QoI.eval_QoIGradient,
                              'output_dimensions' : systemsize}
                     }
        QoI_dict = {'paraboloid' : {'quadrature_degree' : 3,
                                    'reduced_collocation' : False,
                                    'QoI_func' : QoI.eval_QoI,
                                    'output_dimensions' : 1,
                                    'include_derivs' : False,
                                    'deriv_dict' : deriv_dict
                                    }
                    }
        sc_obj = StochasticCollocation3(jdist, 'MvNormal', QoI_dict)
        sc_obj.evaluateQoIs(jdist)
        mu_js = sc_obj.mean(of=['paraboloid'])
        var_js = sc_obj.variance(of=['paraboloid'])
        # Analytical mean
        mu_j_analytical = QoI.eval_QoI_analyticalmean(mu, cp.Cov(jdist))
        err = abs((mu_js['paraboloid'][0] - mu_j_analytical)/ mu_j_analytical)
        self.assertTrue(err < 1e-1)
        # Analytical variance
        var_j_analytical = QoI.eval_QoI_analyticalvariance(mu, cp.Cov(jdist))
        err = abs((var_js['paraboloid'][0,0] - var_j_analytical) / var_j_analytical)
        self.assertTrue(err < 0.01)

    def test_derivatives_scalarQoI(self):
        systemsize = 3
        mu = mean_3dim # np.random.randn(systemsize)
        std_dev = std_dev_3dim # np.diag(np.random.rand(systemsize))
        jdist = cp.MvNormal(mu, std_dev)
        # Create QoI Object
        QoI = examples.Paraboloid3D(systemsize)

        # Create the Stochastic Collocation object
        deriv_dict = {'xi' : {'dQoI_func' : QoI.eval_QoIGradient,
                              'output_dimensions' : systemsize}
                     }
        QoI_dict = {'paraboloid' : {'quadrature_degree' : 3,
                                    'reduced_collocation' : False,
                                    'QoI_func' : QoI.eval_QoI,
                                    'output_dimensions' : 1,
                                    'include_derivs' : True,
                                    'deriv_dict' : deriv_dict
                                    }
                    }
        sc_obj = StochasticCollocation3(jdist, 'MvNormal', QoI_dict)
        sc_obj.evaluateQoIs(jdist)
        dmu_j = sc_obj.dmean(of=['paraboloid'], wrt=['xi'])
        # dvar_j = sc_obj.dvariance(of=['paraboloid'], wrt=['xi'])

        # Analytical dmu_j
        dmu_j_analytical = np.array([100*mu[0], 50*mu[1], 2*mu[2]])
        err = abs((dmu_j['paraboloid']['xi'] - dmu_j_analytical) / dmu_j_analytical)
        self.assertTrue((err < 1.e-12).all())

    def test_nonrv_derivatives(self):
        # This test checks the analytical derivative w.r.t complex step
        systemsize = 2
        n_parameters = 2
        mu = mean_2dim # np.random.randn(systemsize)
        std_dev = std_dev_2dim # np.diag(np.random.rand(systemsize))
        jdist = cp.MvNormal(mu, std_dev)
        QoI = examples.PolyRVDV(data_type=complex)
        # Create the Stochastic Collocation object
        deriv_dict = {'dv' : {'dQoI_func' : QoI.eval_QoIGradient_dv,
                              'output_dimensions' : n_parameters}
                     }
        QoI_dict = {'PolyRVDV' : {'quadrature_degree' : 3,
                                  'reduced_collocation' : False,
                                  'QoI_func' : QoI.eval_QoI,
                                  'output_dimensions' : 1,
                                  'include_derivs' : True,
                                  'deriv_dict' : deriv_dict
                                  }
                    }
        dv = np.random.randn(systemsize) + 0j
        QoI.set_dv(dv)
        sc_obj = StochasticCollocation3(jdist, 'MvNormal', QoI_dict, data_type=complex)
        sc_obj.evaluateQoIs(jdist)
        dmu_j = sc_obj.dmean(of=['PolyRVDV'], wrt=['dv'])
        dvar_j = sc_obj.dvariance(of=['PolyRVDV'], wrt=['dv'])
        dstd_dev = sc_obj.dStdDev(of=['PolyRVDV'], wrt=['dv'])

        # Lets do complex step
        pert = complex(0, 1e-30)
        dmu_j_complex = np.zeros(n_parameters, dtype=complex)
        dvar_j_complex = np.zeros(n_parameters, dtype=complex)
        dstd_dev_complex = np.zeros(n_parameters, dtype=complex)
        for i in range(0, n_parameters):
            dv[i] += pert
            QoI.set_dv(dv)
            sc_obj.evaluateQoIs(jdist)
            mu_j = sc_obj.mean(of=['PolyRVDV'])
            var_j = sc_obj.variance(of=['PolyRVDV'])
            std_dev_j = np.sqrt(var_j['PolyRVDV'][0,0])
            dmu_j_complex[i] = mu_j['PolyRVDV'].imag / pert.imag
            dvar_j_complex[i] = var_j['PolyRVDV'].imag / pert.imag
            dstd_dev_complex[i] = std_dev_j.imag / pert.imag
            dv[i] -= pert

        err1 = dmu_j['PolyRVDV']['dv'] - dmu_j_complex
        self.assertTrue((err1 < 1.e-13).all())

        err2 = dvar_j['PolyRVDV']['dv'] - dvar_j_complex
        self.assertTrue((err2 < 1.e-13).all())

        err3 = dstd_dev['PolyRVDV']['dv'] - dstd_dev_complex
        self.assertTrue((err2 < 1.e-13).all())

    def test_nonrv_derivatives_reduced_collocation(self):
        # This test checks the analytical derivative w.r.t complex step
        systemsize = 2
        n_parameters = 2
        mu = mean_2dim # np.random.randn(systemsize)
        std_dev = std_dev_2dim # abs(np.diag(np.random.randn(systemsize)))
        jdist = cp.MvNormal(mu, std_dev)
        QoI = examples.PolyRVDV(data_type=complex)

        dv = np.random.randn(systemsize) + 0j
        QoI.set_dv(dv)
        # Create dimension reduction object
        threshold_factor = 0.9
        dominant_space = DimensionReduction(threshold_factor=threshold_factor,
                                            exact_Hessian=True)
        # Get the eigenmodes of the Hessian product and the dominant indices
        dominant_space.getDominantDirections(QoI, jdist)
        dominant_dir = dominant_space.iso_eigenvecs[:, dominant_space.dominant_indices]
        # Create the Stochastic Collocation object
        deriv_dict = {'dv' : {'dQoI_func' : QoI.eval_QoIGradient_dv,
                              'output_dimensions' : n_parameters}
                     }
        QoI_dict = {'PolyRVDV' : {'quadrature_degree' : 3,
                                  'reduced_collocation' : False,
                                  'QoI_func' : QoI.eval_QoI,
                                  'output_dimensions' : 1,
                                  'dominant_dir' : dominant_dir,
                                  'include_derivs' : True,
                                  'deriv_dict' : deriv_dict
                                  }
                    }
        sc_obj = StochasticCollocation3(jdist, 'MvNormal', QoI_dict, data_type=complex)
        sc_obj.evaluateQoIs(jdist)
        dmu_j = sc_obj.dmean(of=['PolyRVDV'], wrt=['dv'])
        dvar_j = sc_obj.dvariance(of=['PolyRVDV'], wrt=['dv'])
        dstd_dev = sc_obj.dStdDev(of=['PolyRVDV'], wrt=['dv'])

        # Lets do complex step
        pert = complex(0, 1e-30)
        dmu_j_complex = np.zeros(n_parameters, dtype=complex)
        dvar_j_complex = np.zeros(n_parameters, dtype=complex)
        dstd_dev_complex = np.zeros(n_parameters, dtype=complex)
        for i in range(0, n_parameters):
            dv[i] += pert
            QoI.set_dv(dv)
            sc_obj.evaluateQoIs(jdist)
            mu_j = sc_obj.mean(of=['PolyRVDV'])
            var_j = sc_obj.variance(of=['PolyRVDV'])
            std_dev_j = np.sqrt(var_j['PolyRVDV'][0,0])
            dmu_j_complex[i] = mu_j['PolyRVDV'].imag / pert.imag
            dvar_j_complex[i] = var_j['PolyRVDV'].imag / pert.imag
            dstd_dev_complex[i] = std_dev_j.imag / pert.imag
            dv[i] -= pert

        err1 = dmu_j['PolyRVDV']['dv'] - dmu_j_complex
        self.assertTrue((err1 < 1.e-13).all())

        err2 = dvar_j['PolyRVDV']['dv'] - dvar_j_complex
        self.assertTrue((err2 < 1.e-10).all())

        err3 = dstd_dev['PolyRVDV']['dv'] - dstd_dev_complex
        self.assertTrue((err2 < 1.e-12).all())

if __name__ == "__main__":
    unittest.main()
