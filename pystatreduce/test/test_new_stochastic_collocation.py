# Test new_stochastic_collocation.py
# Test stochastic collocation module
import unittest
import numpy as np
import chaospy as cp

from pystatreduce.new_stochastic_collocation import StochasticCollocation2
from pystatreduce.stochastic_collocation import StochasticCollocation
from pystatreduce.quantity_of_interest import QuantityOfInterest
from pystatreduce.dimension_reduction import DimensionReduction
import pystatreduce.examples as examples

class NewStochasticCollocationTest(unittest.TestCase):
    def test_normalStochasticCollocation3D(self):
        systemsize = 3
        mu = np.random.randn(systemsize)
        std_dev = np.diag(np.random.rand(systemsize))
        jdist = cp.MvNormal(mu, std_dev)
        # Create QoI Object
        QoI = examples.Paraboloid3D(systemsize)

        # Create the Stochastic Collocation object
        deriv_dict = {'xi' : {'dQoI_func' : QoI.eval_QoIGradient,
                              'output_dimensions' : systemsize}
                     }
        QoI_dict = {'paraboloid' : {'QoI_func' : QoI.eval_QoI,
                                    'output_dimensions' : 1,
                                    'deriv_dict' : deriv_dict
                                    }
                    }
        sc_obj = StochasticCollocation2(jdist, 3, 'MvNormal', QoI_dict)
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

    def test_derivatives_scalarQoI(self):
        systemsize = 3
        mu = np.random.randn(systemsize)
        std_dev = np.diag(np.random.rand(systemsize))
        jdist = cp.MvNormal(mu, std_dev)
        # Create QoI Object
        QoI = examples.Paraboloid3D(systemsize)

        # Create the Stochastic Collocation object
        deriv_dict = {'xi' : {'dQoI_func' : QoI.eval_QoIGradient,
                              'output_dimensions' : systemsize}
                     }
        QoI_dict = {'paraboloid' : {'QoI_func' : QoI.eval_QoI,
                                    'output_dimensions' : 1,
                                    'deriv_dict' : deriv_dict
                                    }
                    }
        sc_obj = StochasticCollocation2(jdist, 3, 'MvNormal', QoI_dict, include_derivs=True)
        sc_obj.evaluateQoIs(jdist, include_derivs=True)
        dmu_j = sc_obj.dmean(of=['paraboloid'], wrt=['xi'])
        # dvar_j = sc_obj.dvariance(of=['paraboloid'], wrt=['xi'])

        # Analytical dmu_j
        dmu_j_analytical = np.array([100*mu[0], 50*mu[1], 2*mu[2]])
        err = abs((dmu_j['paraboloid']['xi'] - dmu_j_analytical) / dmu_j_analytical)
        self.assertTrue((err < 1.e-12).all())
        # # Analytical dvar_j
        # rv_dev = cp.Std(jdist)
        # dvar_j_analytical = np.array([(100*rv_dev[0])**2, (50*rv_dev[1])**2, (2*rv_dev[2])**2])
        # err = abs((dvar_j['paraboloid']['xi'] - dvar_j_analytical) / dvar_j_analytical)

    def test_nonrv_derivatives(self):
        # This test checks the analytical derivative w.r.t complex step
        systemsize = 2
        n_parameters = 2
        mu = np.random.randn(systemsize)
        std_dev = np.diag(np.random.rand(systemsize))
        jdist = cp.MvNormal(mu, std_dev)
        QoI = examples.PolyRVDV(data_type=complex)
        # Create the Stochastic Collocation object
        deriv_dict = {'dv' : {'dQoI_func' : QoI.eval_QoIGradient_dv,
                              'output_dimensions' : n_parameters}
                     }
        QoI_dict = {'PolyRVDV' : {'QoI_func' : QoI.eval_QoI,
                                  'output_dimensions' : 1,
                                  'deriv_dict' : deriv_dict
                                  }
                    }
        dv = np.random.randn(systemsize) + 0j
        QoI.set_dv(dv)
        sc_obj = StochasticCollocation2(jdist, 3, 'MvNormal', QoI_dict,
                                        include_derivs=True, data_type=complex)
        sc_obj.evaluateQoIs(jdist, include_derivs=True)
        dmu_j = sc_obj.dmean(of=['PolyRVDV'], wrt=['dv'])
        dvar_j = sc_obj.dvariance(of=['PolyRVDV'], wrt=['dv'])

        # Lets do complex step
        pert = complex(0, 1e-30)
        dmu_j_complex = np.zeros(n_parameters, dtype=complex)
        dvar_j_complex = np.zeros(n_parameters, dtype=complex)
        for i in range(0, n_parameters):
            dv[i] += pert
            QoI.set_dv(dv)
            sc_obj.evaluateQoIs(jdist, include_derivs=False)
            mu_j = sc_obj.mean(of=['PolyRVDV'])
            var_j = sc_obj.variance(of=['PolyRVDV'])
            dmu_j_complex[i] = mu_j['PolyRVDV'].imag / pert.imag
            dvar_j_complex[i] = var_j['PolyRVDV'].imag / pert.imag
            dv[i] -= pert

        np.set_printoptions(precision=12)
        err1 = dmu_j['PolyRVDV']['dv'] - dmu_j_complex
        self.assertTrue((err1 < 1.e-13).all())

        err2 = dvar_j['PolyRVDV']['dv'] - dvar_j_complex
        self.assertTrue((err2 < 1.e-13).all())


if __name__ == "__main__":
    unittest.main()
