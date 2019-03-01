# check_optimal_uq_vals.py
# This file will contain the scripts to check how the optimial UQ values compare
# with each other for the different implementations. These implementations
# include
# 1. MFMC solution
# 2. Full collocation
# 3. Reduced collocation
import sys
import time
import pprint as pp

# pyStatReduce specific imports
import unittest
import numpy as np
import chaospy as cp
import copy
from pystatreduce.new_stochastic_collocation import StochasticCollocation2
from pystatreduce.stochastic_collocation import StochasticCollocation
from pystatreduce.monte_carlo import MonteCarlo
from pystatreduce.quantity_of_interest import QuantityOfInterest
from pystatreduce.dimension_reduction import DimensionReduction
from pystatreduce.stochastic_arnoldi.arnoldi_sample import ArnoldiSampling
import pystatreduce.examples as examples

#pyoptsparse sepecific imports
from scipy import sparse
import argparse
import pyoptsparse # from pyoptsparse import Optimization, OPT, SNOPT

# Import the OpenMDAo shenanigans
from openmdao.api import IndepVarComp, Problem, Group, NewtonSolver, \
    ScipyIterativeSolver, LinearBlockGS, NonlinearBlockGS, \
    DirectSolver, LinearBlockGS, PetscKSP, SqliteRecorder, ScipyOptimizeDriver

from openaerostruct.geometry.utils import generate_mesh
from openaerostruct.geometry.geometry_group import Geometry
from openaerostruct.aerodynamics.aero_groups import AeroPoint

# Declare some global default values
mean_Ma = 0.071
mean_TSFC = 9.80665 * 8.6e-6
mean_W0 = 10.0
mean_E = 85.e9
mean_G = 25.e9
mean_mrho = 1600
mean_R = 1800e3
mean_load_factor = 1.0

mu = np.array([mean_Ma, mean_TSFC, mean_W0, mean_E, mean_G, mean_mrho, mean_R, mean_load_factor])
std_dev = np.diag([0.005, 0.00607/3600, 0.2, 5.e9, 1.e9, 50, 500e3, 0.1])

class UQScanEagleOpt(object):
    """
    This class is the conduit for linking pyStatReduce and OpenAeroStruct with
    pyOptSparse.
    """
    def __init__(self, uq_systemsize, all_rv=False):

        # Total number of nodes to use in the spanwise (num_y) and
        # chordwise (num_x) directions. Vary these to change the level of fidelity.
        num_y = 21
        num_x = 3
        mesh_dict = {'num_y' : num_y,
                     'num_x' : num_x,
                     'wing_type' : 'rect',
                     'symmetry' : True,
                     'span_cos_spacing' : 0.5,
                     'span' : 3.11,
                     'root_chord' : 0.3,
                     }

        rv_dict = {'Mach_number' : mean_Ma,
                   'CT' : mean_TSFC,
                   'W0' : mean_W0,
                   'E' : mean_E, # surface RV
                   'G' : mean_G, # surface RV
                   'mrho' : mean_mrho, # surface RV
                   'R' : mean_R,
                   'load_factor' : mean_load_factor,
                    }

        dv_dict = {'n_twist_cp' : 3,
                   'n_thickness_cp' : 3,
                   'n_CM' : 3,
                   'n_thickness_intersects' : 10,
                   'n_constraints' : 1 + 10 + 1 + 3 + 3,
                   'ndv' : 3 + 3 + 2,
                   'mesh_dict' : mesh_dict,
                   'rv_dict' : rv_dict
                    }

        self.jdist = cp.MvNormal(mu, std_dev)
        self.QoI = examples.oas_scaneagle2.OASScanEagleWrapper2(uq_systemsize,
                                                                dv_dict)
        dfuelburn_dict = {'dv' : {'dQoI_func' : self.QoI.eval_ObjGradient_dv,
                                  'output_dimensions' : dv_dict['ndv'],
                                  }
                         }
        dcon_dict = {'dv' : {'dQoI_func' : self.QoI.eval_ConGradient_dv,
                             'output_dimensions' : dv_dict['ndv']
                            }
                    }
        dcon_failure_dict = {'dv' : {'dQoI_func' : self.QoI.eval_ConFailureGradient_dv,
                                     'output_dimensions' : dv_dict['ndv'],
                                    }
                            }
        self.QoI_dict = {'fuelburn' : {'QoI_func' : self.QoI.eval_QoI,
                                       'output_dimensions' : 1,
                                       'deriv_dict' : dfuelburn_dict
                                      },
                         # 'constraints' : {'QoI_func' : self.QoI.eval_AllConstraintQoI,
                         #                  'output_dimensions' : dv_dict['n_constraints'],
                         #                  'deriv_dict' : dcon_dict
                         #                 },
                         # 'con_failure' : {'QoI_func' : self.QoI.eval_confailureQoI,
                         #                  'output_dimensions' : 1,
                         #                  'deriv_dict' : dcon_failure_dict
                         #                 }
                        }

def eval_uq_fuelburn(dv_dict, collocation_obj):
    UQObj.QoI.p['oas_scaneagle.wing.twist_cp'] = dv_dict['twist_cp']
    UQObj.QoI.p['oas_scaneagle.wing.thickness_cp'] = dv_dict['thickness_cp']
    UQObj.QoI.p['oas_scaneagle.wing.sweep'] = dv_dict['sweep']
    UQObj.QoI.p['oas_scaneagle.alpha'] = dv_dict['alpha']


    # Compute statistical metrics
    collocation_obj.evaluateQoIs(UQObj.jdist)
    # print('fvals = ')
    # print("Mean fuelburn = ", init_mu_j['fuelburn'][0])
    # print("Variance fuelburn = ", init_var_j['fuelburn'][0])
    # print()
    # print(sc_obj.QoI_dict['fuelburn']['fvals'])
    mu_j = collocation_obj.mean(of=['fuelburn'])
    var_j = collocation_obj.variance(of=['fuelburn'])

    return mu_j, var_j

def get_iso_gradients(dv_dict):
    """
    Computes the gradient of the scaneagle problem in the isoprobabilistic space
    w.r.t the random variables.
    """
    UQObj.QoI.p['oas_scaneagle.wing.twist_cp'] = dv_dict['twist_cp']
    UQObj.QoI.p['oas_scaneagle.wing.thickness_cp'] = dv_dict['thickness_cp']
    UQObj.QoI.p['oas_scaneagle.wing.sweep'] = dv_dict['sweep']
    UQObj.QoI.p['oas_scaneagle.alpha'] = dv_dict['alpha']

    mu_val = cp.E(UQObj.jdist)
    covariance = cp.Cov(UQObj.jdist)
    # print('covariance = \n', covariance)
    sqrt_Sigma = np.sqrt(covariance) # !!!! ONLY FOR INDEPENDENT RVs !!!!
    # print('sqrt_Sigma = \n', sqrt_Sigma)
    grad = UQObj.QoI.eval_QoIGradient(mu_val, np.zeros(len(mu_val)))
    iso_grad = np.dot(grad, sqrt_Sigma)

    return iso_grad, grad

if __name__ == "__main__":
    # RDO factor 2 Results
    sc_sol_dict = { 'sc_init' : {'twist_cp' : 2.5*np.ones(3),
                                 'thickness_cp' : 1.e-3 * np.array([5.5, 5.5, 5.5]),
                                 'sweep' : 20.,
                                 'alpha' : 5.,
                                },

                    'sc_sol_1D' : {'twist_cp' : np.array([2.522784, 10, 5.0]),
                                'thickness_cp' : 1.e-3 * np.array([1.0, 1.0, 1.0]),
                                'sweep' : 18.91067,
                                'alpha' : 2.216330,
                                },

                    'sc_sol_2D' : {'twist_cp' : np.array([2.591936, 10, 5.0]),
                                'thickness_cp' : 1.e-3 * np.array([1.0, 1.0, 1.020770]),
                                'sweep' : 18.89184,
                                'alpha' : 2.183216,
                                },

                    'sc_sol_3D' : {'twist_cp' : np.array([2.625412, 10, 5.0]),
                                'thickness_cp' : 1.e-3 * np.array([1.0, 1.0, 1.0]),
                                'sweep' : 18.87683,
                                'alpha' : 2.144569,
                                },

                    'sc_sol_4D' : {'twist_cp' : np.array([2.591936, 10, 5.0]),
                                'thickness_cp' : 1.e-3 * np.array([1.0, 1.0, 1.004335]),
                                'sweep' : 18.86231,
                                'alpha' : 2.088403,
                                },

                    'sc_sol_5D' : {'twist_cp' : np.array([2.643664, 10, 5.0]),
                                'thickness_cp' : 1.e-3 * np.array([1.0, 1.0, 1.023398]),
                                'sweep' : 18.83332,
                                'alpha' : 1.974722,
                                },
                }

    # Step 1: Instantiate all objects needed for test
    uq_systemsize = len(mu)
    UQObj = UQScanEagleOpt(uq_systemsize, all_rv=True)
    UQObj.QoI.p['oas_scaneagle.wing.thickness_cp'] = 1.e-3 * np.array([5.5, 5.5, 5.5])
    UQObj.QoI.p['oas_scaneagle.wing.twist_cp'] = 2.5*np.ones(3)
    UQObj.QoI.p['oas_scaneagle.wing.sweep'] = 20.0
    UQObj.QoI.p['oas_scaneagle.alpha'] = 5.0
    UQObj.QoI.p.final_setup()

    # Dominant dominant
    sample_radius = 1.e-2
    dominant_space = DimensionReduction(n_arnoldi_sample=uq_systemsize+1,
                                        exact_Hessian=False,
                                        sample_radius=sample_radius)
    dominant_space.getDominantDirections(UQObj.QoI, UQObj.jdist, max_eigenmodes=8)
    print('\n#-----------------------------------------------------------#')
    print('sample radius = ', sample_radius)
    print('iso_eigenvals = ', dominant_space.iso_eigenvals)
    # Full collocation
    sc_obj = StochasticCollocation2(UQObj.jdist, 3, 'MvNormal', UQObj.QoI_dict,
                                    include_derivs=False)

    iso_grad, reg_grad = get_iso_gradients(sc_sol_dict['sc_init'])
    # print('reg_grad = \n', reg_grad)
    # print('iso_grad = \n', iso_grad)

    # mu_j_full, var_j_full = eval_uq_fuelburn(sc_sol_dict['sc_init'], sc_obj)
    print('\n#-----------------------------------------------------------#')
    mu_j_full = 5.39597929
    var_j_full = 3.86455449
    print('mu_j_full = ', mu_j_full)
    print('var_j_full = ', var_j_full)

    # # Reduced collocation
    # i = 1
    # for sc_sol in sc_sol_dict:
    #     if sc_sol != 'sc_init':
    #         dominant_dir = dominant_space.iso_eigenvecs[:,0:i]
    #         red_sc_obj = StochasticCollocation2(UQObj.jdist, 3, 'MvNormal', UQObj.QoI_dict,
    #                                         include_derivs=False, reduced_collocation=True,
    #                                         dominant_dir=dominant_dir)
    #         mu_j_red, var_j_red = eval_uq_fuelburn(sc_sol_dict[sc_sol], red_sc_obj)
    #         print('\n', i)
    #         print('mu_j_red = ', mu_j_red)
    #         print('var_j_red = ', var_j_red)
    #         i += 1

    dominant_dir = dominant_space.iso_eigenvecs[:,0:2]
    # print('dominant_dir = \n', dominant_dir)
    red_sc_obj = StochasticCollocation2(UQObj.jdist, 3, 'MvNormal', UQObj.QoI_dict,
                                        include_derivs=False, reduced_collocation=True,
                                        dominant_dir=dominant_dir)
    mu_j_red, var_j_red = eval_uq_fuelburn(sc_sol_dict['sc_init'], red_sc_obj)
    print('mu_j_red = ', mu_j_red['fuelburn'][0])
    print('var_j_red = ', var_j_red['fuelburn'][0,0])

    err_mu = abs((mu_j_red['fuelburn'][0] - mu_j_full) / mu_j_full)
    err_var = abs((var_j_red['fuelburn'][0,0] - var_j_full) / var_j_full)
    print('err_mu = ', err_mu)
    print('err_var = ', err_var)
