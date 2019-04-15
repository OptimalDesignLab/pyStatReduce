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
import pystatreduce.optimize.OAS_ScanEagle.oas_scaneagle_opt as scaneagle_opt

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

# # Default mean values
# mean_Ma = 0.071
# mean_TSFC = 9.80665 * 8.6e-6
# mean_W0 = 10.0
# mean_E = 85.e9
# mean_G = 25.e9
# mean_mrho = 1600
# mean_R = 1800
# mean_load_factor = 1.0
# mean_altitude = 4.57
# # Default standard values
# std_dev_Ma = 0.005
# std_dev_TSFC = 0.00607/3600
# std_dev_W0 = 0.2
# std_dev_mrho = 50
# std_dev_R = 500
# std_dev_load_factor = 0.1
# std_dev_E = 5.e9
# std_dev_G = 1.e9
# std_dev_altitude = 0.5

# Default mean values
mean_Ma = 0.08
mean_TSFC = 9.80665 * 8.6e-6 * 3600
mean_W0 = 10 # 10.0
mean_E = 85.e9
mean_G = 25.e9
mean_mrho = 1600
mean_R = 1800
mean_load_factor = 1.0
mean_altitude = 4.57
# Default standard values
std_dev_Ma = 0.005 # 0.015
std_dev_TSFC = 0.00607 # /3600
std_dev_W0 = 1
std_dev_mrho = 50
std_dev_R = 300 # 500
std_dev_load_factor = 0.3
std_dev_E = 5.e9
std_dev_G = 1.e9
std_dev_altitude = 0.1


def eval_uq_fuelburn(dv_dict, collocation_obj):
    UQObj.QoI.p['oas_scaneagle.wing.twist_cp'] = dv_dict['twist_cp']
    UQObj.QoI.p['oas_scaneagle.wing.thickness_cp'] = dv_dict['thickness_cp']
    UQObj.QoI.p['oas_scaneagle.wing.sweep'] = dv_dict['sweep']
    UQObj.QoI.p['oas_scaneagle.alpha'] = dv_dict['alpha']


    # Compute statistical metrics
    collocation_obj.evaluateQoIs(UQObj.jdist)
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
    sqrt_Sigma = np.sqrt(covariance) # !!!! ONLY FOR INDEPENDENT RVs !!!!
    print('mu_val = ', mu_val)
    print('covariance = ', covariance)
    # print('sqrt_Sigma = \n', sqrt_Sigma)
    grad = UQObj.QoI.eval_QoIGradient(mu_val, np.zeros(len(mu_val)))
    iso_grad = np.dot(grad, sqrt_Sigma)

    return iso_grad, grad

if __name__ == "__main__":
    # Declare the dictionary
    rv_dict = { 'Mach_number' : {'mean' : mean_Ma,
                                 'std_dev' : std_dev_Ma},
                'CT' : {'mean' : mean_TSFC,
                        'std_dev' : std_dev_TSFC},
                'W0' : {'mean' : mean_W0,
                        'std_dev' : std_dev_W0},
                'R' : {'mean' : mean_R,
                       'std_dev' : std_dev_R},
                'load_factor' : {'mean' : mean_load_factor,
                                 'std_dev' : std_dev_load_factor},
                'mrho' : {'mean' : mean_mrho,
                         'std_dev' : std_dev_mrho},
                'altitude' : {'mean' : mean_altitude,
                              'std_dev' : std_dev_altitude},

               }

    # RDO factor 2 Results
    sc_sol_dict = { 'sc_init' : {'twist_cp' : np.array([2.5, 2.5, 5.0]), # 2.5*np.ones(3),
                                 'thickness_cp' : np.array([0.008, 0.008, 0.008]), # 1.e-3 * np.array([5.5, 5.5, 5.5]),
                                 'sweep' : 20.,
                                 'alpha' : 5.,
                                },
                    
                    'deterministic' : {'twist_cp' : np.array([3.37718983, 10, 5]) ,
                                       'thickness_cp' : np.array([0.001, 0.001, 0.00114519]), # 1.e-3 * np.ones(3) ,
                                       'sweep' : 17.97227386,
                                       'alpha' : -0.24701157},

                    # Optimal design values obtained using the initial deterministic design variables
                    # 7 RV, sample radius = 1.e-1, n_dominant_dir = 1, rdo factor=2
                    '7rv_1e_1_1_2' :  {'twist_cp' : np.array([4.5766595, 10, 5]),
                                       'thickness_cp' : 1.e-3 * np.ones(3),
                                       'sweep' : 17.93557251,
                                       'alpha' : 0.52838575},
                    # 7 RV, sample radius = 1.e-1, n_dominant_dir = 2, rdo factor=2
                    '7rv_1e_1_2_2' :  {'twist_cp' : np.array([4.82659706, 10, 5]),
                                       'thickness_cp' : 1.e-3 * np.ones(3),
                                       'sweep' : 17.59063958,
                                       'alpha' : -0.09187924},
                    # 7 RV, sample radius = 1.e-1, n_dominant_dir = 3, rdo factor=2
                    '7rv_1e_1_3_2' :  {'twist_cp' : np.array([4.9103329, 10, 5]),
                                       'thickness_cp' : 1.e-3 * np.ones(3),
                                       'sweep' : 17.46974998,
                                       'alpha' : -0.28339684},
                    # 7 RV, sample radius = 1.e-1, n_dominant_dir = 4, rdo factor=2
                    '7rv_1e_1_4_2' :  {'twist_cp' : np.array([4.85883309, 10, 5]),
                                       'thickness_cp' : 1.e-3 * np.ones(3),
                                       'sweep' : 17.48184078,
                                       'alpha' : -0.28440502},
                    # 7 RV, sample radius = 1.e-1, n_dominant_dir = 5, rdo factor=2
                    '7rv_1e_1_5_2' :  {'twist_cp' : np.array([4.87843855, 10, 5]),
                                       'thickness_cp' : 1.e-3 * np.ones(3),
                                       'sweep' : 17.4693496,
                                       'alpha' : -0.3003593},
                    # 7 RV, sample radius = 1.e-1, n_dominant_dir = 6, rdo factor=2
                    '7rv_1e_1_6_2' :  {'twist_cp' : np.array([4.92920057, 10, 5]),
                                       'thickness_cp' : 1.e-3 * np.ones(3),
                                       'sweep' : 17.45015421,
                                       'alpha' : -0.31792675},
                    # 7rv, Full Monte Carlo simulation
                    '7rv_mc' :  {'twist_cp' : np.array([4.93195234, 10, 5]),
                                 'thickness_cp' : 1.e-3 * np.ones(3),
                                 'sweep' : 17.48339411,
                                 'alpha' : -0.25422074},

                    '7rv_full_2' : {'twist_cp' : np.array([4.87264491, 10, 5]),
                                    'thickness_cp' : 1.e-3 * np.ones(3),
                                    'sweep' : 17.54128459,
                                    'alpha' : -0.17390615},

                    # '7rv_1e_1_2_3' : {'twist_cp' : np.array([4.819646, 10, 5]),
                    #                    'thickness_cp' : 1.e-3 * np.ones(3),
                    #                    'sweep' : 17.5924635,
                    #                    'alpha' : -0.09117171},
                    }

    # Step 1: Instantiate all objects needed for test
    UQObj = scaneagle_opt.UQScanEagleOpt(rv_dict, design_point=sc_sol_dict['deterministic'] ,
                                         krylov_pert=1.e-1, max_eigenmodes=2)

    # Full collocation
    sc_obj = StochasticCollocation2(UQObj.jdist, 3, 'MvNormal', UQObj.QoI_dict,
                                    include_derivs=False)

    # iso_grad, reg_grad = get_iso_gradients(sc_sol_dict['sc_init'])
    # print('reg_grad = \n', reg_grad)
    # print('iso_grad = \n', iso_grad)

    # mu_j_full, var_j_full = eval_uq_fuelburn(sc_sol_dict['sc_init'], sc_obj)
    print("eigenvals = ", UQObj.dominant_space.iso_eigenvals)
    print('eigenvecs = \n', UQObj.dominant_space.iso_eigenvecs)
    print('\n#-----------------------------------------------------------#')

    red_sc_obj = StochasticCollocation2(UQObj.jdist, 3, 'MvNormal', UQObj.QoI_dict,
                                        include_derivs=False, reduced_collocation=True,
                                        dominant_dir=UQObj.dominant_space.dominant_dir)
    key_name = '7rv_full_2'
    print('key_name = ', key_name)
    mu_j_red, var_j_red = eval_uq_fuelburn(sc_sol_dict[key_name], red_sc_obj)
    print('mu_j_red = ', mu_j_red['fuelburn'][0])
    print('var_j_red = ', var_j_red['fuelburn'][0,0])

    # mu_j_full, var_j_full = eval_uq_fuelburn(sc_sol_dict[key_name], sc_obj)
    # print('mu_j_full = ', mu_j_full['fuelburn'][0])
    # print('var_j_full = ', var_j_full['fuelburn'][0,0])

    # for sc_sol in sc_sol_dict:
    #     print('\ndesign point = ', sc_sol)
    #     mu_j, var_j = eval_uq_fuelburn(sc_sol_dict[sc_sol], sc_obj)
    #     print('mu_j = ', mu_j)
    #     print('var_j = ', var_j)
