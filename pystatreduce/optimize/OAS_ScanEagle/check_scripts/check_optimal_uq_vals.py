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

import pystatreduce.optimize.OAS_ScanEagle.check_scripts.optimal_vals_dict as optimal_vals_dict
from pystatreduce.optimize.OAS_ScanEagle.mean_values import *

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

def eval_objective_and_constraint(dv_dict, collocation_obj):
    UQObj.QoI.p['oas_scaneagle.wing.twist_cp'] = dv_dict['twist_cp']
    UQObj.QoI.p['oas_scaneagle.wing.thickness_cp'] = dv_dict['thickness_cp']
    UQObj.QoI.p['oas_scaneagle.wing.sweep'] = dv_dict['sweep']
    UQObj.QoI.p['oas_scaneagle.alpha'] = dv_dict['alpha']


    # Compute statistical metrics
    if isinstance(collocation_obj, StochasticCollocation2):
        collocation_obj.evaluateQoIs(UQObj.jdist)
        mu_j = collocation_obj.mean(of=['fuelburn', 'constraints'])
        var_j = collocation_obj.variance(of=['fuelburn', 'con_failure'])
    elif isinstance(collocation_obj, MonteCarlo):
        collocation_obj.getSamples(UQObj.jdist, include_derivs=False)
        mu_j = collocation_obj.mean(UQObj.jdist, of=['fuelburn', 'constraints'])
        var_j = collocation_obj.variance(UQObj.jdist, of=['fuelburn', 'con_failure'])

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

    sc_sol_dict = optimal_vals_dict.sc_sol_dict

    # Step 1: Instantiate all objects needed for test
    UQObj = scaneagle_opt.UQScanEagleOpt(rv_dict, design_point=sc_sol_dict['sc_init'] ,
                                         active_subspace=False, n_as_samples=1000,
                                         krylov_pert=1.e-1, max_eigenmodes=int(sys.argv[1]))
    n_thickness_intersects = UQObj.QoI.p['oas_scaneagle.AS_point_0.wing_perf.thickness_intersects'].size
    n_CM = 3

    print("eigenvals = ", UQObj.dominant_space.iso_eigenvals)
    print('eigenvecs = \n', UQObj.dominant_space.iso_eigenvecs)
    print('dominant_dir = \n', UQObj.dominant_space.dominant_dir)
    print('\n#-----------------------------------------------------------#')

    # Full collocation
    use_stochastic_collocation = True
    use_monte_carlo = False
    if use_stochastic_collocation:
        colloc_obj = StochasticCollocation2(UQObj.jdist, 3, 'MvNormal', UQObj.QoI_dict,
                                        include_derivs=False)
        red_colloc_obj = StochasticCollocation2(UQObj.jdist, 3, 'MvNormal', UQObj.QoI_dict,
                                            include_derivs=False, reduced_collocation=True,
                                            # dominant_dir=custom_eigenvec[:,0:int(sys.argv[1])])
                                            dominant_dir=UQObj.dominant_space.dominant_dir)
    elif use_monte_carlo:
        nsample = 3
        colloc_obj = MonteCarlo(nsample, UQObj.jdist, UQObj.QoI_dict, include_derivs=False)
        red_colloc_obj = MonteCarlo(nsample, UQObj.jdist, UQObj.QoI_dict,
                                    reduced_collocation=True,
                                    dominant_dir=UQObj.dominant_space.dominant_dir,
                                    include_derivs=False)

    # iso_grad, reg_grad = get_iso_gradients(sc_sol_dict['sc_init'])
    # print('reg_grad = \n', reg_grad)
    # print('iso_grad = \n', iso_grad)

    key_name = 'act_init_7rv_2_2_lf1'
    print('key_name = ', key_name)
    start_time = time.time()
    mu_j_full, var_j_full = eval_uq_fuelburn(sc_sol_dict[key_name], colloc_obj)
    time_elapsed = time.time() - start_time
    # mu_j_full, var_j_full = eval_objective_and_constraint(sc_sol_dict[key_name], colloc_obj)

    mu_mc_mil = 4.681601386475621
    var_mc_mil = 3.812870581168059


    print('mu_j fuelburn = ', mu_j_full['fuelburn'][0])
    print('var_j fuelburn = ', var_j_full['fuelburn'][0])

    # err_mu = np.linalg.norm((mu_j_full['fuelburn'][0] - mu_mc_mil) / mu_mc_mil)
    # err_var = np.linalg.norm((var_j_full['fuelburn'][0,0] - var_mc_mil) / var_mc_mil)
    # print('err_mu = ', err_mu)
    # print('err_var = ', err_var)
    # print('time_elapsed = ', time_elapsed)
    # print('robust fburn = ', mu_j_full['fuelburn'][0] + 2 * np.sqrt(var_j_full['fuelburn'][0]))
    # print('mu_j KS = ', mu_j_full['constraints'][0])
    # print('var_j KS = ', var_j_full['con_failure'][0])
    # print('robust KS = ', mu_j_full['constraints'][0] + 2 * np.sqrt(var_j_full['con_failure'][0]))
    # print('mu_j_lift = ', mu_j_full['constraints'][n_thickness_intersects+1])
    # print('mu_j CM = ', mu_j_full['constraints'][-2]) # [n_thickness_intersects+2:n_thickness_intersects+2+n_CM])
    """
    mu_j_i, var_j_i = eval_objective_and_constraint(sc_sol_dict[key_name], red_colloc_obj)
    print('Reduced')
    print('mu_j fuelburn = ', mu_j_i['fuelburn'][0])
    print('var_j fuelburn = ', var_j_i['fuelburn'][0])
    print('robust fburn = ', mu_j_i['fuelburn'][0] + 2 * np.sqrt(var_j_i['fuelburn'][0]))
    print('mu_j KS = ', mu_j_i['constraints'][0])
    print('var_j KS = ', var_j_i['con_failure'][0])
    print('robust KS = ', mu_j_i['constraints'][0] + 2 * np.sqrt(var_j_i['con_failure'][0]))
    print('mu_j_lift = ', mu_j_i['constraints'][n_thickness_intersects+1])
    print('mu_j CM = ', mu_j_i['constraints'][-2])

    err_mu_f = np.linalg.norm((mu_j_i['fuelburn'][0] - mu_j_full['fuelburn'][0]) / mu_j_full['fuelburn'][0])
    err_var_f = np.linalg.norm((var_j_i['fuelburn'][0,0] -  var_j_full['fuelburn'][0,0]) /  var_j_full['fuelburn'][0,0])
    err_mu_KS = np.linalg.norm((mu_j_i['constraints'][0] - mu_j_full['constraints'][0]) / mu_j_full['constraints'][0])
    err_var_KS = np.linalg.norm((var_j_i['con_failure'][0,0] - var_j_full['con_failure'][0,0]) / var_j_full['con_failure'][0,0])
    err_mu_L = np.linalg.norm((mu_j_i['constraints'][n_thickness_intersects+1] - mu_j_full['constraints'][n_thickness_intersects+1]) / mu_j_full['constraints'][n_thickness_intersects+1])
    err_mu_CM = np.linalg.norm((mu_j_i['constraints'][-2] - mu_j_full['constraints'][-2]) / mu_j_full['constraints'][-2])

    # Print  statements
    print('\nn dir = ', sys.argv[1])
    print('err_mu_f = ', err_mu_f)
    print('err_var_f = ', err_var_f)
    print('err_mu_KS = ', err_mu_KS)
    print('err_var_KS =', err_var_KS)
    print('err_mu_L =', err_mu_L)
    print('err_mu_CM =',  err_mu_CM)
    """
