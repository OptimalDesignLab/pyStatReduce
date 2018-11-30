# This file contains a cost comparison of the Stochastic Collocation method
# against the monte carlo method for the ScanEagle problem.

import sys
import time

# pyStatReduce specific imports
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

uq_systemsize = 6 # There are 6 random variables

# Default mean values of the random variables
mean_Ma = 0.071
mean_TSFC = 9.80665 * 8.6e-6
mean_W0 = 10.0
mean_E = 85.e9
mean_G = 25.e9
mean_mrho = 1600
mu = np.array([mean_Ma, mean_TSFC, mean_W0, mean_E, mean_G, mean_mrho])
std_dev = np.diag([0.005, 0.00607/3600, 0.2, 5.e9, 1.e9, 50])
jdist = cp.MvNormal(mu, std_dev)

# Setup truncated normal distribution
td1 = cp.Truncnorm(lo=mu[0]-2*std_dev[0,0], up=mu[0]+2*std_dev[0,0], mu=mu[0], sigma=std_dev[0,0])
td2 = cp.Truncnorm(lo=mu[1]-2*std_dev[1,1], up=mu[1]+2*std_dev[1,1], mu=mu[1], sigma=std_dev[1,1])
td3 = cp.Truncnorm(lo=mu[2]-2*std_dev[2,2], up=mu[2]+2*std_dev[2,2], mu=mu[2], sigma=std_dev[2,2])
td4 = cp.Truncnorm(lo=mu[3]-2*std_dev[3,3], up=mu[3]+2*std_dev[3,3], mu=mu[3], sigma=std_dev[3,3])
td5 = cp.Truncnorm(lo=mu[4]-2*std_dev[4,4], up=mu[4]+2*std_dev[4,4], mu=mu[4], sigma=std_dev[4,4])
td6 = cp.Truncnorm(lo=mu[5]-2*std_dev[5,5], up=mu[5]+2*std_dev[5,5], mu=mu[5], sigma=std_dev[5,5])
tjdist = cp.J(td1, td2, td3, td4, td5, td6)

ny_arr = [61, 21, 5] # High to low fidelity
nx_arr = [7, 3, 2]   #
i = 1
num_y = ny_arr[i] # Medium fidelity model
num_x = nx_arr[i]  #
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

QoI = examples.OASScanEagleWrapper(uq_systemsize, dv_dict, include_dict_rv=True)
QoI.p['oas_scaneagle.wing.thickness_cp'] = 1.e-3 * np.array([5.5, 5.5, 5.5]) # This setup is according to the one in the scaneagle paper
QoI.p['oas_scaneagle.wing.twist_cp'] = 2.5*np.ones(3)
QoI.p.final_setup()

# Dictionary for the collocation object
QoI_dict = {'fuelburn' : {'QoI_func' : QoI.eval_QoI,
                          'output_dimensions' : 1
                          },
            # 'constraints' : {'QoI_func' : QoI.eval_AllConstraintQoI,
            #                  'output_dimensions' : dv_dict['n_constraints'],
            #                  # 'deriv_dict' : dcon_dict
            #                 },
            # 'con_failure' : {'QoI_func' : QoI.eval_confailureQoI,
            #                  'output_dimensions' : 1,
            #                  # 'deriv_dict' : dcon_failure_dict
            #                 }
            }

print("wing.twist_cp = ", QoI.p['oas_scaneagle.wing.twist_cp'])
print("wing.thickness_cp = ", QoI.p['oas_scaneagle.wing.thickness_cp'])
print("wing.sweep = ", QoI.p['oas_scaneagle.wing.sweep'])
print("alpha = ", QoI.p['oas_scaneagle.alpha'])

# Create the Monte Carlo object
start_time1 = time.time()
nsample = 1# 75
mc_obj = MonteCarlo(nsample, tjdist, QoI_dict) # tjdist: truncated normal distribution
mc_obj.getSamples(tjdist)                      #
t1 = time.time()
# Compute the statistical moments using Monte Carlo
mu_j_mc = mc_obj.mean(jdist, of=['fuelburn'])
t2 = time.time()
var_j_mc = mc_obj.variance(jdist, of=['fuelburn'])
t3 = time.time()
# print("mc_obj fvals = ")
# print(mc_obj.QoI_dict['fuelburn']['fvals'])
print("mean_mc = ", mu_j_mc['fuelburn'])
print("var_mc = ", var_j_mc['fuelburn'])
print()

start_time2 = time.time()
# Create the Stochastic Collocation object (Full collocation)
sc_obj = StochasticCollocation2(jdist, 3, 'MvNormal', QoI_dict)
sc_obj.evaluateQoIs(jdist, include_derivs=False)
# Compute statistical moments using stochastic collocation
t4 = time.time()
mu_j_sc = sc_obj.mean(of=['fuelburn'])
t5 = time.time()
var_j_sc = sc_obj.variance(of=['fuelburn'])
t6 = time.time()
# print("sc_obj fvals = ")
# print(sc_obj.QoI_dict['fuelburn']['fvals'])
print("mean_sc = ", mu_j_sc['fuelburn'])
print("var_sc = ", var_j_sc['fuelburn'])

err_mu = abs((mu_j_sc['fuelburn'] - mu_j_mc['fuelburn']) / mu_j_mc['fuelburn'])
err_var = abs((var_j_sc['fuelburn'] - var_j_mc['fuelburn']) / var_j_mc['fuelburn'])

print("err mu = ", err_mu)
print(" err var = ", err_var)


# Lets look at the timings
# Monte Carlo method
prep_time_mc = t1 - start_time1
mean_time_mc = t2 - t1
var_time_mc = t3 - t2
total_time_mc = prep_time_mc + mean_time_mc + var_time_mc
# stochastic collocation method
prep_time_sc = t4 - start_time2
mean_time_sc = t5 - t4
var_time_sc = t6 - t5
total_time_sc = prep_time_sc + mean_time_sc + var_time_sc
# Print the time elapsed
print("prep_time_mc = ", prep_time_mc)
print("mean_time_mc = ", mean_time_mc)
print("var_time_mc = ", var_time_mc)
print("total_time_mc = ", total_time_mc)
print()


print("prep_time_sc = ", prep_time_sc)
print("mean_time_sc = ", mean_time_sc)
print("var_time_sc = ", var_time_sc)
print("total_time_sc = ", total_time_sc)

print("reduction factor = ", total_time_mc/total_time_sc)
