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

num_y = 21 # Medium fidelity model
num_x = 3  #
mesh_dict = {'num_y' : num_y,
             'num_x' : num_x,
             'wing_type' : 'rect',
             'symmetry' : True,
             'span_cos_spacing' : 0.5,
             'span' : 3.11,
             'root_chord' : 0.3,
             }
surface_dict_rv = {'E' : 85.e9, # RV
                   'G' : 25.e9, # RV
                   'mrho' : 1.6e3, # RV
                  }
dv_dict = {'n_twist_cp' : 3,
           'n_thickness_cp' : 3,
           'n_CM' : 3,
           'n_thickness_intersects' : 10,
           'n_constraints' : 1 + 10 + 1 + 3 + 3,
           'ndv' : 3 + 3 + 2,
           'mesh_dict' : mesh_dict,
           'surface_dict_rv' : surface_dict_rv
            }

QoI = examples.OASScanEagleWrapper(uq_systemsize, dv_dict, include_dict_rv=True)

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

# # Create the Monte Carlo object
# start_time1 = time.time()
# nsample = 1000000
# mc_obj = MonteCarlo(nsample, jdist, QoI_dict)
# mc_obj.getSamples(jdist)
# t1 = time.time()
# # Compute the statistical moments using Monte Carlo
# mu_j_mc = mc_obj.mean(jdist, of=['fuelburn'])
# t2 = time.time()
# var_j_mc = mc_obj.variance(jdist, of=['fuelburn'])
# t3 = time.time()
# # print("mc_obj fvals = ")
# # print(mc_obj.QoI_dict['fuelburn']['fvals'])
# print("mean_mc = ", mu_j_mc['fuelburn'])
# print("var_mc = ", var_j_mc['fuelburn'])
# print()

start_time2 = time.time()
# Create the Stochastic Collocation object (Full collocation)
sc_obj = StochasticCollocation2(jdist, 2, 'MvNormal', QoI_dict)
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

# err_mu = abs((mu_j_sc['fuelburn'] - mu_j_mc['fuelburn']) / mu_j_mc['fuelburn'])
# err_var = abs((var_j_sc['fuelburn'] - var_j_mc['fuelburn']) / var_j_mc['fuelburn'])

# print("err mu = ", err_mu)
# print(" err var = ", err_var)

# # Lets look at the timings
# # Monte Carlo method
# prep_time_mc = t1 - start_time1
# mean_time_mc = t2 - t1
# var_time_mc = t3 - t2
# total_time_mc = prep_time_mc + mean_time_mc + var_time_mc
# stochastic collocation method
prep_time_sc = t4 - start_time2
mean_time_sc = t5 - t4
var_time_sc = t6 - t5
total_time_sc = prep_time_sc + mean_time_sc + var_time_sc

# Print the time elapsed
# print("prep_time_mc = ", prep_time_mc)
# print("mean_time_mc = ", mean_time_mc)
# print("var_time_mc = ", var_time_mc)
# print("total_time_mc = ", total_time_mc)
# print()
print("prep_time_sc = ", prep_time_sc)
print("mean_time_sc = ", mean_time_sc)
print("var_time_sc = ", var_time_sc)
print("total_time_sc = ", total_time_sc)

print("reduction factor = ", total_time_mc/total_time_sc)
