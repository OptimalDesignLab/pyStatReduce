# check_integration_accuracy.py
# This file contains the script that checks for the accuracy of the reduced collocation
# for the objective function fuel burn as a function of the Krylov perturbation
# and number of dominant direction used

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

print("twist = ", QoI.p['oas_scaneagle.wing.geometry.twist'])
print("thickness =", QoI.p['oas_scaneagle.wing.thickness'])
print("sweep = ", QoI.p['oas_scaneagle.wing.sweep'])
print("aoa = ", QoI.p['oas_scaneagle.alpha'])
print()

"""
# Dictionary for the collocation object
QoI_dict = {'fuelburn' : {'QoI_func' : QoI.eval_QoI,
                          'output_dimensions' : 1
                          },
            }

true_mu_fuelburn = 5.29519889     # These two values were computed using full
true_var_fuelburn = 0.21258938 # stochastic collocation, with 3 quadrature
                                   # points in every direction.

sample_radius_arr = [1.e-2, 1.e-3, 1.e-4, 1.e-5, 1.e-6]
mu_j_arr = np.zeros(5)
var_j_arr = np.zeros(5)
start_time_arr = np.zeros(5)
end_time_arr = np.zeros(5)
time_taken_arr = np.zeros(5)

dominant_space_dict = {}
sc_obj_dict = {}
ctr = 0
quadrature_degree = 4
for i in sample_radius_arr:
    str_val1 = 'dominant_space_' + str(i)
    str_val2 = 'sc_obj_' + str(i)
    # print("str_val1 = ", str_val1)
    start_time_arr[ctr] = time.time()
    dominant_space_dict[str_val1] = DimensionReduction(n_arnoldi_sample=uq_systemsize+1,
                                                      exact_Hessian=False,
                                                      sample_radius=i)
    dominant_space_dict[str_val1].getDominantDirections(QoI, jdist, max_eigenmodes=2)
    dominant_dir = dominant_space_dict[str_val1].iso_eigenvecs[:, dominant_space_dict[str_val1].dominant_indices]

    sc_obj_dict[str_val2] = StochasticCollocation2(jdist, quadrature_degree, 'MvNormal', QoI_dict,
                                                   include_derivs=False,
                                                   reduced_collocation=True,
                                                   dominant_dir=dominant_dir)
    sc_obj_dict[str_val2].evaluateQoIs(jdist, include_derivs=False)
    mu_j = sc_obj_dict[str_val2].mean(of=['fuelburn'])
    var_j = sc_obj_dict[str_val2].variance(of=['fuelburn'])
    end_time_arr[ctr] = time.time()
    time_taken_arr[ctr] = end_time_arr[ctr] - start_time_arr[ctr]
    mu_j_arr[ctr] = mu_j['fuelburn'][0]
    var_j_arr[ctr] = var_j['fuelburn'][0]
    ctr += 1
    break

# err_mu_j = abs(mu_j_arr - true_mu_fuelburn)
# err_var_j = abs(var_j_arr - true_var_fuelburn)

print()
print("quadrature degree = ", quadrature_degree)
print('eigenvals = ', dominant_space_dict['dominant_space_0.01'].iso_eigenvals)
print('eigenvacs = ')
print(dominant_space_dict['dominant_space_0.01'].iso_eigenvecs)
# print("mu_j_arr = ", mu_j_arr)
# print("err_mu_j = ", err_mu_j)
# print()
# print("var_j_arr = ", var_j_arr)
# print("err_var_j = ", err_var_j)
# print("time_taken_arr = ", time_taken_arr)
# # Comments include hindsight 2020
# # 1.e-6: In this case there are only 2 dominant eigenvalues
# dominant_space5 = DimensionReduction(n_arnoldi_sample=uq_systemsize+1,
#                                     exact_Hessian=False,
#                                     sample_radius=i)
# dominant_space5.getDominantDirections(QoI, jdist, max_eigenmodes=2)
# dominant_dir = UQObj.dominant_space5.iso_eigenvecs[:, dominant_space5.dominant_indices]
# sc_obj5 = StochasticCollocation2(UQObj.jdist, 3, 'MvNormal', UQObj.QoI_dict,
#                                 include_derivs=False , reduced_collocation=True,
#                                 dominant_dir=dominant_dir)
# sc_obj5.evaluateQoIs(jdist, include_derivs=True)
# mu_j5 = sc_obj5.mean(of=['fuelburn'])
# var_j5 = sc_obj5.variance(of=['fuelburn'])
# print("mean fuelburn = ", mu_j5[''])
"""
