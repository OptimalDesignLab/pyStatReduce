import sys
import time

# pyStatReduce specific imports
import numpy as np
import chaospy as cp
import copy
from pystatreduce.new_stochastic_collocation import StochasticCollocation2
from pystatreduce.monte_carlo import MonteCarlo
from pystatreduce.quantity_of_interest import QuantityOfInterest
from pystatreduce.dimension_reduction import DimensionReduction
from pystatreduce.stochastic_arnoldi.arnoldi_sample import ArnoldiSampling
import pystatreduce.examples as examples
import pystatreduce.utils as utils

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

dv_dict = {'n_twist_cp' : 3,
           'n_thickness_cp' : 3,
           'n_CM' : 3,
           'n_thickness_intersects' : 10,
           'n_constraints' : 1 + 10 + 1 + 3 + 3,
           'ndv' : 3 + 3 + 2,
           'mesh_dict' : mesh_dict,
           'rv_dict' : rv_dict
            }


mu, std_dev = utils.get_scaneagle_input_rv_statistics(rv_dict)

uq_systemsize = len(mu)
jdist = cp.MvNormal(mu, std_dev)
QoI = examples.oas_scaneagle2.OASScanEagleWrapper2(uq_systemsize,
                                                        dv_dict)

# Set the design variable
dict_val = 'sc_init'
design_point = optimal_vals_dict.sc_sol_dict[dict_val]

QoI.p['oas_scaneagle.wing.thickness_cp'] = design_point['thickness_cp']
QoI.p['oas_scaneagle.wing.twist_cp'] = design_point['twist_cp']
QoI.p['oas_scaneagle.wing.sweep'] = design_point['sweep']
QoI.p['oas_scaneagle.alpha'] = design_point['alpha']
QoI.p.final_setup()

QoI_dict = {'fuelburn' : {'QoI_func' : QoI.eval_QoI,
                          'output_dimensions' : 1
                          },
            }

# Create the Monte Carlo object
start_time1 = time.time()
nsample = int(sys.argv[1])
mc_obj = MonteCarlo(nsample, jdist, QoI_dict) # tjdist: truncated normal distribution
mc_obj.getSamples(jdist)                      #
t1 = time.time()
# Compute the statistical moments using Monte Carlo
mu_j_mc = mc_obj.mean(jdist, of=['fuelburn'])
t2 = time.time()
var_j_mc = mc_obj.variance(jdist, of=['fuelburn'])
t3 = time.time()
print('Monte Carlo samples = ', nsample)
print("mean_mc = ", mu_j_mc['fuelburn'][0])
print("var_mc = ", var_j_mc['fuelburn'][0])
print()

# mean_sc_fuelburn = 5.269295151614887
# var_sc_fuelburn = 0.34932256
# err_mu = abs((mean_sc_fuelburn - mu_j_mc['fuelburn']) / mean_sc_fuelburn)
# err_var = abs((var_sc_fuelburn - var_j_mc['fuelburn']) / var_sc_fuelburn)
# print("err mu = ", err_mu)
# print("err var = ", err_var)

print('\n-------- Timing Results -------\n')
prep_time_mc = t1 - start_time1
mean_time_mc = t2 - t1
var_time_mc = t3 - t2
total_time_mc = prep_time_mc + mean_time_mc + var_time_mc
print("prep_time_mc = ", prep_time_mc)
print("mean_time_mc = ", mean_time_mc)
print("var_time_mc = ", var_time_mc)
print("total_time_mc = ", total_time_mc)
print()
