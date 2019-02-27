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

# Default values
mean_Ma = 0.071
mean_TSFC = 9.80665 * 8.6e-6
mean_W0 = 10.0
mean_E = 85.e9
mean_G = 25.e9
mean_mrho = 1600
mean_R = 1800e3
mean_load_factor = 1.0

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


mu = np.array([mean_Ma, mean_TSFC, mean_W0, mean_E, mean_G, mean_mrho, mean_R, mean_load_factor])
std_dev = np.diag([0.005, 0.00607/3600, 0.2, 5.e9, 1.e9, 50, 100e3, 0.02])
uq_systemsize = len(mu)
jdist = cp.MvNormal(mu, std_dev)
QoI = examples.oas_scaneagle2.OASScanEagleWrapper2(uq_systemsize,
                                                        dv_dict)
QoI.p['oas_scaneagle.wing.thickness_cp'] = 1.e-3 * np.array([5.5, 5.5, 5.5]) # This setup is according to the one in the scaneagle paper
QoI.p['oas_scaneagle.wing.twist_cp'] = 2.5*np.ones(3)
QoI.p.final_setup()

QoI_dict = {'fuelburn' : {'QoI_func' : QoI.eval_QoI,
                          'output_dimensions' : 1
                          },
            }

# Create the Monte Carlo object
start_time1 = time.time()
nsample = 1200
mc_obj = MonteCarlo(nsample, jdist, QoI_dict) # tjdist: truncated normal distribution
mc_obj.getSamples(jdist)                      #
t1 = time.time()
# Compute the statistical moments using Monte Carlo
mu_j_mc = mc_obj.mean(jdist, of=['fuelburn'])
t2 = time.time()
var_j_mc = mc_obj.variance(jdist, of=['fuelburn'])
t3 = time.time()
print("mean_mc = ", mu_j_mc['fuelburn'])
print("var_mc = ", var_j_mc['fuelburn'])
print()

mean_sc_fuelburn = 5.269295151614887
var_sc_fuelburn = 0.34932256

err_mu = abs((mean_sc_fuelburn - mu_j_mc['fuelburn']) / mean_sc_fuelburn)
err_var = abs((var_sc_fuelburn - var_j_mc['fuelburn']) / var_sc_fuelburn)

print("err mu = ", err_mu)
print("err var = ", err_var)

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
