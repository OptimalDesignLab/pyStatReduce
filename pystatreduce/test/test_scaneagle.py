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
mu_orig = np.array([mean_Ma, mean_TSFC, mean_W0, mean_E, mean_G, mean_mrho])
std_dev = np.diag([0.005, 0.00607/3600, 0.2, 5.e9, 1.e9, 50])
# mu_orig = np.array([mean_Ma, mean_TSFC, mean_W0])
# std_dev = np.diag([0.005, 0.00607/3600, 0.2])
jdist = cp.MvNormal(mu_orig, std_dev)

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
surface_dict_rv = {'E' : mean_E, # RV
                   'G' : mean_G, # RV
                   'mrho' : mean_mrho, # RV
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
            }

QoI.p.run_model()

pert = np.zeros(uq_systemsize)
fburn1 = QoI.eval_QoI(mu_orig, pert)
print("fburn1 = ", fburn1[0])

# We are now going to preturb on of the values and compare it against run_scaneagle
mu_new = copy.copy(mu_orig)
i = 3
mu_new[i] += std_dev[i,i]
fburn_i = QoI.eval_QoI(mu_new, pert)
print("fburn_i = ", fburn_i[0])