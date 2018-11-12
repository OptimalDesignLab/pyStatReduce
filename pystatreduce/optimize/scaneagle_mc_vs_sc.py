# This file contains a cost comparison of the Stochastic Collocation method
# against the monte carlo method for the ScanEagle problem.

import sys

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

dv_dict = {'n_twist_cp' : 3,
           'n_thickness_cp' : 3,
           'n_CM' : 3,
           'n_thickness_intersects' : 10,
           'n_constraints' : 1 + 10 + 1 + 3 + 3,
           'ndv' : 3 + 3 + 2,
           'mesh_dict' : mesh_dict,
           'surface_dict_rv' : surface_dict_rv
            }

self.QoI = examples.OASScanEagleWrapper(uq_systemsize, dv_dict,
                                        include_dict_rv=True)
