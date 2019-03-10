# ScanEagle_check_krylov_pert
# This file will check the the accuracy of the Krylov perturbation using the
# following methodology. At the starting point of the optimization, we will
# compute all the principle directions of the QoI, in this case fuelburn, and
# then use it to compare against the full tensor product collocation of higher
# order.
#
# This file is for the class with 8 random variables
import sys
import time
import pprint as pp

# pyStatReduce specific imports
import unittest
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

np.set_printoptions(precision=10)

# Default mean values
mean_Ma = 0.071
mean_TSFC = 9.80665 * 8.6e-6
mean_W0 = 10.0
mean_E = 85.e9
mean_G = 25.e9
mean_mrho = 1600
mean_R = 1800e3
mean_load_factor = 1.0
# Default standard values
std_dev_Ma = 0.005
std_dev_TSFC = 0.00607/3600
std_dev_W0 = 0.2
std_dev_mrho = 50
std_dev_R = 500.e3
std_dev_load_factor = 0.1
std_dev_E = 5.e9
std_dev_G = 1.e9

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

# Random variable dictionary
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
            # 'E' : {'mean' : mean_E,
            #        'std_dev' : std_dev_E},
            # 'G' : {'mean' : mean_G,
            #        'std_dev' : std_dev_G},
            'mrho' : {'mean' : mean_mrho,
                     'std_dev' : std_dev_mrho},
           }

uq_systemsize = len(rv_dict)
mu_orig, std_dev = utils.get_scaneagle_input_rv_statistics(rv_dict)
jdist = cp.MvNormal(mu_orig, std_dev)

input_dict = {'n_twist_cp' : 3,
           'n_thickness_cp' : 3,
           'n_CM' : 3,
           'n_thickness_intersects' : 10,
           'n_constraints' : 1 + 10 + 1 + 3 + 3,
           'ndv' : 3 + 3 + 2,
           'mesh_dict' : mesh_dict,
           'rv_dict' : rv_dict
            }

QoI = examples.oas_scaneagle2.OASScanEagleWrapper2(uq_systemsize, input_dict)
dfuelburn_dict = {'dv' : {'dQoI_func' : QoI.eval_ObjGradient_dv,
                          'output_dimensions' : input_dict['ndv'],
                          }
                 }
dcon_dict = {'dv' : {'dQoI_func' : QoI.eval_ConGradient_dv,
                     'output_dimensions' : input_dict['ndv']
                    }
            }
dcon_failure_dict = {'dv' : {'dQoI_func' : QoI.eval_ConFailureGradient_dv,
                             'output_dimensions' : input_dict['ndv'],
                            }
                    }
QoI_dict = {'fuelburn' : {'QoI_func' : QoI.eval_QoI,
                           'output_dimensions' : 1,
                           'deriv_dict' : dfuelburn_dict
                          },

            }

# Check if plugging back value also yields the expected results
QoI.p['oas_scaneagle.wing.thickness_cp'] = np.array([0.008, 0.008, 0.008])
QoI.p['oas_scaneagle.wing.twist_cp'] = np.array([2.5, 2.5, 5.])
QoI.p['oas_scaneagle.wing.sweep'] = 20.0
QoI.p['oas_scaneagle.alpha'] = 5.0

sample_radii = [1.e-1, 1.e-2, 1.e-3, 1.e-4, 1.e-5, 1.e-6]

if sys.argv[1] == 'full':
    # Create the stochastic collocation object
    sc_obj_full = StochasticCollocation2(jdist, 5, 'MvNormal', QoI_dict,
                                    include_derivs=False, reduced_collocation=False)
    sc_obj_full.evaluateQoIs(jdist, include_derivs=False)
    mu_j_full = sc_obj_full.mean(of=['fuelburn'])
    var_j_full = sc_obj_full.variance(of=['fuelburn'])

    print('mu_j_full = ', mu_j_full['fuelburn'][0])
    print('var_j_full = ', var_j_full['fuelburn'][0,0])

elif sys.argv[1] == 'reduced':
    mu_j_full = 5.341619712754059
    var_j_full = 3.786547863834123

    # Create dimension reduction based on system arguments
    dominant_space = DimensionReduction(n_arnoldi_sample=uq_systemsize+1,
                                             exact_Hessian=False,
                                             sample_radius=sample_radii[int(sys.argv[2])])
    dominant_space.getDominantDirections(QoI, jdist, max_eigenmodes=int(sys.argv[3]))
    # Create a stochastic collocation object
    sc_obj = StochasticCollocation2(jdist, 2, 'MvNormal', QoI_dict,
                                    include_derivs=False, reduced_collocation=True,
                                    dominant_dir=dominant_space.dominant_dir)
    sc_obj.evaluateQoIs(jdist, include_derivs=False)
    mu_j = sc_obj.mean(of=['fuelburn'])
    var_j = sc_obj.variance(of=['fuelburn'])
    mu_j_val = mu_j['fuelburn'][0]
    var_j_val = var_j['fuelburn'][0,0]

    print('\nSample radius = ', sample_radii[int(sys.argv[2])], ', max_eigenmodes = ', sys.argv[3])
    print('mu_j = ', mu_j['fuelburn'][0])
    print('var_j = ', var_j['fuelburn'][0,0])

    print('err mu_j = ', abs((mu_j_full - mu_j_val) / mu_j_full))
    print('err var_j = ', abs( (var_j_full - var_j_val) / var_j_full ))
elif sys.argv[1] == 'pick':
    mu_j_arr = np.zeros([len(sample_radii),6])
    var_j_arr = np.zeros([len(sample_radii),6])
    itr_row = 0 # row iterator
    itr_col = 0 # column iterator
    # Read the output file into mu_j_arr
    fname = 'fout'
    with open(fname) as f:
        line = f.readline()
        while line:
            if itr_col == 6:
                itr_row += 1
                itr_col = 0

            if line[0] == 'm':
                mu_j_arr[itr_row, itr_col] = float(line[7:-1])
                line = f.readline()
            elif line[0] == 'v':
                var_j_arr[itr_row, itr_col] = float(line[8:-1])
                itr_col += 1
                line = f.readline()
            else:
                line = f.readline()

    f.close()

    # Compute mu_j and var_j erroors
    mu_j_full = 5.3416197127  # Make sure to update these when something changes
    var_j_full = 3.7865478638 #

    err_mu_j = abs( (mu_j_arr - mu_j_full) / mu_j_full)
    err_var_j = abs( (var_j_arr - var_j_full ) / var_j_full)
    print('err_mu_j = \n', err_mu_j)
    print('err_var_j = \n', err_var_j)

    # Find the index of the lowest value in every column of the err
    min_mu_ind = np.argmin(err_mu_j, axis=0)
    min_var_ind = np.argmin(err_var_j, axis=0)
    print('min_mu_ind = \n', min_mu_ind)
    print('min_var_ind = \n', min_var_ind)

    min_mu = np.amin(err_mu_j, axis=0)
    min_var = np.amin(err_var_j, axis = 0)
    print('min_mu = ', min_mu)
    print('min_var = ', min_var)
