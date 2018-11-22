import sys
import time

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


# Declare some global variables that will be used across different tests
mean_Ma = 0.071
mean_TSFC = 9.80665 * 8.6e-6
mean_W0 = 10.0
mean_E = 85.e9
mean_G = 25.e9
mean_mrho = 1600

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


class OASScanEagleTest(unittest.TestCase):

    def test_deterministic_model3rv(self):
        uq_systemsize = 3
        mu_orig = np.array([mean_Ma, mean_TSFC, mean_W0])
        std_dev = np.diag([0.005, 0.00607/3600, 0.2])
        jdist = cp.MvNormal(mu_orig, std_dev)

        surface_dict_rv = {'E' : mean_E, # RV
                           'G' : mean_G, # RV
                           'mrho' : mean_mrho, # RV
                          }

        input_dict = {'n_twist_cp' : 3,
                   'n_thickness_cp' : 3,
                   'n_CM' : 3,
                   'n_thickness_intersects' : 10,
                   'n_constraints' : 1 + 10 + 1 + 3 + 3,
                   'ndv' : 3 + 3 + 2,
                   'mesh_dict' : mesh_dict,
                   'surface_dict_rv' : surface_dict_rv
                    }

        QoI = examples.OASScanEagleWrapper(uq_systemsize, input_dict, include_dict_rv=False)

        # Check the value at the nominal point
        fval = QoI.eval_QoI(mu_orig, np.zeros(uq_systemsize))
        true_val = 5.229858093218218
        err = abs(fval - true_val)
        self.assertTrue(err < 1.e-7)

    def test_deterministic_model6rv(self):
        uq_systemsize = 6
        mu_orig = np.array([mean_Ma, mean_TSFC, mean_W0, mean_E, mean_G, mean_mrho])
        std_dev = np.diag([0.005, 0.00607/3600, 0.2, 5.e9, 1.e9, 50])
        jdist = cp.MvNormal(mu_orig, std_dev)

        surface_dict_rv = {'E' : mean_E, # RV
                           'G' : mean_G, # RV
                           'mrho' : mean_mrho, # RV
                          }

        input_dict = {'n_twist_cp' : 3,
                   'n_thickness_cp' : 3,
                   'n_CM' : 3,
                   'n_thickness_intersects' : 10,
                   'n_constraints' : 1 + 10 + 1 + 3 + 3,
                   'ndv' : 3 + 3 + 2,
                   'mesh_dict' : mesh_dict,
                   'surface_dict_rv' : surface_dict_rv
                    }

        QoI = examples.OASScanEagleWrapper(uq_systemsize, input_dict, include_dict_rv=True)

        # Check the value at the starting point
        fval = QoI.eval_QoI(mu_orig, np.zeros(uq_systemsize))
        true_val = 5.229858093218218
        err = abs(fval - true_val)
        self.assertTrue(err < 1.e-6)

        # We will now perturb the surface_dict_rv variable individually and
        # ensure that each one of these actually demonstrate the change as
        # expected. We use a loose tolerance of 1.e-6 for function values,
        # because that's how the thing has been coded up in OpenAeroStruct.
        #
        # TODO: I may have to tighten the OpenAeroStruct tolerance is there are
        #       issues with the dominant_dirs
        #
        mu_pert = copy.copy(mu_orig)
        # 1. Check change in Youngs modulus
        mu_pert[3] += std_dev[3,3]
        fval = QoI.eval_QoI(mu_pert, np.zeros(uq_systemsize))
        true_val = 5.225346542013745
        err = abs(fval - true_val)
        self.assertTrue(err < 1.e-6)
        # 2. Check change in shear modulus
        mu_pert[:] = mu_orig[:]
        mu_pert[4] += std_dev[4,4]
        fval = QoI.eval_QoI(mu_pert, np.zeros(uq_systemsize))
        true_val = 5.229615275307261
        err = abs(fval - true_val)
        self.assertTrue(err < 1.e-6)
        # 3. Check material mass density
        mu_pert[:] = mu_orig[:]
        mu_pert[5] += std_dev[5,5]
        fval = QoI.eval_QoI(mu_pert, np.zeros(uq_systemsize))
        true_val = 5.255241409985575
        err = abs(fval - true_val)
        self.assertTrue(err < 1.e-6)
"""
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

uq_systemsize = len(mu_orig) # There are 6 random variables
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
grad1 = QoI.eval_QoIGradient(mu_orig, pert)
print("grad1 = ", grad1)

# dominant_space = DimensionReduction(n_arnoldi_sample=uq_systemsize+1,
#                                             exact_Hessian=False)
# dominant_space.getDominantDirections(QoI, jdist, max_eigenmodes=1)
"""

"""
fburn1 = QoI.eval_QoI(mu_orig, pert)
print("fburn1 = ", fburn1[0])

# We are now going to preturb on of the values and compare it against run_scaneagle
mu_new = copy.copy(mu_orig)
i = 3
mu_new[i] += std_dev[i,i]
fburn_i = QoI.eval_QoI(mu_new, pert)
print("fburn_i = ", fburn_i[0])
"""
if __name__ == "__main__":
    unittest.main()
