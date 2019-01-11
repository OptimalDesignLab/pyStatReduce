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
from pystatreduce.examples.oas_scaneagle_proto import OASScanEagleWrapper, Fuelburn, StressConstraint, LiftConstraint, MomentConstraint
# import pystatreduce.examples as examples

np.set_printoptions(precision=8)
np.set_printoptions(linewidth=150, suppress=True)

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

uq_systemsize = 6
mu_orig = np.array([mean_Ma, mean_TSFC, mean_W0, mean_E, mean_G, mean_mrho])
std_dev = np.diag([0.005, 0.00607/3600, 0.2, 5.e9, 1.e9, 50])
jdist = cp.MvNormal(mu_orig, std_dev)

rv_dict = {'Mach_number' : mean_Ma,
           'CT' : mean_TSFC,
           'W0' : mean_W0,
           'E' : mean_E, # surface RV
           'G' : mean_G, # surface RV
           'mrho' : mean_mrho, # surface RV
            }

input_dict = {'n_twist_cp' : 3,
           'n_thickness_cp' : 3,
           'n_CM' : 3,
           'n_thickness_intersects' : 10,
           'n_constraints' : 1 + 10 + 1 + 3 + 3,
           'ndv' : 3 + 3 + 2,
           'mesh_dict' : mesh_dict,
           'rv_dict' : rv_dict
            }

# Create all the objects that will be tested in this file
oas_obj = OASScanEagleWrapper(uq_systemsize, input_dict, include_dict_rv=True)
obj_QoI = Fuelburn(uq_systemsize, oas_obj)
failure_QoI = StressConstraint(uq_systemsize, oas_obj)
lift_con_QoI = LiftConstraint(uq_systemsize, oas_obj)
moment_con_QoI = MomentConstraint(uq_systemsize, oas_obj)

class OASScanEagleProtoTest(unittest.TestCase):
    def test_OASScanEagleWrapper_functions(self):
        mu_new = mu_orig + np.diagonal(std_dev)
        oas_obj.update_rv(mu_new)
        self.assertEqual(mu_new[0], oas_obj.p['Mach_number'])
        self.assertEqual(mu_new[1], oas_obj.p['CT'])
        self.assertEqual(mu_new[2], oas_obj.p['W0'])
        self.assertEqual(mu_new[3], oas_obj.p['E'])
        self.assertEqual(mu_new[4], oas_obj.p['G'])
        self.assertEqual(mu_new[5], oas_obj.p['mrho'])
        # Revert to the original values
        oas_obj.update_rv(mu_orig)

    def test_Fuelburn_class(self):
        # Check variables are being updated correctly
        mu_new = mu_orig + np.diagonal(std_dev)
        obj_QoI.update_rv(mu_new)
        self.assertEqual(mu_new[0], obj_QoI.p['Mach_number'])
        self.assertEqual(mu_new[1], obj_QoI.p['CT'])
        self.assertEqual(mu_new[2], obj_QoI.p['W0'])
        self.assertEqual(mu_new[3], obj_QoI.p['E'])
        self.assertEqual(mu_new[4], obj_QoI.p['G'])
        self.assertEqual(mu_new[5], obj_QoI.p['mrho'])

        # Check QoI value
        fval = obj_QoI.eval_QoI(mu_orig, np.zeros(uq_systemsize))
        true_val = 5.229858093218218
        err = abs(fval - true_val)
        self.assertTrue(err < 1.e-6)

    def test_StressConstraint_class(self):
        # Check variables are being updated correctly
        mu_new = mu_orig + np.diagonal(std_dev)
        failure_QoI.update_rv(mu_new)
        self.assertEqual(mu_new[0], failure_QoI.p['Mach_number'])
        self.assertEqual(mu_new[1], failure_QoI.p['CT'])
        self.assertEqual(mu_new[2], failure_QoI.p['W0'])
        self.assertEqual(mu_new[3], failure_QoI.p['E'])
        self.assertEqual(mu_new[4], failure_QoI.p['G'])
        self.assertEqual(mu_new[5], failure_QoI.p['mrho'])

        # Check QoI value
        fval = failure_QoI.eval_QoI(mu_orig, np.zeros(uq_systemsize))
        true_val = -0.80021068203315
        err = abs(fval - true_val)
        self.assertTrue(err < 1.e-6)

    def test_LiftConstraint_class(self):
        # Check variables are being updated correctly
        mu_new = mu_orig + np.diagonal(std_dev)
        lift_con_QoI.update_rv(mu_new)
        self.assertEqual(mu_new[0], lift_con_QoI.p['Mach_number'])
        self.assertEqual(mu_new[1], lift_con_QoI.p['CT'])
        self.assertEqual(mu_new[2], lift_con_QoI.p['W0'])
        self.assertEqual(mu_new[3], lift_con_QoI.p['E'])
        self.assertEqual(mu_new[4], lift_con_QoI.p['G'])
        self.assertEqual(mu_new[5], lift_con_QoI.p['mrho'])

        # Check QoI value
        fval = lift_con_QoI.eval_QoI(mu_orig, np.zeros(uq_systemsize))
        true_val = -0.0031524169619576714
        err = abs(fval - true_val)
        self.assertTrue(err < 1.e-6)

    def test_MomentConstraint_class(self):
        # Check variables are being updated correctly
        mu_new = mu_orig + np.diagonal(std_dev)
        moment_con_QoI.update_rv(mu_new)
        self.assertEqual(mu_new[0], moment_con_QoI.p['Mach_number'])
        self.assertEqual(mu_new[1], moment_con_QoI.p['CT'])
        self.assertEqual(mu_new[2], moment_con_QoI.p['W0'])
        self.assertEqual(mu_new[3], moment_con_QoI.p['E'])
        self.assertEqual(mu_new[4], moment_con_QoI.p['G'])
        self.assertEqual(mu_new[5], moment_con_QoI.p['mrho'])

        # Check QoI value
        fval = moment_con_QoI.eval_QoI(mu_orig, np.zeros(uq_systemsize))
        true_val = 0.01036558702945927
        err = abs(fval[1] - true_val)
        self.assertTrue(err < 1.e-6)

"""
# Check original values
orig_fuel_burn = obj_QoI.eval_QoI(mu_orig, np.zeros(uq_systemsize))
orig_aggregated_stress = failure_QoI.eval_QoI(mu_orig, np.zeros(uq_systemsize))
orig_lift_fail_stress = lift_con_QoI.eval_QoI(mu_orig, np.zeros(uq_systemsize))
orig_moment_con_QoI = moment_con_QoI.eval_QoI(mu_orig, np.zeros(uq_systemsize))
print('orig_fuel_burn = ', orig_fuel_burn)
print('orig_aggregated_stress = ', orig_aggregated_stress)
print('orig_lift_fail_stress = ', orig_lift_fail_stress)
print('orig_moment_con_QoI = ', orig_moment_con_QoI)
print()

# Check if all the components work as expected
mu_new = mu_orig + std_dev
new_fuel_burn = obj_QoI.eval_QoI(mu_orig, np.diagonal(std_dev))
new_aggregated_stress = failure_QoI.eval_QoI(mu_orig, np.diagonal(std_dev))
new_lift_fail_stress = lift_con_QoI.eval_QoI(mu_orig, np.diagonal(std_dev))
new_moment_con_QoI = moment_con_QoI.eval_QoI(mu_orig, np.diagonal(std_dev))
print('new_fuel_burn = ', new_fuel_burn)
print('new_aggregated_stress = ', new_aggregated_stress)
print('new_lift_fail_stress = ', new_lift_fail_stress)
print('new_moment_con_QoI = ', new_moment_con_QoI)


# Check if the gradients are being computed correctly
dnew_fuel_burn = obj_QoI.eval_QoIGradient(mu_orig, np.diagonal(std_dev))
dnew_aggregated_stress = failure_QoI.eval_QoIGradient(mu_orig, np.diagonal(std_dev))
dnew_lift_fail_stress = lift_con_QoI.eval_QoIGradient(mu_orig, np.diagonal(std_dev))
dnew_moment_con_QoI = moment_con_QoI.eval_QoIGradient(mu_orig, np.diagonal(std_dev))
print('dnew_fuel_burn = ', dnew_fuel_burn)
print('dnew_aggregated_stress = ', dnew_aggregated_stress)
print('dnew_lift_fail_stress = ', dnew_lift_fail_stress)
print('dnew_moment_con_QoI = ', dnew_moment_con_QoI)
"""

if __name__ == "__main__":
    unittest.main()
