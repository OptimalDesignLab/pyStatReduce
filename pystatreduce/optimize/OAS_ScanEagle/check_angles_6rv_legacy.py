################################################################################
# The following file contains the script that checks the angles between the
# different quantities of interest in the ScanEagle problem. The angles are
# checked w.r.t the objective function, and the values are printed in radians.
################################################################################
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
import pystatreduce.utils as utils

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

# Create the base openaerostruct problem wrapper that will be used by the
# different quantity of interests
oas_obj = OASScanEagleWrapper(uq_systemsize, input_dict, include_dict_rv=True)
# Create the QoI objects
obj_QoI = Fuelburn(uq_systemsize, oas_obj)
failure_QoI = StressConstraint(uq_systemsize, oas_obj)
lift_con_QoI = LiftConstraint(uq_systemsize, oas_obj)
moment_con_QoI = MomentConstraint(uq_systemsize, oas_obj)

# Create the dimension reduction objects for all of the different quantity of interest
# Get the dominant directions of the different QoIs here
dominant_space_obj = DimensionReduction(n_arnoldi_sample=uq_systemsize+1,
                                        exact_Hessian=False, sample_radius=1.e-2)
dominant_space_obj.getDominantDirections(obj_QoI, jdist, max_eigenmodes=4)
dominant_space_failure = DimensionReduction(n_arnoldi_sample=uq_systemsize+1,
                                            exact_Hessian=False,
                                            sample_radius=1.e-2)
dominant_space_failure.getDominantDirections(failure_QoI, jdist, max_eigenmodes=4)
dominant_space_liftcon = DimensionReduction(n_arnoldi_sample=uq_systemsize+1,
                                            exact_Hessian=False,
                                            sample_radius=1.e-2)
dominant_space_liftcon.getDominantDirections(lift_con_QoI, jdist, max_eigenmodes=4)
dominant_space_CM = DimensionReduction(n_arnoldi_sample=uq_systemsize+1,
                                       exact_Hessian=False, sample_radius=1.e-2)
dominant_space_CM.getDominantDirections(moment_con_QoI, jdist, max_eigenmodes=4)

# Print all the eigenvalues
print('Wf iso_eigenvals = ', dominant_space_obj.iso_eigenvals)
print('KS iso_eigenvals = ', dominant_space_failure.iso_eigenvals)
print('Lift_Con iso_eigenvals = ', dominant_space_liftcon.iso_eigenvals)
print('dominant_space_CM iso_eigenvals = ', dominant_space_CM.iso_eigenvals)

# Collect all the directions
dominant_dir_obj = dominant_space_obj.dominant_dir
dominant_dir_KS_fail = dominant_space_failure.dominant_dir
dominant_dir_L_equal_w = dominant_space_liftcon.dominant_dir
dominant_dir_CM = dominant_space_CM.dominant_dir
# print('dominant_dir_obj =\n', dominant_dir_obj)
# print('dominant_dir_KS_fail =\n', dominant_dir_KS_fail)
# print('dominant_dir_L_equal_w =\n', dominant_dir_L_equal_w)
# print('dominant_dir_CM =\n', dominant_dir_CM)

print(dominant_dir_obj - dominant_dir_L_equal_w)
print(dominant_dir_obj - dominant_dir_CM)

# We now compare the angles w.r.t the objective functions
# angles_KSfail = utils.compute_subspace_angles(dominant_dir_obj, dominant_dir_KS_fail)
angles_liftcon = utils.compute_subspace_angles(dominant_dir_obj, dominant_dir_L_equal_w)
angles_CM = utils.compute_subspace_angles(dominant_dir_obj, dominant_dir_CM)

# Print thes angles
print()
# print('angles_KSfail = ', angles_KSfail)
print('angles_liftcon = ', angles_liftcon)
print('angles_CM = ', angles_CM)
