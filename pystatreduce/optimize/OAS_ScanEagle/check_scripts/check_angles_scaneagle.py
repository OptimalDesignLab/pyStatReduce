################################################################################
# check_angles_scaneagle.py
# This file will contain the script that contains the code that compares the
# angles between the different subspaces spanned by the dominant directions of
# of the different QoIs
################################################################################

import sys
import time

# pyStatReduce specific imports
import numpy as np
import chaospy as cp
import copy
from pystatreduce.new_stochastic_collocation import StochasticCollocation2
from pystatreduce.quantity_of_interest import QuantityOfInterest
from pystatreduce.dimension_reduction import DimensionReduction
from pystatreduce.stochastic_arnoldi.arnoldi_sample import ArnoldiSampling
from pystatreduce.examples.oas_scaneagle_proto import OASScanEagleWrapper, \
    Fuelburn, StressConstraint, LiftConstraint, MomentConstraint
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

# Default mean values
mean_Ma = 0.08
mean_TSFC = 9.80665 * 8.6e-6 * 3600
mean_W0 = 10 # 10.0
mean_E = 85.e9
mean_G = 25.e9
mean_mrho = 1600
mean_R = 1800
mean_load_factor = 1.0
mean_altitude = 4.57
# Default standard values
std_dev_Ma = 0.005 # 0.015
std_dev_TSFC = 0.00607 # /3600
std_dev_W0 = 1
std_dev_mrho = 50
std_dev_R = 300 # 500
std_dev_load_factor = 0.3
std_dev_E = 5.e9
std_dev_G = 1.e9
std_dev_altitude = 0.1

# initial values of the design variables
init_twist_cp = np.array([2.5, 2.5, 5.0])
init_thickness_cp = np.array([0.008, 0.008, 0.008])
init_sweep = 20.0
init_alpha = 5.

def update_dv(oas_object, design_pt):
    oas_object.p['oas_scaneagle.wing.twist_cp'] = design_pt['twist_cp'] #  init_twist_cp
    oas_object.p['oas_scaneagle.wing.thickness_cp'] = design_pt['thickness_cp']
    oas_object.p['oas_scaneagle.wing.sweep'] = design_pt['sweep']
    oas_object.p['oas_scaneagle.alpha'] = design_pt['alpha']
    oas_object.p.final_setup()

dv_val_dict = { 'init_design' : { 'twist_cp' : np.array([2.5, 2.5, 5.0]),
                                  'thickness_cp' : np.array([0.008, 0.008, 0.008]),
                                  'sweep' : 20.,
                                  'alpha' : 5.},

                '7rv_1e_1_2_2' :  {'twist_cp' : np.array([4.825586, 10, 5]),
                                   'thickness_cp' : 1.e-3 * np.ones(3),
                                   'sweep' : 17.59059178,
                                   'alpha' : -0.09239151},
              }

# Random variable dictionary
rv_dict = { 'Mach_number' : {'mean' : mean_Ma,
                             'std_dev' : std_dev_Ma},
            'CT' : {'mean' : mean_TSFC,
                    'std_dev' : std_dev_TSFC},
            'altitude' : {'mean' : mean_altitude,
                          'std_dev' : std_dev_altitude},
            'W0' : {'mean' : mean_W0,
                    'std_dev' : std_dev_W0},
            'R' : {'mean' : mean_R,
                   'std_dev' : std_dev_R},
            'load_factor' : {'mean' : mean_load_factor,
                             'std_dev' : std_dev_load_factor},
            'mrho' : {'mean' : mean_mrho,
                     'std_dev' : std_dev_mrho},
           }

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

input_dict = {'n_twist_cp' : 3,
           'n_thickness_cp' : 3,
           'n_CM' : 3,
           'n_thickness_intersects' : 10,
           'n_constraints' : 1 + 10 + 1 + 3 + 3,
           'ndv' : 3 + 3 + 2,
           'mesh_dict' : mesh_dict,
           'rv_dict' : rv_dict
            }

uq_systemsize = len(rv_dict)
mu_orig, std_dev = utils.get_scaneagle_input_rv_statistics(rv_dict)
jdist = cp.MvNormal(mu_orig, std_dev)

# Create the base openaerostruct problem wrapper that will be used by the
# different quantity of interests
oas_obj = OASScanEagleWrapper(uq_systemsize, input_dict)
# Create the QoI objects
obj_QoI = Fuelburn(uq_systemsize, oas_obj)
failure_QoI = StressConstraint(uq_systemsize, oas_obj)
lift_con_QoI = LiftConstraint(uq_systemsize, oas_obj)
moment_con_QoI = MomentConstraint(uq_systemsize, oas_obj)

# Set the design point
design_pt = 'init_design'
update_dv(oas_obj, dv_val_dict[design_pt])

# Create the dimension reduction objects for all of the different quantity of interest
# Get the dominant directions of the different QoIs here
dominant_space_obj = DimensionReduction(n_arnoldi_sample=uq_systemsize+1,
                                        exact_Hessian=False, sample_radius=1.e-1)
dominant_space_obj.getDominantDirections(obj_QoI, jdist, max_eigenmodes=4)
dominant_space_failure = DimensionReduction(n_arnoldi_sample=uq_systemsize+1,
                                            exact_Hessian=False,
                                            sample_radius=1.e-1)
dominant_space_failure.getDominantDirections(failure_QoI, jdist, max_eigenmodes=4)
dominant_space_liftcon = DimensionReduction(n_arnoldi_sample=uq_systemsize+1,
                                            exact_Hessian=False,
                                            sample_radius=1.e-1)
dominant_space_liftcon.getDominantDirections(lift_con_QoI, jdist, max_eigenmodes=4)
dominant_space_CM = DimensionReduction(n_arnoldi_sample=uq_systemsize+1,
                                       exact_Hessian=False, sample_radius=1.e-1)
dominant_space_CM.getDominantDirections(moment_con_QoI, jdist, max_eigenmodes=4)

# Print all the eigenvalues
print('Wf iso_eigenvals = ', dominant_space_obj.iso_eigenvals)
print('KS iso_eigenvals = ', dominant_space_failure.iso_eigenvals)
print('Lift_Con iso_eigenvals = ', dominant_space_liftcon.iso_eigenvals)
print('iso_eigenvals = ', dominant_space_CM.iso_eigenvals)

# Collect all the directions
dominant_dir_obj = np.copy(dominant_space_obj.dominant_dir)
dominant_dir_KS_fail = np.copy(dominant_space_failure.dominant_dir)
dominant_dir_L_equal_w = np.copy(dominant_space_liftcon.dominant_dir)
dominant_dir_CM = np.copy(dominant_space_CM.dominant_dir)

print('\ndominant_dir_obj =\n', dominant_dir_obj)
# print('dominant_dir_KS_fail =\n', dominant_dir_KS_fail)
# print('dominant_dir_L_equal_w =\n', dominant_dir_L_equal_w)
# print('dominant_dir_CM =\n', dominant_dir_CM)

# Copy the initial eigenmodes
orig_evals_fburn = np.copy(dominant_space_obj.iso_eigenvals)
orig_evals_KS_fail = np.copy(dominant_space_failure.iso_eigenvals)
orig_evals_liftcon = np.copy(dominant_space_liftcon.iso_eigenvals)
orig_evals_CM = np.copy(dominant_space_CM.iso_eigenvals)
orig_evecs_fburn = np.copy(dominant_space_obj.iso_eigenvecs)
orig_evecs_KS_fail = np.copy(dominant_space_failure.iso_eigenvecs)
orig_evecs_liftcon = np.copy(dominant_space_liftcon.iso_eigenvecs)
orig_evecs_CM = np.copy(dominant_space_CM.iso_eigenvecs)

# We now compare the angles w.r.t the objective functions
# angles_KSfail = utils.compute_subspace_angles(dominant_dir_obj[:,0], dominant_dir_KS_fail)
angles_liftcon = utils.compute_subspace_angles(dominant_dir_obj, dominant_dir_L_equal_w)
angles_CM = utils.compute_subspace_angles(dominant_dir_obj, dominant_dir_CM)

# Print thes angles
print()
# print('angles_KSfail = ', angles_KSfail)
print('angles_liftcon = ', angles_liftcon)
print('angles_CM = ', angles_CM)

# Update the design variables to the new design point
update_dv(oas_obj, dv_val_dict['7rv_1e_1_2_2'])
new_dominant_space_obj = DimensionReduction(n_arnoldi_sample=uq_systemsize+1,
                                        exact_Hessian=False, sample_radius=1.e-1)
new_dominant_space_obj.getDominantDirections(obj_QoI, jdist, max_eigenmodes=4)
new_dominant_space_failure = DimensionReduction(n_arnoldi_sample=uq_systemsize+1,
                                            exact_Hessian=False,
                                            sample_radius=1.e-1)
new_dominant_space_failure.getDominantDirections(failure_QoI, jdist, max_eigenmodes=4)
new_dominant_space_liftcon = DimensionReduction(n_arnoldi_sample=uq_systemsize+1,
                                            exact_Hessian=False,
                                            sample_radius=1.e-1)
new_dominant_space_liftcon.getDominantDirections(lift_con_QoI, jdist, max_eigenmodes=4)
new_dominant_space_CM = DimensionReduction(n_arnoldi_sample=uq_systemsize+1,
                                       exact_Hessian=False, sample_radius=1.e-1)
new_dominant_space_CM.getDominantDirections(moment_con_QoI, jdist, max_eigenmodes=4)

# Collect all the directions
new_dominant_dir_obj = new_dominant_space_obj.dominant_dir
new_dominant_dir_KS_fail = new_dominant_space_failure.dominant_dir
new_dominant_dir_L_equal_w = new_dominant_space_liftcon.dominant_dir
new_dominant_dir_CM = new_dominant_space_CM.dominant_dir

print('#---------------------------#')
print('\nnew iso_eigenvals fburn = ', new_dominant_space_obj.iso_eigenvals)
print('\nnew_dominant_dir_obj =\n', new_dominant_dir_obj)
# print('new_dominant_dir_KS_fail =\n', new_dominant_dir_KS_fail)
# print('new_dominant_dir_L_equal_w =\n', new_dominant_dir_L_equal_w)
# print('new_dominant_dir_CM =\n', new_dominant_dir_CM)

angles_obj = utils.compute_subspace_angles(dominant_dir_obj, new_dominant_dir_obj)
print('angles_obj = ', angles_obj)
