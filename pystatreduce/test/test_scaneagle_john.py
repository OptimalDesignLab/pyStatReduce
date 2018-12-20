# This file containst tests for john's fork and branch
# https://github.com/johnjasa/OpenAeroStruct/tree/move_surface_vars
# commit hash ee10ee86e0aec273d8e4db9cfe2871426d2e57a8

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

class OASScanEagleTest(unittest.TestCase):

    def test_deterministic_model(self):
        # Check if the quantity of interest is being computed as expected
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

        QoI = examples.OASScanEagleWrapper(uq_systemsize, input_dict, include_dict_rv=True)

        # Check the value at the starting point
        fval = QoI.eval_QoI(mu_orig, np.zeros(uq_systemsize))
        true_val = 5.229858093218218
        err = abs(fval - true_val)
        self.assertTrue(err < 1.e-6)

        # Check if plugging back value also yields the expected results
        QoI.p['oas_scaneagle.wing.twist_cp'] = np.array([2.60830137, 10., 5.])
        QoI.p['oas_scaneagle.wing.thickness_cp'] = np.array([0.001, 0.001, 0.001])
        QoI.p['oas_scaneagle.wing.sweep'] = [18.89098985]
        QoI.p['oas_scaneagle.alpha'] = [2.19244059]
        fval = QoI.eval_QoI(mu_orig, np.zeros(uq_systemsize))
        true_val = 4.735819672292367
        err = abs(fval - true_val)
        self.assertTrue(err < 1.e-6)

    def test_dfuelburn_drv(self):
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

        QoI = examples.OASScanEagleWrapper(uq_systemsize, input_dict, include_dict_rv=True)
        dJdrv = QoI.eval_QoIGradient(mu_orig, np.zeros(uq_systemsize))
        true_val = np.array([-83.76493292024509,
                             74045.31234313066,
                             0.44175879007053753,
                             -7.34403789212763e-13,
                             -2.527193348815028e-13,
                             0.8838194148741767])
        err = abs(dJdrv - true_val) / true_val
        self.assertTrue((err < 1.e-6).all())

    def test_dominant_dir(self):
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

        QoI = examples.OASScanEagleWrapper(uq_systemsize, input_dict, include_dict_rv=True)
        QoI.p['oas_scaneagle.wing.thickness_cp'] = 1.e-3 * np.array([5.5, 5.5, 5.5]) # This setup is according to the one in the scaneagle paper
        QoI.p['oas_scaneagle.wing.twist_cp'] = 2.5*np.ones(3)
        QoI.p.final_setup()

        dominant_space = DimensionReduction(n_arnoldi_sample=uq_systemsize+1,
                                            exact_Hessian=False,
                                            sample_radius=1.e-2)
        dominant_space.getDominantDirections(QoI, jdist, max_eigenmodes=6)
        true_iso_eigenvals = np.array([1.95266202, -1.94415145, 0.57699699, -0.17056458, 0., 0.])
        # We will only check for the eigenvalues because we are using a native
        # library for the eigenvalue factorization. Which means that the Hessenberg
        # matrix which was factorized is in effect being tested with this.
        print('dominant_space.iso_eigenvals = ', dominant_space.iso_eigenvals)
        print('dominant_space.iso_eigenvecs = ')
        print(dominant_space.iso_eigenvecs)
        err = abs(dominant_space.iso_eigenvals - true_iso_eigenvals)
        print('err = ', err)
        self.assertTrue((err < 1.e-6).all())

if __name__ == "__main__":
    unittest.main()
