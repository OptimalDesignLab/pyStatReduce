# This file containst tests for john's fork and branch
# https://github.com/johnjasa/OpenAeroStruct/tree/move_surface_vars
# commit hash ee10ee86e0aec273d8e4db9cfe2871426d2e57a8
#
# This file contains tests for the class with 8 random variables.

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

np.set_printoptions(precision=12)

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

class OASScanEagleTest(unittest.TestCase):

    def get_input_rv_statistics(self, rv_dict):
        """
        This utility function is used to select the random variables in a specific
        order.
        """
        mu = np.zeros(len(rv_dict))
        std_dev = np.eye(len(rv_dict))
        i = 0
        for rvs in rv_dict:
            if rvs == 'Mach_number':
                mu[i] = mean_Ma
                std_dev[i,i] = std_dev_Ma
            elif rvs == 'CT':
                mu[i] = mean_TSFC
                std_dev[i,i] = std_dev_TSFC
            elif rvs == 'W0':
                mu[i] = mean_W0
                std_dev[i,i] = std_dev_W0
            elif rvs == 'mrho':
                mu[i] = mean_mrho
                std_dev[i,i] = std_dev_mrho
            elif rvs == 'R':
                mu[i] = mean_R
                std_dev[i,i] = std_dev_R
            elif rvs == 'load_factor':
                mu[i] = mean_load_factor
                std_dev[i,i] = std_dev_load_factor
            elif rvs == 'E':
                mu[i] = mean_E
                std_dev[i,i] = std_dev_E
            elif rvs == 'G':
                mu[i] = mean_G
                std_dev[i,i] = std_dev_G
            i += 1

        return mu, std_dev

    def test_deterministic_model(self):
        # Check if the quantity of interest is being computed as expected
        uq_systemsize = 8
        mu_orig = np.array([mean_Ma, mean_TSFC, mean_W0, mean_E, mean_G, mean_mrho, mean_R, mean_load_factor])
        std_dev = np.diag([0.005, 0.00607/3600, 0.2, 5.e9, 1.e9, 50, 100e3, 0.02])
        jdist = cp.MvNormal(mu_orig, std_dev)

        rv_dict = {'Mach_number' : mean_Ma,
                   'CT' : mean_TSFC,
                   'W0' : mean_W0,
                   'E' : mean_E, # surface RV
                   'G' : mean_G, # surface RV
                   'mrho' : mean_mrho, # surface RV
                   'R' : mean_R,
                   'load_factor' : mean_load_factor,
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

        QoI = examples.oas_scaneagle2.OASScanEagleWrapper2(uq_systemsize, input_dict)

        # Check if plugging back value also yields the expected results
        QoI.p['oas_scaneagle.wing.thickness_cp'] = np.array([0.008, 0.008, 0.008])
        QoI.p['oas_scaneagle.wing.twist_cp'] = np.array([2.5, 2.5, 5.]) # 2.5*np.ones(3)
        QoI.p['oas_scaneagle.wing.sweep'] = 20.0
        QoI.p['oas_scaneagle.alpha'] = 5.0

        # Check the value at the starting point
        fval = QoI.eval_QoI(mu_orig, np.zeros(uq_systemsize))
        true_val = 5.2059024220429615
        err = abs(fval - true_val)
        self.assertTrue(err < 1.e-6)

    def test_dfuelburn_drv(self):
        uq_systemsize = 8
        mu_orig = np.array([mean_Ma, mean_TSFC, mean_W0, mean_E, mean_G, mean_mrho, mean_R, mean_load_factor])
        std_dev = np.diag([0.005, 0.00607/3600, 0.2, 5.e9, 1.e9, 50, 100e3, 0.02])
        jdist = cp.MvNormal(mu_orig, std_dev)

        rv_dict = {'Mach_number' : mean_Ma,
                   'CT' : mean_TSFC,
                   'W0' : mean_W0,
                   'E' : mean_E, # surface RV
                   'G' : mean_G, # surface RV
                   'mrho' : mean_mrho, # surface RV
                   'R' : mean_R,
                   'load_factor' : mean_load_factor,
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

        QoI = examples.oas_scaneagle2.OASScanEagleWrapper2(uq_systemsize, input_dict)
        dJdrv = QoI.eval_QoIGradient(mu_orig, np.zeros(uq_systemsize))

        # print('dJdrv = ', dJdrv)
        true_val = np.array([-86.161737972784,
                             73657.54314740685,
                             0.439735298404,
                             -7.34403789212763e-13,
                             -2.527193348815028e-13,
                             0.879771028302,
                             0.00000345115,
                             6.014451860042])
        err = abs(dJdrv - true_val) / true_val
        self.assertTrue((err < 1.e-6).all())

    def test_variable_update(self):
        uq_systemsize = 8
        mu_orig = np.array([mean_Ma, mean_TSFC, mean_W0, mean_E, mean_G, mean_mrho, mean_R, mean_load_factor])
        std_dev = np.diag([0.005, 0.00607/3600, 0.2, 5.e9, 1.e9, 50, 100e3, 0.02])
        jdist = cp.MvNormal(mu_orig, std_dev)

        rv_dict = {'Mach_number' : mean_Ma,
                   'CT' : mean_TSFC,
                   'W0' : mean_W0,
                   'E' : mean_E, # surface RV
                   'G' : mean_G, # surface RV
                   'mrho' : mean_mrho, # surface RV
                   'R' : mean_R,
                   'load_factor' : mean_load_factor,
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

        QoI = examples.oas_scaneagle2.OASScanEagleWrapper2(uq_systemsize, input_dict)

        mu_pert = mu_orig + np.diagonal(std_dev)
        QoI.update_rv(mu_pert)
        QoI.p.final_setup()
        self.assertEqual(mu_pert[0], QoI.p['Mach_number'])
        self.assertEqual(mu_pert[1], QoI.p['CT'])
        self.assertEqual(mu_pert[2], QoI.p['W0'])
        self.assertEqual(mu_pert[3], QoI.p['E'])
        self.assertEqual(mu_pert[4], QoI.p['G'])
        self.assertEqual(mu_pert[5], QoI.p['mrho'])
        self.assertEqual(mu_pert[6], QoI.p['R'])
        self.assertEqual(mu_pert[7], QoI.p['load_factor'])

    def test_selective_rv(self):
        rv_dict = {'Mach_number' : mean_Ma,
                   'CT' : mean_TSFC,
                   'W0' : mean_W0,
                   'R' : mean_R,
                   'load_factor' : mean_load_factor,
                   # 'E' : mean_E, # surface RV
                   # 'G' : mean_G, # surface RV
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

        uq_systemsize = len(rv_dict)
        mu, std_dev = self.get_input_rv_statistics(rv_dict)
        jdist = cp.MvNormal(mu, std_dev)

        QoI = examples.oas_scaneagle2.OASScanEagleWrapper2(uq_systemsize, input_dict)
        mu_pert = mu + np.diagonal(std_dev)

        QoI.update_rv(mu_pert)
        QoI.p.final_setup()
        self.assertEqual(mu_pert[0], QoI.p['Mach_number'])
        self.assertEqual(mu_pert[1], QoI.p['CT'])
        self.assertEqual(mu_pert[2], QoI.p['W0'])
        self.assertEqual(mu_pert[3], QoI.p['R'])
        self.assertEqual(mu_pert[4], QoI.p['load_factor'])
        self.assertEqual(mu_pert[5], QoI.p['mrho'])



    """
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
    """

if __name__ == "__main__":
    unittest.main()
