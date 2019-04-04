# check_optimal_uq_vals.py
# This file will contain the scripts to check how the optimial UQ values compare
# with each other for the different implementations. These implementations
# include
# 1. MFMC solution
# 2. Full collocation
# 3. Reduced collocation
# This is for the 6 random variable case
#
import sys
import time
import pprint as pp

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

# Declare some global default values
mean_Ma = 0.071
mean_TSFC = 9.80665 * 8.6e-6
mean_W0 = 10.0
mean_E = 85.e9
mean_G = 25.e9
mean_mrho = 1600

mu = np.array([mean_Ma, mean_TSFC, mean_W0, mean_E, mean_G, mean_mrho])
std_dev = np.diag([0.005, 0.00607/3600, 0.2, 5.e9, 1.e9, 50])

class UQScanEagleOpt(object):
    """
    This class is the conduit for linking pyStatReduce and OpenAeroStruct with
    pyOptSparse.
    """
    def __init__(self, uq_systemsize, all_rv=False):

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

        self.jdist = cp.MvNormal(mu, std_dev)
        self.QoI = examples.OASScanEagleWrapper(uq_systemsize, dv_dict,
                                                include_dict_rv=all_rv)
        dfuelburn_dict = {'dv' : {'dQoI_func' : self.QoI.eval_ObjGradient_dv,
                                  'output_dimensions' : dv_dict['ndv'],
                                  }
                         }
        dcon_dict = {'dv' : {'dQoI_func' : self.QoI.eval_ConGradient_dv,
                             'output_dimensions' : dv_dict['ndv']
                            }
                    }
        dcon_failure_dict = {'dv' : {'dQoI_func' : self.QoI.eval_ConFailureGradient_dv,
                                     'output_dimensions' : dv_dict['ndv'],
                                    }
                            }
        self.QoI_dict = {'fuelburn' : {'QoI_func' : self.QoI.eval_QoI,
                                       'output_dimensions' : 1,
                                       'deriv_dict' : dfuelburn_dict
                                      },
                         # 'constraints' : {'QoI_func' : self.QoI.eval_AllConstraintQoI,
                         #                  'output_dimensions' : dv_dict['n_constraints'],
                         #                  'deriv_dict' : dcon_dict
                         #                 },
                         # 'con_failure' : {'QoI_func' : self.QoI.eval_confailureQoI,
                         #                  'output_dimensions' : 1,
                         #                  'deriv_dict' : dcon_failure_dict
                         #                 }
                        }

def eval_uq_fuelburn(dv_dict):
    UQObj.QoI.p['oas_scaneagle.wing.twist_cp'] = dv_dict['twist_cp']
    UQObj.QoI.p['oas_scaneagle.wing.thickness_cp'] = dv_dict['thickness_cp']
    UQObj.QoI.p['oas_scaneagle.wing.sweep'] = dv_dict['sweep']
    UQObj.QoI.p['oas_scaneagle.alpha'] = dv_dict['alpha']

    # print("wing.twist_cp = ", UQObj.QoI.p['oas_scaneagle.wing.twist_cp'])
    # print("wing.thickness_cp = ", UQObj.QoI.p['oas_scaneagle.wing.thickness_cp'])
    # print("wing.sweep = ", UQObj.QoI.p['oas_scaneagle.wing.sweep'])
    # print("alpha = ", UQObj.QoI.p['oas_scaneagle.alpha'])

    # Compute statistical metrics
    sc_obj.evaluateQoIs(UQObj.jdist)
    # print('fvals = ')
    # print(sc_obj.QoI_dict['fuelburn']['fvals'])
    mu_j = sc_obj.mean(of=['fuelburn'])
    var_j = sc_obj.variance(of=['fuelburn'])

    return mu_j, var_j

if __name__ == "__main__":
    # Step 1: Instantiate all objects needed for test
    uq_systemsize = 6
    UQObj = UQScanEagleOpt(uq_systemsize, all_rv=True)
    UQObj.QoI.p['oas_scaneagle.wing.thickness_cp'] = 1.e-3 * np.array([5.5, 5.5, 5.5])
    UQObj.QoI.p['oas_scaneagle.wing.twist_cp'] = 2.5*np.ones(3)
    UQObj.QoI.p.final_setup()

    # Dominant dominant
    dominant_space = DimensionReduction(n_arnoldi_sample=uq_systemsize+1,
                                             exact_Hessian=False,
                                             sample_radius=1.e-2)
    dominant_space.getDominantDirections(UQObj.QoI, UQObj.jdist, max_eigenmodes=4)
    # # Full collocation
    # sc_obj = StochasticCollocation2(UQObj.jdist, 3, 'MvNormal', UQObj.QoI_dict,
    #                                 include_derivs=False)
    # sc_obj.evaluateQoIs(UQObj.jdist, include_derivs=False)

    # REduced collocation
    dominant_dir = dominant_space.iso_eigenvecs[:, dominant_space.dominant_indices]
    sc_obj = StochasticCollocation2(UQObj.jdist, 3, 'MvNormal', UQObj.QoI_dict,
                                    include_derivs=False, reduced_collocation=True,
                                    dominant_dir=dominant_dir)
    sc_obj.evaluateQoIs(UQObj.jdist, include_derivs=False)
    # print('fvals = ')
    # print(sc_obj.QoI_dict['fuelburn']['fvals'])
    # Print initial value
    init_mu_j = sc_obj.mean(of=['fuelburn'])
    init_var_j = sc_obj.variance(of=['fuelburn'])
    # print("Mean fuelburn = ", init_mu_j['fuelburn'][0])
    # print("Variance fuelburn = ", init_var_j['fuelburn'][0])
    # print()

    mfmc_sol = {'twist_cp' : np.array([-1.59, 0.34, 4.50]),
                'thickness_cp' : 1.e-3 * np.array([1.0, 1.04, 3.41]),
                'sweep' : 18.73,
                'alpha' : 5.54,
                }

    full_SC_sol = {'twist_cp' : np.array([2.586845, 10, 5.0]),
                   'thickness_cp' : 1.e-3 * np.array([1.0, 1.0, 1.034094]),
                   'sweep' : 18.89138,
                   'alpha' : 2.176393,
                  }

    SC_sol1D = {'twist_cp' : np.array([2.592991, 10, 5.0]),
                'thickness_cp' : 1.e-3 * np.array([1.0, 1.0, 1.0]),
                'sweep' : 18.89514,
                'alpha' : 2.203623,
                }

    SC_sol2D = {'twist_cp' : np.array([2.587855, 10, 5.0]),
                'thickness_cp' : 1.e-3 * np.array([1.0, 1.0, 1.042386]),
                'sweep' : 18.88911,
                'alpha' : 2.166430,
                }

    SC_sol3D = {'twist_cp' : np.array([2.588266, 10, 5.0]),
                'thickness_cp' : 1.e-3 * np.array([1.0, 1.0, 1.029025]),
                'sweep' : 18.89295,
                'alpha' : 2.184638,
                }

    SC_sol4D = {'twist_cp' : np.array([2.588816, 10, 5.0]),
                'thickness_cp' : 1.e-3 * np.array([1.0, 1.0, 1.028649]),
                'sweep' : 18.89294,
                'alpha' : 2.185064,
                }


    mu_j_mfmc, var_j_mfmc = eval_uq_fuelburn(mfmc_sol)
    print("# MFMC solution")
    print("Mean fuelburn = ", mu_j_mfmc['fuelburn'][0])
    print("Variance fuelburn = ", var_j_mfmc['fuelburn'][0])
    print("robust objective = ", mu_j_mfmc['fuelburn'][0] + 2*np.sqrt(var_j_mfmc['fuelburn'][0]))
    print()

    mu_j_full, var_j_full = eval_uq_fuelburn(full_SC_sol)
    print("# Full collocation")
    print("Mean fuelburn = ", mu_j_full['fuelburn'][0])
    print("Variance fuelburn = ", var_j_full['fuelburn'][0])
    print("robust objective = ", mu_j_full['fuelburn'][0] + 2*np.sqrt(var_j_full['fuelburn'][0]))

    mu_j_1D, var_j_1D = eval_uq_fuelburn(SC_sol1D)
    print("# 1D")
    print("Mean fuelburn = ", mu_j_1D['fuelburn'][0])
    print("Variance fuelburn = ", var_j_1D['fuelburn'][0])
    print("robust objective = ", mu_j_1D['fuelburn'][0] + 2*np.sqrt(var_j_1D['fuelburn'][0]))

    mu_j_2D, var_j_2D = eval_uq_fuelburn(SC_sol2D)
    print("# 2D")
    print("Mean fuelburn = ", mu_j_2D['fuelburn'][0])
    print("Variance fuelburn = ", var_j_2D['fuelburn'][0])
    print("robust objective = ", mu_j_2D['fuelburn'][0] + 2*np.sqrt(var_j_2D['fuelburn'][0]))

    mu_j_3D, var_j_3D = eval_uq_fuelburn(SC_sol3D)
    print("# 3D")
    print("Mean fuelburn = ", mu_j_3D['fuelburn'][0])
    print("Variance fuelburn = ", var_j_3D['fuelburn'][0])
    print("robust objective = ", mu_j_3D['fuelburn'][0] + 2*np.sqrt(var_j_3D['fuelburn'][0]))

    mu_j_4D, var_j_4D = eval_uq_fuelburn(SC_sol4D)
    print("# 4D")
    print("Mean fuelburn = ", mu_j_4D['fuelburn'][0])
    print("Variance fuelburn = ", var_j_4D['fuelburn'][0])
    print("robust objective = ", mu_j_4D['fuelburn'][0] + 2*np.sqrt(var_j_4D['fuelburn'][0]))
