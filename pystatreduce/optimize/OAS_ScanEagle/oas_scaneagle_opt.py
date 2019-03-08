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

class UQScanEagleOpt(object):
    """
    This class is the conduit for linking pyStatReduce and OpenAeroStruct with
    pyOptSparse.
    """
    def __init__(self, rv_dict, rdo_factor=2.0, krylov_pert=1.e-6, max_eigenmodes=2):

        self.rdo_factor = rdo_factor

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

        self.uq_systemsize = len(rv_dict)

        dv_dict = {'n_twist_cp' : 3,
                   'n_thickness_cp' : 3,
                   'n_CM' : 3,
                   'n_thickness_intersects' : 10,
                   'n_constraints' : 1 + 10 + 1 + 3 + 3,
                   'ndv' : 3 + 3 + 2,
                   'mesh_dict' : mesh_dict,
                   'rv_dict' : rv_dict
                    }

        mu, std_dev = self.get_input_rv_statistics(rv_dict)
        self.jdist = cp.MvNormal(mu, std_dev)
        self.QoI = examples.oas_scaneagle2.OASScanEagleWrapper2(self.uq_systemsize,
                                                                dv_dict)
        self.QoI.p['oas_scaneagle.wing.thickness_cp'] = 1.e-3 * np.array([5.5, 5.5, 5.5]) # This setup is according to the one in the scaneagle paper
        self.QoI.p['oas_scaneagle.wing.twist_cp'] = 2.5*np.ones(3)
        self.QoI.p['oas_scaneagle.wing.sweep'] = 20.0
        self.QoI.p['oas_scaneagle.alpha'] = 5.0
        self.QoI.p.final_setup()

        self.dominant_space = DimensionReduction(n_arnoldi_sample=self.uq_systemsize+1,
                                                 exact_Hessian=False,
                                                 sample_radius=krylov_pert)
        self.dominant_space.getDominantDirections(self.QoI, self.jdist, max_eigenmodes=max_eigenmodes)
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
                         'constraints' : {'QoI_func' : self.QoI.eval_AllConstraintQoI,
                                          'output_dimensions' : dv_dict['n_constraints'],
                                          'deriv_dict' : dcon_dict
                                         },
                         'con_failure' : {'QoI_func' : self.QoI.eval_confailureQoI,
                                          'output_dimensions' : 1,
                                          'deriv_dict' : dcon_failure_dict
                                         }
                        }
    def get_input_rv_statistics(self, rv_dict):
        mu, std_dev = utils.get_scaneagle_input_rv_statistics(rv_dict)
        return mu, std_dev

if __name__ == "__main__":
    # Set some of the initial values of the design variables
    init_twist_cp = np.array([2.5, 2.5, 2.5])
    init_thickness_cp = 1.e-3 * np.array([5.5, 5.5, 5.5]) # np.array([0.008, 0.008, 0.008])
    init_sweep = 20.0
    init_alpha = 5.

    start_time = time.time()
    UQObj = UQScanEagleOpt()
