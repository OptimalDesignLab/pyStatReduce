import sys
import time

# pyStatReduce specific imports
import numpy as np
import chaospy as cp
import copy
from pystatreduce.new_stochastic_collocation import StochasticCollocation2
from pystatreduce.quantity_of_interest import QuantityOfInterest
from pystatreduce.dimension_reduction import DimensionReduction
from pystatreduce.active_subspace import ActiveSubspace
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

class UQScanEagleOpt(object):
    """
    This class is the conduit for linking pyStatReduce and OpenAeroStruct with
    pyOptSparse.

    **Arguments**
    * `rv_dict` : The random variable dictionary which specifies the random variable,
                  its mean and standard deviation.
    * `design_point` : dictionary of the design variable values.
    * `rdo_factor` : robust design optimization factor, its a multiplier that gets
                     multiplied to the wuantity of interest. Currently a single
                     value is used across all QoIs.
    * `krylov_pert` : finite difference perturbation for the modified Arnoldi method.
    * `active_subspace` : Boolean for either using active subspace or Arnoldi
                          dimension reduction method.
    * `max_eigenmodes` : Maximum number of dominant directons expected from the user.
                         This is the maximum number dominant directions that will be used
                         in the subsequent stochastic collocation object.
    * `n_as_samples` : Number of monte carlo samples used by the active subspace
                       method. This is argument is only utilized if we are using
                       the active subspace method.
    """
    def __init__(self, rv_dict, design_point, rdo_factor=2.0, krylov_pert=1.e-1,
                 active_subspace=False, max_eigenmodes=2, n_as_samples=1000):

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
        self.QoI.p['oas_scaneagle.wing.thickness_cp'] = design_point['thickness_cp']
        self.QoI.p['oas_scaneagle.wing.twist_cp'] = design_point['twist_cp']
        self.QoI.p['oas_scaneagle.wing.sweep'] = design_point['sweep']
        self.QoI.p['oas_scaneagle.alpha'] = design_point['alpha']
        self.QoI.p.final_setup()

        # Figure out which dimension reduction technique to use
        start_time = time.time()
        if active_subspace == False:
            self.dominant_space = DimensionReduction(n_arnoldi_sample=self.uq_systemsize+1,
                                                     exact_Hessian=False,
                                                     sample_radius=krylov_pert)
            self.dominant_space.getDominantDirections(self.QoI, self.jdist, max_eigenmodes=max_eigenmodes)
        else:
            self.dominant_space = ActiveSubspace(self.QoI,
                                                 n_dominant_dimensions=max_eigenmodes,
                                                 n_monte_carlo_samples=n_as_samples,
                                                 read_rv_samples=False,
                                                 use_svd=True,
                                                 use_iso_transformation=True)
            self.dominant_space.getDominantDirections(self.QoI, self.jdist)
            # Reset the design point
            self.QoI.p['oas_scaneagle.wing.thickness_cp'] = design_point['thickness_cp']
            self.QoI.p['oas_scaneagle.wing.twist_cp'] = design_point['twist_cp']
            self.QoI.p['oas_scaneagle.wing.sweep'] = design_point['sweep']
            self.QoI.p['oas_scaneagle.alpha'] = design_point['alpha']
            self.QoI.p.final_setup()
            # Reset the random variables
            self.QoI.update_rv(mu)
        time_elapsed = time.time() - start_time
        print('time_elapsed =', time_elapsed)

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
    # Default mean values
    mean_Ma = 0.071
    mean_TSFC = 9.80665 * 8.6e-6
    mean_W0 = 10.0
    mean_E = 85.e9
    mean_G = 25.e9
    mean_mrho = 1600
    mean_R = 1800
    mean_load_factor = 1.0
    mean_altitude = 4.57
    # Default standard values
    std_dev_Ma = 0.005
    std_dev_TSFC = 0.00607/3600
    std_dev_W0 = 0.2
    std_dev_mrho = 50
    std_dev_R = 500
    std_dev_load_factor = 0.1
    std_dev_E = 5.e9
    std_dev_G = 1.e9
    std_dev_altitude = 0.5

    # Random variable
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
                'mrho' : {'mean' : mean_mrho,
                         'std_dev' : std_dev_mrho},
                'altitude' :{'mean' : mean_altitude,
                             'std_dev' : std_dev_altitude},
               }


    UQObj = UQScanEagleOpt(rv_dict)
    print("Design variables")
    print('thickness_cp = ', UQObj.QoI.p['oas_scaneagle.wing.thickness_cp'])
    print('twist_cp = ', UQObj.QoI.p['oas_scaneagle.wing.twist_cp'])
    print('sweep = ', UQObj.QoI.p['oas_scaneagle.wing.sweep'])
    print('alpha = ', UQObj.QoI.p['oas_scaneagle.alpha'])
