################################################################################
# pyopt_uq_scaneagle2.py
# This file performs a robust design optimizarion on a Boeing ScanEagle aircraft
# with 6 random variables. The random variables are Ma, TSFC, W0, E, G, mrho.
# This file can be run in 3 ways
#
# 1. Reduced optimization:
#              run as `python pyopt_uq_scaneagle_6rv.py reduced`
#    This runs the optimization using reduced collocation. Currently the number
#    of dominant directions are set to 2. This can be changed to anywhere between
#    1 & 6 by setting the keyword argument in the function call
#    `getDominantDirections(..., max_eigenmodes=5)`. This assumes that the user
#    wants to set the number of dominant directions to 5
#
# 2. debug:
#              run as `python pyopt_uq_scaneagle_6rv.py debug`
#    This only creates a UQScanEagleOpt object. Its useful for printing out values
#
# 3. Full collocation
#              run as `python pyopt_uq_scaneagle_6rv.py full_collocation`
#    This performs RDO using a full 6 dimensional tensor product grid. It takes
#    a long time to run this optimization (Last time I ran on an i7-3820, it took
#    13 hours)
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
import pystatreduce.examples as examples

#pyoptsparse sepecific imports
from scipy import sparse
import argparse
import pyoptsparse # from pyoptsparse import Optimization, OPT, SNOPT

# Import the OpenMDAO
from openmdao.api import IndepVarComp, Problem, Group, NewtonSolver, \
    ScipyIterativeSolver, LinearBlockGS, NonlinearBlockGS, \
    DirectSolver, LinearBlockGS, PetscKSP, SqliteRecorder, ScipyOptimizeDriver

from openaerostruct.geometry.utils import generate_mesh
from openaerostruct.geometry.geometry_group import Geometry
from openaerostruct.aerodynamics.aero_groups import AeroPoint

# Default values
mean_Ma = 0.071
mean_TSFC = 9.80665 * 8.6e-6
mean_W0 = 10.0
mean_E = 85.e9
mean_G = 25.e9
mean_mrho = 1600

class UQScanEagleOpt(object):
    """
    This class is the conduit for linking pyStatReduce and OpenAeroStruct with
    pyOptSparse.
    """
    def __init__(self, uq_systemsize, all_rv=False):
        self.rdo_factor = 2.0

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


        mu = np.array([mean_Ma, mean_TSFC, mean_W0, mean_E, mean_G, mean_mrho])
        std_dev = np.diag([0.005, 0.00607/3600, 0.2, 5.e9, 1.e9, 50])
        self.jdist = cp.MvNormal(mu, std_dev)
        self.QoI = examples.OASScanEagleWrapper(uq_systemsize, dv_dict)

        # This setup is according to the one in the scaneagle paper
        self.QoI.p['oas_scaneagle.wing.thickness_cp'] = 1.e-3 * np.array([5.5, 5.5, 5.5])
        self.QoI.p['oas_scaneagle.wing.twist_cp'] = 2.5*np.ones(3)
        self.QoI.p.final_setup()

        # Compute the dominant directions
        self.dominant_space = DimensionReduction(n_arnoldi_sample=uq_systemsize+1,
                                                 exact_Hessian=False,
                                                 sample_radius=1.e-2)
        self.dominant_space.getDominantDirections(self.QoI, self.jdist, max_eigenmodes=2)


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

def objfunc_uq(xdict):
    """
    Objective funtion supplied to pyOptSparse for RDO.
    """
    rdo_factor = UQObj.rdo_factor
    UQObj.QoI.p['oas_scaneagle.wing.twist_cp'] = xdict['twist_cp']
    UQObj.QoI.p['oas_scaneagle.wing.thickness_cp'] = xdict['thickness_cp']
    UQObj.QoI.p['oas_scaneagle.wing.sweep'] = xdict['sweep']
    UQObj.QoI.p['oas_scaneagle.alpha'] = xdict['alpha']
    funcs = {}

    # Compute statistical metrics
    sc_obj.evaluateQoIs(UQObj.jdist)
    mu_j = sc_obj.mean(of=['fuelburn', 'constraints'])
    var_j = sc_obj.variance(of=['fuelburn', 'con_failure'])

    # The RDO Objective function is
    funcs['obj'] = mu_j['fuelburn'] + rdo_factor * np.sqrt(var_j['fuelburn'])

    # The RDO Constraint function is
    n_thickness_intersects = UQObj.QoI.p['oas_scaneagle.AS_point_0.wing_perf.thickness_intersects'].size
    n_CM = UQObj.QoI.p['oas_scaneagle.AS_point_0.CM'].size
    funcs['con_failure'] = mu_j['constraints'][0] + rdo_factor * np.sqrt(var_j['con_failure'][0,0])
    funcs['con_thickness_intersects'] = mu_j['constraints'][1:n_thickness_intersects+1]
    funcs['con_L_equals_W'] = mu_j['constraints'][n_thickness_intersects+1]
    funcs['con_CM'] = mu_j['constraints'][n_thickness_intersects+2:n_thickness_intersects+2+n_CM]
    funcs['con_twist_cp'] = mu_j['constraints'][n_thickness_intersects+2+n_CM:]

    fail = False
    return funcs, fail

def sens_uq(xdict, funcs):
    """
    Sensitivity function provided to pyOptSparse for RDO.
    """
    rdo_factor = UQObj.rdo_factor
    UQObj.QoI.p['oas_scaneagle.wing.twist_cp'] = xdict['twist_cp']
    UQObj.QoI.p['oas_scaneagle.wing.thickness_cp'] = xdict['thickness_cp']
    UQObj.QoI.p['oas_scaneagle.wing.sweep'] = xdict['sweep']
    UQObj.QoI.p['oas_scaneagle.alpha'] = xdict['alpha']

    # Compute the statistical metrics
    sc_obj.evaluateQoIs(UQObj.jdist, include_derivs=True)
    dmu_js = sc_obj.dmean(of=['fuelburn', 'constraints'], wrt=['dv'])
    dstd_dev_js = sc_obj.dStdDev(of=['fuelburn', 'con_failure'], wrt=['dv'])

    # Get some of the intermediate variables
    n_twist_cp = UQObj.QoI.input_dict['n_twist_cp']
    n_cp = n_twist_cp + UQObj.QoI.input_dict['n_thickness_cp']
    n_CM = UQObj.QoI.input_dict['n_CM']
    n_thickness_intersects = UQObj.QoI.p['oas_scaneagle.AS_point_0.wing_perf.thickness_intersects'].size

    # Populate the dictionary
    funcsSens = {}
    dmu_j = dmu_js['fuelburn']['dv']
    dstd_dev_j = dstd_dev_js['fuelburn']['dv']
    funcsSens['obj', 'twist_cp'] = dmu_j[0,0:n_twist_cp] + rdo_factor * dstd_dev_j[0,0:n_twist_cp]
    funcsSens['obj', 'thickness_cp'] = dmu_j[0,n_twist_cp:n_cp] + rdo_factor * dstd_dev_j[0,n_twist_cp:n_cp]
    funcsSens['obj', 'sweep'] = dmu_j[0,n_cp:n_cp+1] + rdo_factor * dstd_dev_j[0,n_cp:n_cp+1]
    funcsSens['obj', 'alpha'] = dmu_j[0,n_cp+1:n_cp+2] + rdo_factor * dstd_dev_j[0,n_cp+1:n_cp+2]

    dmu_con = dmu_js['constraints']['dv']
    dstd_dev_con = dstd_dev_js['con_failure']['dv']
    funcsSens['con_failure', 'twist_cp'] = dmu_con[0,0:n_twist_cp] + rdo_factor * dstd_dev_con[0,0:n_twist_cp]
    funcsSens['con_failure', 'thickness_cp'] = dmu_con[0,n_twist_cp:n_cp] + rdo_factor * dstd_dev_con[0,n_twist_cp:n_cp]
    funcsSens['con_failure', 'sweep'] = dmu_con[0,n_cp] + rdo_factor * dstd_dev_con[0,n_cp]
    funcsSens['con_failure', 'alpha'] = dmu_con[0,n_cp+1] + rdo_factor * dstd_dev_con[0,n_cp+1]

    funcsSens['con_thickness_intersects', 'twist_cp'] = dmu_con[1:n_thickness_intersects+1,0:n_twist_cp]
    funcsSens['con_thickness_intersects', 'thickness_cp'] = dmu_con[1:n_thickness_intersects+1,n_twist_cp:n_cp]
    funcsSens['con_thickness_intersects', 'sweep'] = dmu_con[1:n_thickness_intersects+1,n_cp:n_cp+1]
    funcsSens['con_thickness_intersects', 'alpha'] = dmu_con[1:n_thickness_intersects+1,n_cp+1:]

    funcsSens['con_L_equals_W', 'twist_cp'] = dmu_con[n_thickness_intersects+1,0:n_twist_cp]
    funcsSens['con_L_equals_W', 'thickness_cp'] = dmu_con[n_thickness_intersects+1,n_twist_cp:n_cp]
    funcsSens['con_L_equals_W', 'sweep'] = dmu_con[n_thickness_intersects+1,n_cp]
    funcsSens['con_L_equals_W', 'alpha'] = dmu_con[n_thickness_intersects+1,n_cp+1]

    idx = n_thickness_intersects + 2
    funcsSens['con_CM', 'twist_cp'] = dmu_con[idx:idx+n_CM,0:n_twist_cp]
    funcsSens['con_CM', 'thickness_cp'] = dmu_con[idx:idx+n_CM,n_twist_cp:n_cp]
    funcsSens['con_CM', 'sweep'] = dmu_con[idx:idx+n_CM,n_cp:n_cp+1]
    funcsSens['con_CM', 'alpha'] = dmu_con[idx:idx+n_CM,n_cp+1:]

    idx = n_thickness_intersects + 2 + n_CM
    funcsSens['con_twist_cp', 'twist_cp'] = dmu_con[idx:,0:n_twist_cp]

    fail = False
    return funcsSens, fail

if __name__ == "__main__":

    # Set some of the initial values of the design variables
    init_twist_cp = np.array([2.5, 2.5, 2.5])
    init_thickness_cp = 1.e-3 * np.array([5.5, 5.5, 5.5]) # np.array([0.008, 0.008, 0.008])
    init_sweep = 20.0
    init_alpha = 5.

    ndv = 3 + 3 + 1 + 1 # number of design variabels

    start_time = time.time()

    if sys.argv[1] == "reduced":
        start_time = time.time()
        uq_systemsize = 6
        UQObj = UQScanEagleOpt(uq_systemsize, all_rv=True)
        xi = np.zeros(uq_systemsize)

        # Get some information on the total number of constraints
        n_thickness_intersects = UQObj.QoI.p['oas_scaneagle.AS_point_0.wing_perf.thickness_intersects'].size
        n_CM = 3
        n_constraints = 1 + n_thickness_intersects  + 1 + n_CM + 3

        # Create the stochastic collocation object based on the dominant directions
        dominant_dir = UQObj.dominant_space.dominant_dir
        sc_obj = StochasticCollocation2(UQObj.jdist, 3, 'MvNormal', UQObj.QoI_dict,
                                        include_derivs=True , reduced_collocation=True,
                                        dominant_dir=dominant_dir)
        sc_obj.evaluateQoIs(UQObj.jdist, include_derivs=True)

        # Set up optimization in pyOptSparse
        optProb = pyoptsparse.Optimization('UQ_OASScanEagle', objfunc_uq)
        n_twist_cp = UQObj.QoI.input_dict['n_twist_cp']
        n_thickness_cp = UQObj.QoI.input_dict['n_thickness_cp']
        optProb.addVarGroup('twist_cp', n_twist_cp, 'c', lower=-5., upper=10, value=init_twist_cp)
        optProb.addVarGroup('thickness_cp', n_thickness_cp, 'c', lower=0.001, upper=0.01, scale=1.e3, value=init_thickness_cp)
        optProb.addVar('sweep', lower=10., upper=30., value=init_sweep)
        # optProb.addVar('alpha', lower=-10., upper=10.)
        optProb.addVar('alpha', lower=0., upper=10., value=init_alpha)

        # Constraints
        optProb.addConGroup('con_failure', 1, upper=0.)
        optProb.addConGroup('con_thickness_intersects', n_thickness_intersects,
                            upper=0., wrt=['thickness_cp'])
        optProb.addConGroup('con_L_equals_W', 1, lower=0., upper=0.)
        optProb.addConGroup('con_CM', n_CM, lower=-0.001, upper=0.001)
        optProb.addConGroup('con_twist_cp', 3, lower=np.array([-1e20, -1e20, 5.]),
                            upper=np.array([1e20, 1e20, 5.]), wrt=['twist_cp'])

        # Objective
        optProb.addObj('obj')
        opt = pyoptsparse.SNOPT(optOptions = {'Major feasibility tolerance' : 1e-9})
        sol = opt(optProb, sens=sens_uq)
        sol = opt(optProb)
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(sol)
        print(sol.fStar)
        print()
        print("twist = ", UQObj.QoI.p['oas_scaneagle.wing.geometry.twist'])
        print("thickness =", UQObj.QoI.p['oas_scaneagle.wing.thickness'])
        print("sweep = ", UQObj.QoI.p['oas_scaneagle.wing.sweep'])
        print("aoa = ", UQObj.QoI.p['oas_scaneagle.alpha'])
        print()
        print('time elapsed = ', elapsed_time)

    elif sys.argv[1] == "debug":
        uq_systemsize = 6
        UQObj = UQScanEagleOpt(uq_systemsize, all_rv=True)

        # Compute gradient
        mu = np.array([mean_Ma, mean_TSFC, mean_W0, mean_E, mean_G, mean_mrho])
        grad = UQObj.QoI.eval_QoIGradient(mu, np.zeros(uq_systemsize))
        print('\ngrad =', grad)
        # print('grad[4] =', grad[4])
        print("eigenvals = ", UQObj.dominant_space.iso_eigenvals)
        print('eigenvecs = \n', UQObj.dominant_space.iso_eigenvecs)

    elif sys.argv[1] == "full_collocation":
        uq_systemsize = 6
        UQObj = UQScanEagleOpt(uq_systemsize, all_rv=True)
        xi = np.zeros(uq_systemsize)

        # Get some information on the total number of constraints
        n_thickness_intersects = UQObj.QoI.p['oas_scaneagle.AS_point_0.wing_perf.thickness_intersects'].size
        n_CM = 3
        n_constraints = 1 + n_thickness_intersects  + 1 + n_CM + 3

        # Full collocation
        sc_obj = StochasticCollocation2(UQObj.jdist, 3, 'MvNormal', UQObj.QoI_dict, include_derivs=True)
        sc_obj.evaluateQoIs(UQObj.jdist, include_derivs=True)

        optProb = pyoptsparse.Optimization('UQ_OASScanEagle', objfunc_uq)
        n_twist_cp = UQObj.QoI.input_dict['n_twist_cp']
        n_thickness_cp = UQObj.QoI.input_dict['n_thickness_cp']
        optProb.addVarGroup('twist_cp', n_twist_cp, 'c', lower=-5., upper=10, value=init_twist_cp)
        optProb.addVarGroup('thickness_cp', n_thickness_cp, 'c', lower=0.001, upper=0.01, scale=1.e3, value=init_thickness_cp)
        optProb.addVar('sweep', lower=10., upper=30., value=init_sweep)
        # optProb.addVar('alpha', lower=-10., upper=10.)
        optProb.addVar('alpha', lower=0., upper=10., value=init_alpha)

        # Constraints
        optProb.addConGroup('con_failure', 1, upper=0.)
        optProb.addConGroup('con_thickness_intersects', n_thickness_intersects,
                            upper=0., wrt=['thickness_cp'])
        optProb.addConGroup('con_L_equals_W', 1, lower=0., upper=0.)
        optProb.addConGroup('con_CM', n_CM, lower=-0.001, upper=0.001)
        optProb.addConGroup('con_twist_cp', 3, lower=np.array([-1e20, -1e20, 5.]),
                            upper=np.array([1e20, 1e20, 5.]), wrt=['twist_cp'])

        # Objective
        optProb.addObj('obj')
        opt = pyoptsparse.SNOPT(optOptions = {'Major feasibility tolerance' : 1e-9})
        sol = opt(optProb, sens=sens_uq)
        sol = opt(optProb)
        print(sol)

        time_elapsed = time.time() - start_time
        print("time taken = ", time_elapsed)
    else:
        raise NotImplementedError
