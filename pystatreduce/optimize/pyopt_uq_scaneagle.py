# pyopt_uq_scaneagle.py
# The following file contains the deterministic and robust design optimization
# of the ScanEagle problem. In order to run the deterministic problem, enter the
# following command in the terminal
#               python pyopt_uq_scaneagle deterministic
# else enter
#               python pyopt_uq_scaneagle stochastic

import os
import sys
import errno
import time, copy
# sys.path.insert(0, '../../src')

# pyStatReduce specific imports
import numpy as np
import chaospy as cp
from pystatreduce.stochastic_collocation import StochasticCollocation
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

class UQScanEagleOpt(object):
    """
    This class is the conduit for linking pyStatReduce and OpenAeroStruct with
    pyOptSparse.
    """
    def __init__(self, uq_systemsize):

        # Default values
        mean_Ma = 0.071
        mean_TSFC = 9.80665 * 8.6e-6
        mean_W0 = 10.0

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

        surface_dict_rv = {'E' : 85.e9, # RV
                           'G' : 25.e9, # RV
                           'mrho' : 1.6e3, # RV
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

        # Standard deviation
        std_dev = np.diag([0.005, 0.00607/3600, 0.2])
        mu = np.array([mean_Ma, mean_TSFC, mean_W0])
        self.jdist = cp.MvNormal(mu, std_dev)
        self.QoI = examples.OASScanEagleWrapper(uq_systemsize, dv_dict)
        self.dominant_space = DimensionReduction(n_arnoldi_sample=uq_systemsize+1,
                                            exact_Hessian=False)
        self.dominant_space.getDominantDirections(self.QoI, self.jdist, max_eigenmodes=1)

def objfunc_uq(xdict):
    """
    Objective funtion supplied to pyOptSparse for RDO.
    """
    rdo_factor = 2.0
    UQObj.QoI.p['oas_scaneagle.wing.twist_cp'] = xdict['twist_cp']
    UQObj.QoI.p['oas_scaneagle.wing.thickness_cp'] = xdict['thickness_cp']
    UQObj.QoI.p['oas_scaneagle.wing.sweep'] = xdict['sweep']
    UQObj.QoI.p['oas_scaneagle.alpha'] = xdict['alpha']
    obj_func = UQObj.QoI.eval_QoI
    con_func = UQObj.QoI.eval_ConstraintQoI
    funcs = {}

    # Objective function
    # # Full integration
    # mu_j = collocation_obj.normal.mean(cp.E(UQObj.jdist), cp.Std(UQObj.jdist), obj_func)
    # var_j = collocation_obj.normal.variance(obj_func, UQObj.jdist, mu_j)
    # Reduced Integration
    mu_j = collocation_obj.normal.reduced_mean(obj_func, UQObj.jdist, UQObj.dominant_space)
    var_j = collocation_obj.normal.reduced_variance(obj_func, UQObj.jdist, UQObj.dominant_space, mu_j)
    funcs['obj'] = mu_j + rdo_factor * np.sqrt(var_j)

    # Constraint function
    # # full Integration
    # mu_con = collocation_con.normal.mean(cp.E(UQObj.jdist), cp.Std(UQObj.jdist), con_func)
    # var_con = collocation_con.normal.variance(con_func, UQObj.jdist, mu_con)
    # Reduced Integration
    mu_con = collocation_con.normal.reduced_mean(con_func, UQObj.jdist, UQObj.dominant_space)
    var_con = collocation_con.normal.reduced_variance(con_func, UQObj.jdist, UQObj.dominant_space, mu_con)
    n_thickness_intersects = UQObj.QoI.p['oas_scaneagle.AS_point_0.wing_perf.thickness_intersects'].size
    n_CM = UQObj.QoI.p['oas_scaneagle.AS_point_0.CM'].size
    funcs['con_failure'] = mu_con[0] + rdo_factor * np.sqrt(var_con[0,0])
    funcs['con_thickness_intersects'] = mu_con[1:n_thickness_intersects+1]
    funcs['con_L_equals_W'] = mu_con[n_thickness_intersects+1]
    funcs['con_CM'] = mu_con[n_thickness_intersects+2:n_thickness_intersects+2+n_CM]
    funcs['con_twist_cp'] = mu_con[n_thickness_intersects+2+n_CM:]

    fail = False
    return funcs, fail

def objfunc(xdict):
    """
    Objective function supplied to pyOptSparse for deterministic optimization.
    """
    UQObj.QoI.p['oas_scaneagle.wing.twist_cp'] = xdict['twist_cp']
    UQObj.QoI.p['oas_scaneagle.wing.thickness_cp'] = xdict['thickness_cp']
    UQObj.QoI.p['oas_scaneagle.wing.sweep'] = xdict['sweep']
    UQObj.QoI.p['oas_scaneagle.alpha'] = xdict['alpha']
    UQObj.QoI.p.run_model()

    funcs = {}
    funcs['obj'] = UQObj.QoI.p['oas_scaneagle.AS_point_0.fuelburn']
    funcs['con_failure'] = UQObj.QoI.p['oas_scaneagle.AS_point_0.wing_perf.failure']
    funcs['con_thickness_intersects'] = UQObj.QoI.p['oas_scaneagle.AS_point_0.wing_perf.thickness_intersects']
    funcs['con_L_equals_W'] = UQObj.QoI.p['oas_scaneagle.AS_point_0.L_equals_W']
    funcs['con_CM'] = UQObj.QoI.p['oas_scaneagle.AS_point_0.CM']
    funcs['con_twist_cp'] = UQObj.QoI.p['oas_scaneagle.wing.twist_cp']

    fail = False
    # print("fuel burn = ", funcs['obj'])
    return funcs, fail

def sens_uq(xdict, funcs):
    """
    Sensitivity function provided to pyOptSparse for RDO.
    """
    rdo_factor = 2.0
    UQObj.QoI.p['oas_scaneagle.wing.twist_cp'] = xdict['twist_cp']
    UQObj.QoI.p['oas_scaneagle.wing.thickness_cp'] = xdict['thickness_cp']
    UQObj.QoI.p['oas_scaneagle.wing.sweep'] = xdict['sweep']
    UQObj.QoI.p['oas_scaneagle.alpha'] = xdict['alpha']
    obj_func = UQObj.QoI.eval_QoI
    dobj_func = UQObj.QoI.eval_ObjGradient_dv
    dcon_func = UQObj.QoI.eval_ConGradient_dv
    var_con_func = UQObj.QoI.eval_confailureQoI
    dvar_con_func = UQObj.QoI.eval_failureGrad_dv
    funcsSens = {}

    # Objective function
    # # Full integration
    # mu_j = collocation_obj.normal.mean(cp.E(UQObj.jdist), cp.Std(UQObj.jdist), obj_func)
    # var_j = collocation_obj.normal.variance(obj_func, UQObj.jdist, mu_j)
    # Reduced Integration
    mu_j = collocation_obj.normal.reduced_mean(obj_func, UQObj.jdist, UQObj.dominant_space)
    var_j = collocation_obj.normal.reduced_variance(obj_func, UQObj.jdist, UQObj.dominant_space, mu_j)
    dmu_j = collocation_obj_grad.normal.reduced_mean(dobj_func, UQObj.jdist, UQObj.dominant_space)
    dstd_dev_j = collocation_obj_grad.normal.dReducedStdDev(obj_func, UQObj.jdist,
                 UQObj.dominant_space, mu_j, var_j, dobj_func, dmu_j)

    # Constraint function
    # # full Integration
    # mu_con = collocation_con.normal.mean(cp.E(UQObj.jdist), cp.Std(UQObj.jdist), con_func)
    # var_con = collocation_con.normal.variance(con_func, UQObj.jdist, mu_con)
    # Reduced Integration
    dmu_con = collocation_con_grad.normal.reduced_mean(dcon_func, UQObj.jdist, UQObj.dominant_space)
    # - We only need the variance of the of the failure constraint
    #
    mu_confailure = collocation_obj.normal.reduced_mean(var_con_func, UQObj.jdist, UQObj.dominant_space)
    var_confailure = collocation_obj.normal.reduced_variance(var_con_func, UQObj.jdist,
                     UQObj.dominant_space, mu_confailure)
    std_dev_con = collocation_obj_grad.normal.dReducedStdDev(var_con_func, UQObj.jdist,
                  UQObj.dominant_space, mu_confailure, var_confailure, dvar_con_func, dmu_con[0,:])



    # Get some of the intermediate variables
    n_twist_cp = UQObj.QoI.input_dict['n_twist_cp']
    n_cp = n_twist_cp + UQObj.QoI.input_dict['n_thickness_cp']
    n_CM = UQObj.QoI.input_dict['n_CM']
    n_thickness_intersects = UQObj.QoI.p['oas_scaneagle.AS_point_0.wing_perf.thickness_intersects'].size

    # Populate the dictionary
    funcsSens['obj', 'twist_cp'] = dmu_j[0:n_twist_cp] + rdo_factor * dstd_dev_j[0:n_twist_cp]
    funcsSens['obj', 'thickness_cp'] = dmu_j[n_twist_cp:n_cp] + rdo_factor * dstd_dev_j[n_twist_cp:n_cp]
    funcsSens['obj', 'sweep'] = dmu_j[n_cp:n_cp+1] + rdo_factor * dstd_dev_j[n_cp:n_cp+1]
    funcsSens['obj', 'alpha'] = dmu_j[n_cp+1:n_cp+2] + rdo_factor * dstd_dev_j[n_cp+1:n_cp+2]

    funcsSens['con_failure', 'twist_cp'] = dmu_con[0,0:n_twist_cp] + rdo_factor * std_dev_con[0:n_twist_cp]
    funcsSens['con_failure', 'thickness_cp'] = dmu_con[0,n_twist_cp:n_cp] + rdo_factor * std_dev_con[n_twist_cp:n_cp]
    funcsSens['con_failure', 'sweep'] = dmu_con[0,n_cp] + rdo_factor * std_dev_con[n_cp]
    funcsSens['con_failure', 'alpha'] = dmu_con[0,n_cp+1] + rdo_factor * std_dev_con[n_cp+1]

    funcsSens['con_thickness_intersects', 'twist_cp'] = dmu_con[1:n_thickness_intersects+1,0:n_twist_cp]
    funcsSens['con_thickness_intersects', 'thickness_cp'] = dmu_con[1:n_thickness_intersects+1,n_twist_cp:n_cp]
    # print "mu_con[1:n_thickness_intersects+1,n_cp].shape = ", mu_con[1:n_thickness_intersects+1,n_cp].shape
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
    # funcsSens['con_twist_cp', 'thickness_cp'] = mu_con[idx:,n_twist_cp:n_cp]
    # funcsSens['con_twist_cp', 'sweep'] = mu_con[idx:,n_cp:n_cp+1]
    # funcsSens['con_twist_cp', 'alpha'] = mu_con[idx:,n_cp+1:]

    fail = False
    return funcsSens, fail

def sens(xdict, funcs):
    """
    Sensitivity function provided to pyOptSparse for deterministic optimization.
    """
    UQObj.QoI.p['oas_scaneagle.wing.twist_cp'] = xdict['twist_cp']
    UQObj.QoI.p['oas_scaneagle.wing.thickness_cp'] = xdict['thickness_cp']
    UQObj.QoI.p['oas_scaneagle.wing.sweep'] = xdict['sweep']
    UQObj.QoI.p['oas_scaneagle.alpha'] = xdict['alpha']
    UQObj.QoI.p.run_model()

    # Compute derivatives
    deriv = UQObj.QoI.p.compute_totals(of=['oas_scaneagle.AS_point_0.fuelburn',
                                           'oas_scaneagle.AS_point_0.wing_perf.failure',
                                           'oas_scaneagle.AS_point_0.wing_perf.thickness_intersects',
                                           'oas_scaneagle.AS_point_0.L_equals_W',
                                           'oas_scaneagle.AS_point_0.CM'],
                                       wrt=['oas_scaneagle.wing.twist_cp',
                                            'oas_scaneagle.wing.thickness_cp',
                                            'oas_scaneagle.wing.sweep',
                                            'oas_scaneagle.alpha'])
    funcsSens = {}
    funcsSens['obj', 'twist_cp'] = deriv['oas_scaneagle.AS_point_0.fuelburn', 'oas_scaneagle.wing.twist_cp']
    funcsSens['obj', 'thickness_cp'] = deriv['oas_scaneagle.AS_point_0.fuelburn', 'oas_scaneagle.wing.thickness_cp']
    funcsSens['obj', 'sweep'] = deriv['oas_scaneagle.AS_point_0.fuelburn', 'oas_scaneagle.wing.sweep']
    funcsSens['obj', 'alpha'] = deriv['oas_scaneagle.AS_point_0.fuelburn', 'oas_scaneagle.alpha']

    funcsSens['con_failure', 'twist_cp'] = deriv['oas_scaneagle.AS_point_0.wing_perf.failure', 'oas_scaneagle.wing.twist_cp']
    funcsSens['con_failure', 'thickness_cp'] = deriv['oas_scaneagle.AS_point_0.wing_perf.failure', 'oas_scaneagle.wing.thickness_cp']
    funcsSens['con_failure', 'sweep'] = deriv['oas_scaneagle.AS_point_0.wing_perf.failure', 'oas_scaneagle.wing.sweep']
    funcsSens['con_failure', 'alpha'] = deriv['oas_scaneagle.AS_point_0.wing_perf.failure', 'oas_scaneagle.alpha']

    # con_thickness_intersects only depends on thickness_cp
    funcsSens['con_thickness_intersects', 'thickness_cp'] = deriv['oas_scaneagle.AS_point_0.wing_perf.thickness_intersects', 'oas_scaneagle.wing.thickness_cp']

    funcsSens['con_L_equals_W', 'twist_cp'] = deriv['oas_scaneagle.AS_point_0.L_equals_W', 'oas_scaneagle.wing.twist_cp']
    funcsSens['con_L_equals_W', 'thickness_cp'] = deriv['oas_scaneagle.AS_point_0.L_equals_W', 'oas_scaneagle.wing.thickness_cp']
    funcsSens['con_L_equals_W', 'sweep'] = deriv['oas_scaneagle.AS_point_0.L_equals_W', 'oas_scaneagle.wing.sweep']
    funcsSens['con_L_equals_W', 'alpha'] = deriv['oas_scaneagle.AS_point_0.L_equals_W', 'oas_scaneagle.alpha']

    funcsSens['con_CM', 'twist_cp'] = deriv['oas_scaneagle.AS_point_0.CM', 'oas_scaneagle.wing.twist_cp']
    funcsSens['con_CM', 'thickness_cp'] = deriv['oas_scaneagle.AS_point_0.CM', 'oas_scaneagle.wing.thickness_cp']
    funcsSens['con_CM', 'sweep'] = deriv['oas_scaneagle.AS_point_0.CM', 'oas_scaneagle.wing.sweep']
    funcsSens['con_CM', 'alpha'] = deriv['oas_scaneagle.AS_point_0.CM', 'oas_scaneagle.alpha']

    # con_twist_cp only depends on twist_cp and is an identity matrix
    funcsSens['con_twist_cp', 'twist_cp'] = np.eye(UQObj.QoI.input_dict['n_twist_cp'])

    fail = False
    return funcsSens, fail

if __name__ == "__main__":

    uq_systemsize = 3
    UQObj = UQScanEagleOpt(uq_systemsize)
    xi = np.zeros(uq_systemsize)
    mu = np.array([0.071, 9.80665 * 8.6e-6, 10.])
    # Set some of the initial values
    init_twist_cp = np.array([2.5, 2.5, 5.0])
    init_thickness_cp = np.array([0.008, 0.008, 0.008])
    init_sweep = 20.0
    init_alpha = 5.

    if sys.argv[1] == "stochastic":
        """
        This piece of code runs the RDO of the ScanEagle program.
        """
        # Stochastic collocation Objects
        ndv = 3 + 3 + 1 + 1
        collocation_obj = StochasticCollocation(5, "Normal")
        collocation_obj_grad = StochasticCollocation(5, "Normal", QoI_dimensions=ndv)
        n_thickness_intersects = UQObj.QoI.p['oas_scaneagle.AS_point_0.wing_perf.thickness_intersects'].size
        n_CM = 3
        n_constraints = 1 + n_thickness_intersects  + 1 + n_CM + 3
        collocation_con = StochasticCollocation(5, "Normal", QoI_dimensions=n_constraints)
        collocation_con_grad = StochasticCollocation(5, "Normal", QoI_dimensions=(n_constraints, ndv))

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
        # sol = opt(optProb, sens=sens_uq)
        sol = opt(optProb)
        print(sol)

    elif sys.argv[1] == "deterministic":
        """
        This problem runs the deterministic problem which is equivalent to
        `run_scaneagle.py`
        """
        start_time = time.time()
        # fval = UQObj.QoI.eval_QoI(mu, xi)
        # print "twist_cp = ", UQObj.QoI.p['oas_scaneagle.wing.twist_cp']
        # print "thickness_cp = ", UQObj.QoI.p['oas_scaneagle.wing.thickness_cp']
        # print "sweep = ", UQObj.QoI.p['oas_scaneagle.wing.sweep']
        # print "alpha = ", UQObj.QoI.p['oas_scaneagle.alpha'], '\n'
        UQObj.QoI.p.run_model()
        # print "init_fuel_burn = ", UQObj.QoI.p['oas_scaneagle.AS_point_0.fuelburn'], '\n'

        # Design Variables
        optProb = pyoptsparse.Optimization('UQ_OASScanEagle', objfunc)
        n_twist_cp = UQObj.QoI.input_dict['n_twist_cp']
        n_thickness_cp = UQObj.QoI.input_dict['n_thickness_cp']
        optProb.addVarGroup('twist_cp', n_twist_cp, 'c', lower=-5., upper=10, value=init_twist_cp)
        optProb.addVarGroup('thickness_cp', n_thickness_cp, 'c', lower=0.001, upper=0.01, scale=1.e3, value=init_thickness_cp)
        optProb.addVar('sweep', lower=10., upper=30., value=init_sweep)
        optProb.addVar('alpha', lower=-10., upper=10., value=init_alpha)

        # Constraints
        n_thickness_intersects = UQObj.QoI.p['oas_scaneagle.AS_point_0.wing_perf.thickness_intersects'].size
        n_CM = 3
        optProb.addConGroup('con_failure', 1, upper=0.)
        optProb.addConGroup('con_thickness_intersects', n_thickness_intersects,
                            upper=0., wrt=['thickness_cp'])
        optProb.addConGroup('con_L_equals_W', 1, lower=0., upper=0.)
        optProb.addConGroup('con_CM', n_CM, lower=-0.001, upper=0.001)
        optProb.addConGroup('con_twist_cp', 3, lower=np.array([-1e20, -1e20, 5.]),
                            upper=np.array([1e20, 1e20, 5.]), wrt=['twist_cp'])

        # Objective
        optProb.addObj('obj', scale=.1)
        opt = pyoptsparse.SNOPT(optOptions = {'Major feasibility tolerance' : 1e-9,
                                              'Verify level':[int,0],})
        sol = opt(optProb, sens=sens, storeHistory='deterministic.hst')
        time_elapsed = time.time() - start_time
        print(sol)
        print(sol.fStar)
        print("time_elapsed = ", time_elapsed)
