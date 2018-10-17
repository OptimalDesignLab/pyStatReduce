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
sys.path.insert(0, '../../src')

# pyStatReduce specific imports
import numpy as np
import chaospy as cp
from stochastic_collocation import StochasticCollocation
from quantity_of_interest import QuantityOfInterest
from dimension_reduction import DimensionReduction
from stochastic_arnoldi.arnoldi_sample import ArnoldiSampling
import examples

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
        dv_dict = {
                   'n_twist_cp' : 3,
                   'n_thickness_cp' : 3,
                   'n_CM' : 3,
                   'n_thickness_intersects' : 10,
                   'n_constraints' : 1 + 10 + 1 + 3 + 3,
                   'ndv' : 3 + 3 + 2,
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
    funcs['con_failure'] = mu_con[0] + rdo_factor * var_con[0]
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
    # print funcs['obj']
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
    obj_func = UQObj.QoI.eval_ObjGradient_dv
    con_func = UQObj.QoI.eval_ConGradient_dv
    funcsSens = {}

    # Objective function
    # # Full integration
    # mu_j = collocation_obj.normal.mean(cp.E(UQObj.jdist), cp.Std(UQObj.jdist), obj_func)
    # var_j = collocation_obj.normal.variance(obj_func, UQObj.jdist, mu_j)
    # Reduced Integration
    mu_j = collocation_obj_grad.normal.reduced_mean(obj_func, UQObj.jdist, UQObj.dominant_space)
    var_j = collocation_obj_grad.normal.reduced_variance(obj_func, UQObj.jdist, UQObj.dominant_space, mu_j)
    funcs['obj'] = mu_j + rdo_factor * np.sqrt(var_j)

    # Constraint function
    # # full Integration
    # mu_con = collocation_con.normal.mean(cp.E(UQObj.jdist), cp.Std(UQObj.jdist), con_func)
    # var_con = collocation_con.normal.variance(con_func, UQObj.jdist, mu_con)
    # Reduced Integration
    mu_con = collocation_con_grad.normal.reduced_mean(con_func, UQObj.jdist, UQObj.dominant_space)
    var_con = collocation_con_grad.normal.reduced_variance(con_func, UQObj.jdist, UQObj.dominant_space, mu_con)


    # Get some of the intermediate variables
    n_twist_cp = UQObj.QoI.input_dict['n_twist_cp']
    n_cp = n_twist_cp + UQObj.QoI.input_dict['n_thickness_cp']
    n_CM = UQObj.QoI.input_dict['n_CM']
    n_thickness_intersects = UQObj.QoI.p['oas_scaneagle.AS_point_0.wing_perf.thickness_intersects'].size

    # Populate the dictionary
    funcsSens['obj', 'twist_cp'] = mu_j[0:n_twist_cp] + rdo_factor * var_j[0:n_twist_cp]
    funcsSens['obj', 'thickness_cp'] = mu_j[n_twist_cp:n_cp] + rdo_factor * var_j[n_twist_cp:n_cp]
    funcsSens['obj', 'sweep'] = mu_j[n_cp:n_cp+1] + rdo_factor * var_j[n_cp:n_cp+1]
    funcsSens['obj', 'alpha'] = mu_j[n_cp+1:n_cp+2] + rdo_factor * var_j[n_cp+1:n_cp+2]

    funcsSens['con_failure', 'twist_cp'] = mu_con[0,0:n_twist_cp] + rdo_factor * var_con[0,0:n_twist_cp]
    funcsSens['con_failure', 'thickness_cp'] = mu_con[0,n_twist_cp:n_cp] + rdo_factor * var_con[0,n_twist_cp:n_cp]
    funcsSens['con_failure', 'sweep'] = mu_con[0,n_cp] + rdo_factor * var_con[0,n_cp]
    funcsSens['con_failure', 'alpha'] = mu_con[0,n_cp+1] + rdo_factor * var_con[0,n_cp+1]

    funcsSens['con_thickness_intersects', 'twist_cp'] = mu_con[1:n_thickness_intersects+1,0:n_twist_cp]
    funcsSens['con_thickness_intersects', 'thickness_cp'] = mu_con[1:n_thickness_intersects+1,n_twist_cp:n_cp]
    # print "mu_con[1:n_thickness_intersects+1,n_cp].shape = ", mu_con[1:n_thickness_intersects+1,n_cp].shape
    funcsSens['con_thickness_intersects', 'sweep'] = mu_con[1:n_thickness_intersects+1,n_cp:n_cp+1]
    funcsSens['con_thickness_intersects', 'alpha'] = mu_con[1:n_thickness_intersects+1,n_cp+1:]

    funcsSens['con_L_equals_W', 'twist_cp'] = mu_con[n_thickness_intersects+1,0:n_twist_cp]
    funcsSens['con_L_equals_W', 'thickness_cp'] = mu_con[n_thickness_intersects+1,n_twist_cp:n_cp]
    funcsSens['con_L_equals_W', 'sweep'] = mu_con[n_thickness_intersects+1,n_cp]
    funcsSens['con_L_equals_W', 'alpha'] = mu_con[n_thickness_intersects+1,n_cp+1]

    idx = n_thickness_intersects + 2
    funcsSens['con_CM', 'twist_cp'] = mu_con[idx:idx+n_CM,0:n_twist_cp]
    funcsSens['con_CM', 'thickness_cp'] = mu_con[idx:idx+n_CM,n_twist_cp:n_cp]
    funcsSens['con_CM', 'sweep'] = mu_con[idx:idx+n_CM,n_cp:n_cp+1]
    funcsSens['con_CM', 'alpha'] = mu_con[idx:idx+n_CM,n_cp+1:]

    idx = n_thickness_intersects + 2 + n_CM
    funcsSens['con_twist_cp', 'twist_cp'] = mu_con[idx:,0:n_twist_cp]
    funcsSens['con_twist_cp', 'thickness_cp'] = mu_con[idx:,n_twist_cp:n_cp]
    funcsSens['con_twist_cp', 'sweep'] = mu_con[idx:,n_cp:n_cp+1]
    funcsSens['con_twist_cp', 'alpha'] = mu_con[idx:,n_cp+1:]

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

    # print "funcsSens['obj', 'twist_cp'] = ", funcsSens['obj', 'twist_cp']
    # print "funcsSens['obj', 'thickness_cp'] = ", funcsSens['obj', 'thickness_cp']
    # print "funcsSens['obj', 'sweep'] = ", funcsSens['obj', 'sweep']
    # print "funcsSens['obj', 'alpha'] = ", funcsSens['obj', 'alpha'], '\n'

    funcsSens['con_failure', 'twist_cp'] = deriv['oas_scaneagle.AS_point_0.wing_perf.failure', 'oas_scaneagle.wing.twist_cp']
    funcsSens['con_failure', 'thickness_cp'] = deriv['oas_scaneagle.AS_point_0.wing_perf.failure', 'oas_scaneagle.wing.thickness_cp']
    funcsSens['con_failure', 'sweep'] = deriv['oas_scaneagle.AS_point_0.wing_perf.failure', 'oas_scaneagle.wing.sweep']
    funcsSens['con_failure', 'alpha'] = deriv['oas_scaneagle.AS_point_0.wing_perf.failure', 'oas_scaneagle.alpha']

    # print "funcsSens['con_failure', 'twist_cp'] = ", funcsSens['con_failure', 'twist_cp']
    # print "funcsSens['con_failure', 'thickness_cp'] = ", funcsSens['con_failure', 'thickness_cp']
    # print "funcsSens['con_failure', 'sweep'] = ", funcsSens['con_failure', 'sweep']
    # print "funcsSens['con_failure', 'alpha'] = ", funcsSens['con_failure', 'alpha'], '\n'

    # con_thickness_intersects only depends on thickness_cp
    funcsSens['con_thickness_intersects', 'thickness_cp'] = deriv['oas_scaneagle.AS_point_0.wing_perf.thickness_intersects', 'oas_scaneagle.wing.thickness_cp']

    # print "funcsSens['con_thickness_intersects', 'twist_cp'] = \n", funcsSens['con_thickness_intersects', 'twist_cp']
    # print "funcsSens['con_thickness_intersects', 'thickness_cp'] = \n", funcsSens['con_thickness_intersects', 'thickness_cp']
    # print "funcsSens['con_thickness_intersects', 'sweep'] = \n", funcsSens['con_thickness_intersects', 'sweep']
    # print "funcsSens['con_thickness_intersects', 'alpha'] = \n", funcsSens['con_thickness_intersects', 'alpha'], '\n'

    funcsSens['con_L_equals_W', 'twist_cp'] = deriv['oas_scaneagle.AS_point_0.L_equals_W', 'oas_scaneagle.wing.twist_cp']
    funcsSens['con_L_equals_W', 'thickness_cp'] = deriv['oas_scaneagle.AS_point_0.L_equals_W', 'oas_scaneagle.wing.thickness_cp']
    funcsSens['con_L_equals_W', 'sweep'] = deriv['oas_scaneagle.AS_point_0.L_equals_W', 'oas_scaneagle.wing.sweep']
    funcsSens['con_L_equals_W', 'alpha'] = deriv['oas_scaneagle.AS_point_0.L_equals_W', 'oas_scaneagle.alpha']

    # print "funcsSens['con_L_equals_W', 'twist_cp'] = ", funcsSens['con_L_equals_W', 'twist_cp']
    # print "funcsSens['con_L_equals_W', 'thickness_cp'] = ", funcsSens['con_L_equals_W', 'thickness_cp']
    # print "funcsSens['con_L_equals_W', 'sweep'] = ", funcsSens['con_L_equals_W', 'sweep']
    # print "funcsSens['con_L_equals_W', 'alpha'] = ", funcsSens['con_L_equals_W', 'alpha'], '\n'

    funcsSens['con_CM', 'twist_cp'] = deriv['oas_scaneagle.AS_point_0.CM', 'oas_scaneagle.wing.twist_cp']
    funcsSens['con_CM', 'thickness_cp'] = deriv['oas_scaneagle.AS_point_0.CM', 'oas_scaneagle.wing.thickness_cp']
    funcsSens['con_CM', 'sweep'] = deriv['oas_scaneagle.AS_point_0.CM', 'oas_scaneagle.wing.sweep']
    funcsSens['con_CM', 'alpha'] = deriv['oas_scaneagle.AS_point_0.CM', 'oas_scaneagle.alpha']

    # print "funcsSens['con_CM', 'twist_cp'] = \n", funcsSens['con_CM', 'twist_cp']
    # print "funcsSens['con_CM', 'thickness_cp'] = \n", funcsSens['con_CM', 'thickness_cp']
    # print "funcsSens['con_CM', 'sweep'] = \n", funcsSens['con_CM', 'sweep']
    # print "funcsSens['con_CM', 'alpha'] = \n", funcsSens['con_CM', 'alpha'], '\n'

    # con_twist_cp only depends on twist_cp and is an identity matrix
    funcsSens['con_twist_cp', 'twist_cp'] = np.eye(UQObj.QoI.input_dict['n_twist_cp'])

    # print "funcsSens['con_twist_cp', 'twist_cp'] = \n", funcsSens['con_twist_cp', 'twist_cp']
    # print "funcsSens['con_twist_cp', 'thickness_cp'] = \n", funcsSens['con_twist_cp', 'thickness_cp']
    # print "funcsSens['con_twist_cp', 'sweep'] = ", funcsSens['con_twist_cp', 'sweep']
    # print "funcsSens['con_twist_cp', 'alpha'] = ", funcsSens['con_twist_cp', 'alpha'], '\n'

    fail = False
    return funcsSens, fail

if __name__ == "__main__":

    uq_systemsize = 3
    UQObj = UQScanEagleOpt(uq_systemsize)
    xi = np.zeros(uq_systemsize)
    mu = np.array([0.071, 9.80665 * 8.6e-6, 10.])

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
        optProb.addVarGroup('twist_cp', n_twist_cp, 'c', lower=-5., upper=10)
        optProb.addVarGroup('thickness_cp', n_thickness_cp, 'c', lower=0.001, upper=0.01, scale=1.e3)
        optProb.addVar('sweep', lower=10., upper=30.)
        # optProb.addVar('alpha', lower=-10., upper=10.)
        optProb.addVar('alpha', lower=0., upper=10.)

        # Constraints
        optProb.addConGroup('con_failure', 1, upper=0.)
        optProb.addConGroup('con_thickness_intersects', n_thickness_intersects, upper=0.)
        optProb.addConGroup('con_L_equals_W', 1, lower=0., upper=0.)
        optProb.addConGroup('con_CM', n_CM, lower=-0.001, upper=0.001)
        optProb.addConGroup('con_twist_cp', 3, lower=np.array([-1e20, -1e20, 5.]), upper=np.array([1e20, 1e20, 5.]))

        # Objective
        optProb.addObj('obj')
        opt = pyoptsparse.SNOPT(optOptions = {'Major feasibility tolerance' : 1e-9})
        # sol = opt(optProb, sens=sens_uq)
        sol = opt(optProb, sens='FD')
        print sol

    elif sys.argv[1] == "deterministic":
        """
        This problem runs the deterministic problem which is equivalent to
        `run_scaneagle.py`
        """

        # fval = UQObj.QoI.eval_QoI(mu, xi)
        # UQObj.QoI.p.run_model()

        # Design Variables
        optProb = pyoptsparse.Optimization('UQ_OASScanEagle', objfunc)
        n_twist_cp = UQObj.QoI.input_dict['n_twist_cp']
        n_thickness_cp = UQObj.QoI.input_dict['n_thickness_cp']
        optProb.addVarGroup('twist_cp', n_twist_cp, 'c', lower=-5., upper=10)
        optProb.addVarGroup('thickness_cp', n_thickness_cp, 'c', lower=0.001, upper=0.01, scale=1.e3)
        optProb.addVar('sweep', lower=10., upper=30.)
        optProb.addVar('alpha', lower=-10., upper=10.)

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
        optProb.addObj('obj', scale=0.1)
        opt = pyoptsparse.SNOPT(optOptions = {'Major feasibility tolerance' : 1e-9})
        # sol = opt(optProb, sens='FD')
        sol = opt(optProb, sens=sens)
        print sol
        print sol.fStar
