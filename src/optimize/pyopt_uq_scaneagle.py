# pyopt_uq_scaneagle.py
# The following file will contain the optimization under uncertainty of the
# ScanEagle problem.
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
                    }

        # Standard deviation
        std_dev = np.diag([0.005, 0.00607/3600, 0.2])
        mu = np.array([mean_Ma, mean_TSFC, mean_W0])
        self.jdist = cp.MvNormal(mu, std_dev)
        self.QoI = examples.OASScanEagleWrapper(uq_systemsize, dv_dict)
        self.dominant_space = DimensionReduction(n_arnoldi_sample=uq_systemsize+1,
                                            exact_Hessian=False)
        self.dominant_space.getDominantDirections(self.QoI, self.jdist, max_eigenmodes=3)
        # print 'iso_eigenvals = ', self.dominant_space.iso_eigenvals
        # print 'iso_eigenvecs = ', '\n', self.dominant_space.iso_eigenvecs
        # print 'dominant_indices = ', self.dominant_space.dominant_indices
        # print 'ratio = ', abs(self.dominant_space.iso_eigenvals[0] / self.dominant_space.iso_eigenvals[1])
"""
def objfunc(xdict):
    dv = xdict['xvars']
    # Get the different design variable sizes
    n_twist_cp = UQObj.QoI.dv_dict['n_twist_cp']
    n_thickness_cp = UQObj.QoI.dv_dict['n_thickness_cp']
    n_cp = n_twist_cp + n_thickness_cp
    UQObj.QoI.p['oas_scaneagle.wing.twist_cp'] = dv[0:n_twist_cp]
    UQObj.QoI.p['oas_scaneagle.wing.thickness_cp'] = dv[n_twist_cp:n_cp]
    UQObj.QoI.p['oas_scaneagle.wing.sweep'] = dv[-2]
    UQObj.QoI.p['oas_scaneagle.alpha'] = dv[-1]
    funcs = {}

    # Deterministic
    UQObj.QoI.p.run_model()
    funcs['obj'] = UQObj.QoI.p['oas_scaneagle.AS_point_0.fuelburn']
    # - Get the requisite sizes of the different constraints
    n_fail = 1
    n_thickness_intersects = UQObj.QoI.p['oas_scaneagle.AS_point_0.wing_perf.thickness_intersects'].size
    n_LW = 1
    n_CM = 3
    n_constraints = n_fail + n_thickness_intersects + n_LW + n_CM
    constraint_arr = np.zeros(n_constraints)
    constraint_arr[0] = UQObj.QoI.p['oas_scaneagle.AS_point_0.wing_perf.failure'][0]
    constraint_arr[1:n_thickness_intersects+1] = \
          UQObj.QoI.p['oas_scaneagle.AS_point_0.wing_perf.thickness_intersects']
    constraint_arr[n_thickness_intersects+1] = UQObj.QoI.p['oas_scaneagle.AS_point_0.L_equals_W']
    constraint_arr[n_thickness_intersects+2:n_constraints] = UQObj.QoI.p['oas_scaneagle.AS_point_0.CM']
    funcs['con'] = constraint_arr
    fail = False
    return funcs, fail
"""
def objfunc(xdict):
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

def sens(xdict, funcs):
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
                                           'oas_scaneagle.AS_point_0.CM',
                                           'oas_scaneagle.wing.twist_cp'],
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

    funcsSens['con_thickness_intersects', 'twist_cp'] = deriv['oas_scaneagle.AS_point_0.wing_perf.thickness_intersects', 'oas_scaneagle.wing.twist_cp']
    funcsSens['con_thickness_intersects', 'thickness_cp'] = deriv['oas_scaneagle.AS_point_0.wing_perf.thickness_intersects', 'oas_scaneagle.wing.thickness_cp']
    funcsSens['con_thickness_intersects', 'sweep'] = deriv['oas_scaneagle.AS_point_0.wing_perf.thickness_intersects', 'oas_scaneagle.wing.sweep']
    funcsSens['con_thickness_intersects', 'alpha'] = deriv['oas_scaneagle.AS_point_0.wing_perf.thickness_intersects', 'oas_scaneagle.alpha']

    funcsSens['con_L_equals_W', 'twist_cp'] = deriv['oas_scaneagle.AS_point_0.L_equals_W', 'oas_scaneagle.wing.twist_cp']
    funcsSens['con_L_equals_W', 'thickness_cp'] = deriv['oas_scaneagle.AS_point_0.L_equals_W', 'oas_scaneagle.wing.thickness_cp']
    funcsSens['con_L_equals_W', 'sweep'] = deriv['oas_scaneagle.AS_point_0.L_equals_W', 'oas_scaneagle.wing.sweep']
    funcsSens['con_L_equals_W', 'alpha'] = deriv['oas_scaneagle.AS_point_0.L_equals_W', 'oas_scaneagle.alpha']

    funcsSens['con_CM', 'twist_cp'] = deriv['oas_scaneagle.AS_point_0.CM', 'oas_scaneagle.wing.twist_cp']
    funcsSens['con_CM', 'thickness_cp'] = deriv['oas_scaneagle.AS_point_0.CM', 'oas_scaneagle.wing.thickness_cp']
    funcsSens['con_CM', 'sweep'] = deriv['oas_scaneagle.AS_point_0.CM', 'oas_scaneagle.wing.sweep']
    funcsSens['con_CM', 'alpha'] = deriv['oas_scaneagle.AS_point_0.CM', 'oas_scaneagle.alpha']

    funcsSens['con_twist_cp', 'twist_cp'] = deriv['oas_scaneagle.wing.twist_cp', 'oas_scaneagle.wing.twist_cp']
    funcsSens['con_twist_cp', 'thickness_cp'] = deriv['oas_scaneagle.wing.twist_cp', 'oas_scaneagle.wing.thickness_cp']
    funcsSens['con_twist_cp', 'sweep'] = deriv['oas_scaneagle.wing.twist_cp', 'oas_scaneagle.wing.sweep']
    funcsSens['con_twist_cp', 'alpha'] = deriv['oas_scaneagle.wing.twist_cp', 'oas_scaneagle.alpha']

    fail = False
    return funcsSens, fail

if __name__ == "__main__":
    uq_systemsize = 3
    UQObj = UQScanEagleOpt(uq_systemsize)
    xi = np.zeros(uq_systemsize)
    mu = np.array([0.071, 9.80665 * 8.6e-6, 10.])
    fval = UQObj.QoI.eval_QoI(mu, xi)
    UQObj.QoI.p.run_model()


    # Design Variables
    optProb = pyoptsparse.Optimization('UQ_OASScanEagle', objfunc)
    n_twist_cp = UQObj.QoI.dv_dict['n_twist_cp']
    n_thickness_cp = UQObj.QoI.dv_dict['n_thickness_cp']
    optProb.addVarGroup('twist_cp', n_twist_cp, 'c', lower=-5., upper=10)
    optProb.addVarGroup('thickness_cp', n_thickness_cp, 'c', lower=0.001, upper=0.01, scale=1.e3)
    optProb.addVar('sweep', lower=10., upper=30.)
    optProb.addVar('alpha', lower=-10., upper=10.)

    # Constraints
    n_thickness_intersects = UQObj.QoI.p['oas_scaneagle.AS_point_0.wing_perf.thickness_intersects'].size
    n_CM = 3
    optProb.addConGroup('con_failure', 1, upper=0.)
    optProb.addConGroup('con_thickness_intersects', n_thickness_intersects, upper=0.)
    optProb.addConGroup('con_L_equals_W', 1, lower=0., upper=0.)
    optProb.addConGroup('con_CM', n_CM, lower=-0.001, upper=0.001)
    optProb.addConGroup('con_twist_cp', 3, lower=np.array([-1e20, -1e20, 5.]), upper=np.array([1e20, 1e20, 5.]))

    # Objective
    optProb.addObj('obj')
    opt = pyoptsparse.SNOPT(optOptions = {'Major feasibility tolerance' : 1e-9})
    # sol = opt(optProb, sens='FD')
    sol = opt(optProb, sens=sens)
    print sol

    """
    optProb = pyoptsparse.Optimization('UQ_OASScanEagle', objfunc)
    # Add the design variables
    ndv = UQObj.QoI.dv_dict['n_twist_cp'] + \
          UQObj.QoI.dv_dict['n_thickness_cp'] + 2
    twist_cp_lb = -5.0 * np.ones(3)
    twist_cp_ub = 10.0 * np.ones(3)
    thickness_cp_lb = 0.001 * np.ones(3) #  * 1.e3
    thickness_cp_ub = 0.01 * np.ones(3) #  * 1.e3
    dv_lb = np.concatenate([twist_cp_lb, thickness_cp_lb, [10.], [-10.]])
    dv_ub = np.concatenate([twist_cp_ub, thickness_cp_ub, [30.], [10.]])
    optProb.addVarGroup('xvars', ndv, 'c', lower=dv_lb, upper=dv_ub)

    # Add the constraints
    n_fail = 1
    n_thickness_intersects = UQObj.QoI.p['oas_scaneagle.AS_point_0.wing_perf.thickness_intersects'].size
    n_LW = 1
    n_CM = 3
    n_constraints = n_fail + n_thickness_intersects + n_LW + n_CM
    con_lb = np.concatenate([[-1.e20], -1.e20*np.ones(n_thickness_intersects), [0.],
                        -0.001*np.ones(n_CM)])
    con_ub = np.concatenate([[0.], np.zeros(n_thickness_intersects), [0.],
                            0.001*np.ones(n_CM)])
    optProb.addConGroup('con', n_constraints, lower=con_lb, upper=con_ub)
    optProb.addObj('obj', scale=0.1)
    opt = pyoptsparse.SNOPT(optOptions = {'Major feasibility tolerance' : 1e-10})
    sol = opt(optProb, sens='FD')
    print sol

    # Design variables
    print 'oas_scaneagle.wing.twist_cp = ', UQObj.QoI.p['oas_scaneagle.wing.twist_cp']
    print 'oas_scaneagle.wing.thickness_cp = ', UQObj.QoI.p['oas_scaneagle.wing.thickness_cp']
    print 'oas_scaneagle.wing.sweep = ', UQObj.QoI.p['oas_scaneagle.wing.sweep']
    print 'alpha = ', UQObj.QoI.p['oas_scaneagle.alpha'], '\n'

    # Contraints
    print 'oas_scaneagle.AS_point_0.wing_perf.failure = ', UQObj.QoI.p['oas_scaneagle.AS_point_0.wing_perf.failure']
    print 'oas_scaneagle.AS_point_0.wing_perf.thickness_intersects = ', UQObj.QoI.p['oas_scaneagle.AS_point_0.wing_perf.thickness_intersects']
    print 'oas_scaneagle.AS_point_0.L_equals_W = ', UQObj.QoI.p['oas_scaneagle.AS_point_0.L_equals_W']
    print 'oas_scaneagle.AS_point_0.CM = ', UQObj.QoI.p['oas_scaneagle.AS_point_0.CM']
    """
