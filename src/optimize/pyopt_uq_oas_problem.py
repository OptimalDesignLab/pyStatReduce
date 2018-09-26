# pyopt_uq_oas_problem.py
# The following file contains UQ optimization of the quick example from
# openaerostruct where the the random variables are not the design variables
# from __future__ import division, print_function
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

class UQOASExample1Opt(object):

    def __init__(self):
        uq_systemsize = 5
        mean_v = 248.136 # Mean value of input random variable
        mean_alpha = 5   #
        mean_Ma = 0.84
        mean_re = 1.e6
        mean_rho = 0.38
        mean_cg = np.zeros((3))

        std_dev = np.diag([1.0, 0.2, 0.01, 1.e2, 0.01]) # np.eye(uq_systemsize)
        mu_init = np.array([mean_v, mean_alpha, mean_Ma, mean_re, mean_rho])
        self.QoI = examples.OASAerodynamicWrapper(uq_systemsize)
        self.jdist = cp.MvNormal(mu_init, std_dev)
        self.dominant_space = DimensionReduction(n_arnoldi_sample=uq_systemsize+1,
                                            exact_Hessian=False)
        self.dominant_space.getDominantDirections(self.QoI, self.jdist, max_eigenmodes=2)

    def compute_sens(self):
        pass

def objfunc(xdict):
    dv = xdict['xvars'] # Get the design variable out
    UQObj.QoI.p['oas_example1.wing.twist_cp'] = dv
    obj_func = UQObj.QoI.eval_QoI
    con_func = UQObj.QoI.eval_ConstraintQoI
    funcs = {}

    # Objective function

    # Full integration
    mu_j = collocation_obj.normal.mean(cp.E(UQObj.jdist), cp.Std(UQObj.jdist), obj_func)
    var_j = collocation_obj.normal.variance(obj_func, UQObj.jdist, mu_j)
    # # Reduced integration
    # mu_j = collocation_obj.normal.reduced_mean(obj_func, UQObj.jdist, UQObj.dominant_space)
    # var_j = collocation_obj.normal.reduced_variance(obj_func, UQObj.jdist, UQObj.dominant_space, mu_j)
    funcs['obj'] = mu_j + 2*np.sqrt(var_j)

    # Constraint function
    # Full integration
    funcs['con'] = collocation_con.normal.mean(cp.E(UQObj.jdist), cp.Std(UQObj.jdist), con_func)
    # # Reduced integration
    # funcs['con'] = collocation_con.normal.reduced_mean(con_func, UQObj.jdist, UQObj.dominant_space)
    fail = False
    return funcs, fail

def sens(xdict, funcs):
    dv = xdict['xvars'] # Get the design variable out
    UQObj.QoI.p['oas_example1.wing.twist_cp'] = dv
    obj_func = UQObj.QoI.eval_ObjGradient
    con_func = UQObj.QoI.eval_ConstraintQoIGradient
    funcsSens = {}

    # Objective function
    # Full integration
    g_mu_j = collocation_grad_obj.normal.mean(cp.E(UQObj.jdist),
                                              cp.Std(UQObj.jdist), obj_func)
    g_var_j = collocation_grad_obj.normal.variance(obj_func, UQObj.jdist, g_mu_j)
    # # Reduced integration
    # g_mu_j = collocation_grad_obj.normal.reduced_mean(obj_func, UQObj.jdist, UQObj.dominant_space)
    # g_var_j = collocation_grad_obj.normal.reduced_variance(obj_func, UQObj.jdist, UQObj.dominant_space, g_mu_j)

    funcsSens['obj', 'xvars'] = g_mu_j + 2*np.sqrt(g_var_j) # collocation_grad_obj.normal.reduced_mean(obj_func, UQObj.jdist, UQObj.dominant_space)

    # Constraint function
    # Reduced integration
    funcsSens['con', 'xvars'] = collocation_grad_con.normal.mean(cp.E(UQObj.jdist), cp.Std(UQObj.jdist), con_func)
    # # Full integration
    # funcsSens['con', 'xvars'] = collocation_grad_con.normal.reduced_mean(con_func, UQObj.jdist, UQObj.dominant_space)
    fail = False
    return funcsSens, fail

if __name__ == "__main__":
    """
    The following script aims to solve the following problem

            min      mu_CD(twist_cp, rv) + 2 * sigma_CD(twist_cp, rv)
         twist_cp

        subject to   mu_CL = 0.5
    """

    uq_systemsize = 5
    UQObj = UQOASExample1Opt()
    collocation_obj = StochasticCollocation(5, "Normal")
    collocation_con = collocation_obj
    collocation_grad_obj = StochasticCollocation(5, "Normal", QoI_dimensions=uq_systemsize)
    collocation_grad_con = collocation_grad_obj
    optProb = pyoptsparse.Optimization('UQ_OASExample1', objfunc)
    optProb.addVarGroup('xvars', 5, 'c', lower=-10., upper=15.)
    optProb.addConGroup('con', 1, lower=0.5, upper=0.5)
    optProb.addObj('obj')
    opt = pyoptsparse.SNOPT(optOptions = {'Major feasibility tolerance' : 1e-10})
    sol = opt(optProb, sens=sens)
    print sol
