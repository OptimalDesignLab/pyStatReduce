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
        # print "wing.twist_cp",  self.QoI.p['oas_example1.wing.twist_cp']

    def compute_sens(self):
        pass

def objfunc(xdict):
    dv = xdict['xvars'] # Get the design variable out
    UQObj.QoI.p['oas_example1.wing.twist_cp'] = dv
    obj_func = UQObj.QoI.eval_QoI
    con_func = UQObj.QoI.eval_ConstraintQoI
    funcs = {}
    funcs['obj'] = collocation_obj.normal.reduced_mean(obj_func, UQObj.jdist, UQObj.dominant_space)
    funcs['con'] = collocation_con.normal.reduced_mean(con_func, UQObj.jdist, UQObj.dominant_space)
    fail = False
    return funcs, fail

def sens(xdict, funcs):
    dv = xdict['xvars'] # Get the design variable out
    UQObj.QoI.p['oas_example1.wing.twist_cp'] = dv
    obj_func = UQObj.QoI.eval_ObjGradient
    con_func = UQObj.QoI.eval_ConstraintQoIGradient
    funcsSens = {}
    funcsSens['obj', 'xvars'] = collocation_grad_obj.normal.reduced_mean(obj_func, UQObj.jdist, UQObj.dominant_space)
    funcsSens['con', 'xvars'] = collocation_grad_con.normal.reduced_mean(con_func, UQObj.jdist, UQObj.dominant_space)
    fail = False
    return funcsSens, fail

if __name__ == "__main__":

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
