# Use the OpenMDAO framework for OUU for optimizing a Rosenbrock function
from __future__ import division, print_function
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
from pyoptsparse import Optimization, OPT, SNOPT

# pyDimReduce imports
from pydimreduce.quantities_of_interest.rosenbrock import Rosenbrock

# OpenMDAO imports
from openmdao.api import Problem, IndepVarComp, ExplicitComponent, Group

class RosenbrockOpt(object):

    def __init__(self, n_random):
        self.systemsize = n_random
        self.p = Problem()
        self.ivc = self.p.model.add_subsystem('design_point', IndepVarComp())
        self.ivc.add_output('x', shape=(n_random,))
        self.p.model.add_subsystem('rosenbrock', Rosenbrock(size=n_random))
        self.p.model.connect('design_point.x', 'rosenbrock.rv')
        self.p.setup()


    def eval_QoI(self, mu, xi):
        rv = mu + xi
        self.p['design_point.x'] = rv
        self.p.run_model()
        # print(self.p['rosenbrock.fval'][0])
        return self.p['rosenbrock.fval'][0]

    def eval_QoIGradient(self, mu, xi):
        rv = mu + xi
        self.p['design_point.x'] = rv
        self.p.run_model()
        deriv = self.p.compute_totals(of=['rosenbrock.fval'], wrt=['design_point.x'])
        # print(deriv['rosenbrock.fval', 'design_point.x'][0])

        return deriv['rosenbrock.fval', 'design_point.x'][0]


def objfunc(xdict):
    mu = xdict['xvars']
    funcs = {}
    jdist = cp.MvNormal(mu, std_dev)
    QoI_func = QoI.eval_QoI
    funcs['obj'] = collocation.normal.reduced_mean(QoI_func, jdist, dominant_space)
    fail = False
    return funcs, fail

def sens(xdict, funcs):
    mu = xdict['xvars']
    jdist = cp.MvNormal(mu, std_dev)
    QoI_func = QoI.eval_QoIGradient
    funcsSens = {}
    funcsSens['obj', 'xvars'] = collocation_grad.normal.reduced_mean(QoI_func, jdist, dominant_space)

    fail = False
    return funcsSens, fail

if __name__ == "__main__":

    # Instantiate the rosenbrock problem globally
    rv_systemsize = 2
    initial_seed = 2*np.ones(rv_systemsize)
    QoI = RosenbrockOpt(rv_systemsize)
    std_dev = np.eye(rv_systemsize)
    jdist = cp.MvNormal(initial_seed, std_dev)
    collocation = StochasticCollocation(3, "Normal")
    collocation_grad = StochasticCollocation(3, "Normal", QoI_dimensions=rv_systemsize)
    threshold_factor = 0.9
    dominant_space = DimensionReduction(threshold_factor, n_arnoldi_sample=3, exact_Hessian=False)
    dominant_space.getDominantDirections(QoI, jdist)

    # Setup the problem
    optProb = Optimization('Paraboloid', objfunc)
    lower_bound = -20*np.ones(rv_systemsize)
    upper_bound = 20*np.ones(rv_systemsize)
    optProb.addVarGroup('xvars', rv_systemsize, 'c', lower=lower_bound,
                        upper=upper_bound, value=10*np.ones(rv_systemsize))
    optProb.addObj('obj')
    # Optimizer
    opt = SNOPT(optOptions = {'Major feasibility tolerance' : 1e-6})
    sol = opt(optProb, sens=sens)

    # Check Solution
    import inspect
    print(sol.fStar)
    print(sol.getDVs()['xvars'])
