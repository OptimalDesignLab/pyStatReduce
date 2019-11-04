# Optimize a paraboloid
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

def objfunc(xdict):
    mu = xdict['xvars']
    funcs = {}

    # Initialize pyStatReduce
    systemsize = len(mu)
    theta = 0
    sigma = np.array([0.3, 0.2, 0.1])
    jdist = cp.MvNormal(mu, np.diag(sigma))          # Create joint distribution
    collocation = StochasticCollocation(3, "Normal") # Create a Stochastic collocation object
    QoI = examples.Paraboloid3D(systemsize)          # Create QoI
    threshold_factor = 0.9
    dominant_space = DimensionReduction(threshold_factor, exact_Hessian=True)
    dominant_space.getDominantDirections(QoI, jdist)

    QoI_func = QoI.eval_QoI
    # funcs['obj'] = collocation.normal.mean(mu, sigma, QoI_func)
    funcs['obj'] = collocation.normal.reduced_mean(QoI_func, jdist, dominant_space)

    fail = False
    return funcs, fail

def sens(xdict, funcs):

    mu = xdict['xvars']
    # Initialize pyStatReduce
    systemsize = len(mu)
    sigma = np.array([0.3, 0.2, 0.1])
    jdist = cp.MvNormal(mu, np.diag(sigma))
    collocation = StochasticCollocation(3, "Normal", systemsize)
    QoI = examples.Paraboloid3D(systemsize)   # Create QoI
    threshold_factor = 0.9
    dominant_space = DimensionReduction(threshold_factor, exact_Hessian=True)
    dominant_space.getDominantDirections(QoI, jdist)
    QoI_func = QoI.eval_QoIGradient

    funcsSens = {}
    # funcsSens['obj', 'xvars'] = collocation.normal.mean(mu, sigma, QoI_func)
    funcsSens['obj', 'xvars'] = collocation.normal.reduced_mean(QoI_func, jdist, dominant_space)

    fail = False
    return funcsSens, fail

# Optimization Object
optProb = Optimization('Paraboloid', objfunc)
lower_bound = -20*np.ones(3)
upper_bound = 20*np.ones(3)
optProb.addVarGroup('xvars', 3, 'c', lower=lower_bound, upper=upper_bound, value=10*np.ones(3))
optProb.addObj('obj')
# Optimizer
opt = SNOPT(optOptions = {'Major feasibility tolerance' : 1e-1})
sol = opt(optProb, sens=sens)

# Check Solution
print(sol)
