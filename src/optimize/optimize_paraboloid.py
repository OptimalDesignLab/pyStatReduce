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
from pyoptsparse import Optimization, OPT

def objfunc(xdict):
    mu = xdict['xvars']
    funcs = {}

    # Initialize pyStatReduce
    systemsize = len(mu)
    theta = 0
    sigma = np.array([0.2, 0.1])
    tuple = (theta,)
    jdist = cp.MvNormal(mu, np.diag(sigma))          # Create joint distribution
    collocation = StochasticCollocation(3, "Normal") # Create a Stochastic collocation object
    QoI = examples.Paraboloid2D(systemsize, tuple)   # Create QoI
    dominant_space = DimensionReduction(threshold_factor, exact_Hessian=True)
    dominant_space.getDominantDirections(QoI, jdist)

    QoI_func = QoI.eval_QoI
    funcs['obj'] = collocation.normal.reduced_mean(QoI_func, jdist, dominant_space)

    fail = False

def sens(xdict, funcs):

    mu = xdict['xvars']
    funcsSens = {}
    funcsSens['obj', 'xvars'] =
