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
from pystatreduce.examples.supersonic_interceptor.interceptor_rdo2 import DymosInterceptorGlue # DymosInterceptorQoI
import pystatreduce.utils as utils

#pyoptsparse sepecific imports
from scipy import sparse
import argparse
import pyoptsparse # from pyoptsparse import Optimization, OPT, SNOPT

# Import the OpenMDAo shenanigans
from openmdao.api import IndepVarComp, Problem, Group, NewtonSolver, \
    ScipyIterativeSolver, LinearBlockGS, NonlinearBlockGS, \
    DirectSolver, LinearBlockGS, PetscKSP, SqliteRecorder, ScipyOptimizeDriver

input_dict = {'num_segments': 15,
              'transcription_order' : 3,
              'transcription_type': 'LGR',
              'solve_segments': False,
              'use_for_collocation' : False,
              'n_collocation_samples': 20,
              'use_polynomial_control': False}
systemsize = input_dict['num_segments'] * input_dict['transcription_order']

dymos_obj = DymosInterceptorGlue(systemsize, input_dict)

# t_f = dymos_obj.eval_QoI(np.zeros(systemsize), np.zeros(systemsize))
# print('t_f = ', t_f)

# grad_tf = dymos_obj.eval_QoIGradient(np.zeros(systemsize), np.zeros(systemsize))
# print('grad_tf = ', grad_tf.size)
mu = np.zeros(systemsize)
std_dev = 0.04*np.eye(systemsize) # 0.04* np.random.rand(systemsize)
jdist = cp.MvNormal(mu, std_dev)

dominant_space = DimensionReduction(n_arnoldi_sample=20,
                                    exact_Hessian=False,
                                    sample_radius=1.e-1)
dominant_space.getDominantDirections(dymos_obj, jdist, max_eigenmodes=10)

print('eigenvals = ', dominant_space.iso_eigenvals)
print('eigenvecs = \n', dominant_space.iso_eigenvecs)
