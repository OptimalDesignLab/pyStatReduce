################################################################################
# surrogate_eigen_analysis.py
#
# This file contains the eigenvalue analysis of the quadratic surrogate that was
# constructed using the surrogate-modeling toolbox from UMich.
#
################################################################################
from __future__ import division, print_function
import os, sys, errno, copy
import warnings

# pyStatReduce specific imports
import numpy as np
import chaospy as cp
from pystatreduce.new_stochastic_collocation import StochasticCollocation2
from pystatreduce.monte_carlo import MonteCarlo
from pystatreduce.quantity_of_interest import QuantityOfInterest
from pystatreduce.dimension_reduction import DimensionReduction
from pystatreduce.stochastic_arnoldi.arnoldi_sample import ArnoldiSampling
import pystatreduce.utils as utils
import pystatreduce.examples as examples
from pystatreduce.optimize.dymos_interceptor.quadratic_surrogate.interceptor_surrogate_qoi import InterceptorSurrogateQoI

#pyoptsparse sepecific imports
from scipy import sparse
import argparse
from pyoptsparse import Optimization, OPT, SNOPT

from openmdao.api import Problem, Group, IndepVarComp, pyOptSparseDriver, DirectSolver
from openmdao.utils.assert_utils import assert_rel_error

import dymos as dm

input_dict = {'surrogate info full path' : os.environ['HOME'] + '/UserApps/pyStatReduce/pystatreduce/optimize/dymos_interceptor/quadratic_surrogate/surrogate_samples_pseudo_random_0.0001.npz',
              'surrogate_type' : 'quadratic',
             }
systemsize = 45

# Create the distribution
mu = np.zeros(systemsize)
std_dev =  np.array([0.1659134,  0.1659134, 0.16313925, 0.16080975, 0.14363596, 0.09014088, 0.06906912, 0.03601839, 0.0153984 , 0.01194864, 0.00705978, 0.0073889 , 0.00891946,
 0.01195811, 0.01263033, 0.01180144, 0.00912247, 0.00641914, 0.00624566, 0.00636504, 0.0064624 , 0.00639544, 0.0062501 , 0.00636687, 0.00650337, 0.00699955,
 0.00804997, 0.00844582, 0.00942114, 0.01080109, 0.01121497, 0.01204432, 0.0128207 , 0.01295824, 0.01307331, 0.01359864, 0.01408001, 0.01646131, 0.02063841,
 0.02250183, 0.02650464, 0.02733539, 0.02550976, 0.01783919, 0.0125073 , 0.01226541])
jdist = cp.MvNormal(mu, np.diag(std_dev[:-1]))

# Create the QoI object
QoI = InterceptorSurrogateQoI(systemsize, input_dict)
QoI_dict = {'time_duration': {'QoI_func': QoI.eval_QoI,
                              'output_dimensions': 1,}
            }


# Get the dominant directions
dominant_space = DimensionReduction(n_arnoldi_sample=int(sys.argv[1]),
                                    exact_Hessian=False,
                                    sample_radius=1.e-1)
dominant_space.getDominantDirections(QoI, jdist, max_eigenmodes=15)
n_dominant_dir = 10 # int(sys.argv[2])
dominant_dir = dominant_space.iso_eigenvecs[:,0:n_dominant_dir]

# Do the monte carlo integration
nsample = 10000
mc_obj = MonteCarlo(nsample, jdist, QoI_dict, reduced_collocation=False,
                    include_derivs=False)
mc_obj.getSamples(jdist, include_derivs=False)


mu_j = mc_obj.mean(jdist, of=['time_duration'])
var_j = mc_obj.variance(jdist, of=['time_duration'])
# Print everything
print()
print('eigenvals[0:10] = \n', dominant_space.iso_eigenvals[0:10])
print('mean time duration = ', mu_j['time_duration'])
print('std dev time_duration = ', var_j['time_duration'] ** 0.5)
print()
