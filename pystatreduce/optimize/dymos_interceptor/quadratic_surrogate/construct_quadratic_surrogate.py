################################################################################
# construct_quadratic_surrogate.py
#
# The following code constructs a quadratic surrogate for the supersonic
# interceptor problem
################################################################################

import numpy as np
import cmath
import chaospy as cp
import time
import sys
import os

from smt.surrogate_models import QP # Surrogate Modeling

from pystatreduce.new_stochastic_collocation import StochasticCollocation2
from pystatreduce.stochastic_collocation import StochasticCollocation
from pystatreduce.quantity_of_interest import QuantityOfInterest
from pystatreduce.dimension_reduction import DimensionReduction
import pystatreduce.examples as examples
from pystatreduce.examples.supersonic_interceptor.interceptor_rdo2 import DymosInterceptorGlue
import pystatreduce.utils as utils
import pystatreduce.optimize.dymos_interceptor.eigen_information as eigen_info

np.set_printoptions(precision=16)

input_dict = {'num_segments': 15,
              'transcription_order' : 3,
              'transcription_type': 'LGR',
              'solve_segments': False,
              'use_polynomial_control': False}

systemsize = input_dict['num_segments'] * input_dict['transcription_order']

mu = np.zeros(systemsize)
std_dev =  np.array([0.1659134,  0.1659134, 0.16313925, 0.16080975, 0.14363596, 0.09014088, 0.06906912, 0.03601839, 0.0153984 , 0.01194864, 0.00705978, 0.0073889 , 0.00891946,
 0.01195811, 0.01263033, 0.01180144, 0.00912247, 0.00641914, 0.00624566, 0.00636504, 0.0064624 , 0.00639544, 0.0062501 , 0.00636687, 0.00650337, 0.00699955,
 0.00804997, 0.00844582, 0.00942114, 0.01080109, 0.01121497, 0.01204432, 0.0128207 , 0.01295824, 0.01307331, 0.01359864, 0.01408001, 0.01646131, 0.02063841,
 0.02250183, 0.02650464, 0.02733539, 0.02550976, 0.01783919, 0.0125073 , 0.01226541])

start_time = time.time() # Timing

jdist = cp.MvNormal(mu, np.diag(std_dev[:-1]))

default_fpath = os.environ['HOME'] + '/UserApps/pyStatReduce/pystatreduce/optimize/dymos_interceptor/quadratic_surrogate/'
default_fname = default_fpath + 'surrogate_samples_pseudo_random'

generate_data = True
if generate_data:
    # We need to get at least n_surrogate_samples to construct the quadratic surrogate problem
    n_surrogate_samples = int(0.5 * (systemsize + 1) * (systemsize + 2))
    surrogate_samples = jdist.sample(n_surrogate_samples, rule='R')

    # Evaluate the the function at those points
    QoI = DymosInterceptorGlue(systemsize, input_dict)
    fval_arr = np.zeros(n_surrogate_samples)
    for i in range(n_surrogate_samples):
        fval_arr[i] = QoI.eval_QoI(surrogate_samples[:,i], np.zeros(systemsize))

    write_files = True
    if write_files:
        fname = default_fname
        np.savez(fname, input_samples=surrogate_samples, fvals=fval_arr)

# If data doesn't need to be generated, use the existing data or read in the values
read_data = False
if read_data == True or 'fval_arr' not in locals() or 'surrogate_samples' not in locals():
    fname = default_fname
    surrogate_info = np.load(fname + '.npz')
    surrogate_samples = surrogate_info['input_samples']
    fval_arr = surrogate_info['fvals']


# Now that we have either read the inputs or outputs, or run it from scratch,
# lets construct the surrogate
sm = QP() # Surrogate model object
sm.set_training_values(surrogate_samples.T, fval_arr)
sm.train()

df_bar = sm.predict_derivatives(np.expand_dims(std_dev[:-1], axis=0), 1)
print('df_bar at 1 std dev: ', df_bar)
