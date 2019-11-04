################################################################################
# interceptor_statistics.py
#
# This file contains the script for evaluating the mean and standard deviation
# for time duration for the supersonic interceptor trajectory analysis using
# reduced stochastic collocation.
#
################################################################################

import numpy as np
import cmath
import chaospy as cp
import time, sys, os

from pystatreduce.new_stochastic_collocation import StochasticCollocation2
from pystatreduce.stochastic_collocation import StochasticCollocation
from pystatreduce.quantity_of_interest import QuantityOfInterest
from pystatreduce.dimension_reduction import DimensionReduction
import pystatreduce.examples as examples
from pystatreduce.examples.supersonic_interceptor.interceptor_rdo2 import DymosInterceptorGlue
import pystatreduce.utils as utils
import pystatreduce.optimize.dymos_interceptor.eigen_information as eigen_info
from pystatreduce.optimize.dymos_interceptor.quadratic_surrogate.interceptor_surrogate_qoi import InterceptorSurrogateQoI

np.set_printoptions(precision=16)

input_dict = {'num_segments': 15,
              'transcription_order' : 3,
              'transcription_type': 'LGR',
              'solve_segments': False,
              'use_for_collocation' : False,
              'n_collocation_samples': 20,
              'use_polynomial_control': False}

systemsize = input_dict['num_segments'] * input_dict['transcription_order']

mu = np.zeros(systemsize)
deviations =  np.array([0.1659134,  0.1659134, 0.16313925, 0.16080975, 0.14363596, 0.09014088, 0.06906912, 0.03601839, 0.0153984 , 0.01194864, 0.00705978, 0.0073889 , 0.00891946,
 0.01195811, 0.01263033, 0.01180144, 0.00912247, 0.00641914, 0.00624566, 0.00636504, 0.0064624 , 0.00639544, 0.0062501 , 0.00636687, 0.00650337, 0.00699955,
 0.00804997, 0.00844582, 0.00942114, 0.01080109, 0.01121497, 0.01204432, 0.0128207 , 0.01295824, 0.01307331, 0.01359864, 0.01408001, 0.01646131, 0.02063841,
 0.02250183, 0.02650464, 0.02733539, 0.02550976, 0.01783919, 0.0125073 , 0.01226541])

start_time = time.time()

jdist = cp.MvNormal(mu, np.diag(deviations[:-1]))

QoI = DymosInterceptorGlue(systemsize, input_dict)

QoI_dict = {'time_duration': {'QoI_func': QoI.eval_QoI,
                              'output_dimensions': 1,}
            }

use_surrogate_eigenmodes = False
if use_surrogate_eigenmodes:
    fname = 'surrogate_samples_pseudo_random.npz' # 'surrogate_samples_pseudo_random_0.1.npz'
    surrogate_input_dict = {'surrogate info full path' : os.environ['HOME'] + '/UserApps/pyStatReduce/pystatreduce/optimize/dymos_interceptor/quadratic_surrogate/' + fname,
                            'surrogate_type' : 'kriging',
                            'kriging_theta' : 1.e-4,
                            'correlation function' : 'squar_exp',
                           }
    surrogate_QoI = InterceptorSurrogateQoI(systemsize, surrogate_input_dict)
    # Get the dominant directions
    n_arnoldi_sample = 10
    dominant_space = DimensionReduction(n_arnoldi_sample=n_arnoldi_sample, # systemsize+1,
                                        exact_Hessian=False,
                                        sample_radius=1.e-1)
    dominant_space.getDominantDirections(surrogate_QoI, jdist, max_eigenmodes=10)
    n_dominant_dir = 6
    dominant_dir = dominant_space.iso_eigenvecs[:,0:n_dominant_dir]
    # print('iso_eigenvals = \n', dominant_space.iso_eigenvals[0:6])
    # print('dominant_space.dominant_dir.shape = ', dominant_space.dominant_dir.shape)
else:
    # Read in the eigenmodes
    # arnoldi_sample_sizes = [20, 25, 30, 35, 40, 46]
    fname = './eigenmodes/eigenmodes_' + sys.argv[1] + '_samples.npz'
    # fname = './eigenmodes/eigenmodes_' + sys.argv[1] + '_samples_1e_1.npz'
    eigenmode = np.load(fname)
    eigenvecs = eigenmode['eigenvecs']
    n_dominant_dir = int(sys.argv[2]) # 11
    dominant_dir = eigenvecs[:,0:n_dominant_dir]

use_surrogate_qoi_obj = False
if use_surrogate_qoi_obj:
    surrogate_QoI_dict = {'time_duration': {'QoI_func': surrogate_QoI.eval_QoI,
                                            'output_dimensions': 1,}
                         }
    # Create the stochastic collocation object
    sc_obj = StochasticCollocation2(jdist, 3, 'MvNormal', surrogate_QoI_dict,
                                    reduced_collocation=True,
                                    dominant_dir=dominant_dir,
                                    include_derivs=False)
    sc_obj.evaluateQoIs(jdist)

else:
    # Create the stochastic collocation object
    sc_obj = StochasticCollocation2(jdist, 2, 'MvNormal', QoI_dict,
                                      reduced_collocation=True,
                                      dominant_dir=dominant_dir,
                                      include_derivs=False)
    sc_obj.evaluateQoIs(jdist)

evalutation_time = time.time() - start_time

mu_j = sc_obj.mean(of=['time_duration'])
var_j = sc_obj.variance(of=['time_duration'])

final_time = time.time() - start_time

if use_surrogate_eigenmodes == True:
    print('surrogate type = ', surrogate_input_dict['surrogate_type'])
    print('surrogate_sample = ', fname)
    print('kriging_theta = ', surrogate_input_dict['kriging_theta'])
    print('\nn_arnoldi_sample = ', n_arnoldi_sample)
else:
    print('\nn_arnoldi_sample = ', sys.argv[1])
print('mean time duration = ', mu_j['time_duration'])
print('variance time_duration = ', var_j['time_duration'])
print('standard deviation time_duration = ', np.sqrt(var_j['time_duration']))

# print the time elapsed
print('evalutation_time = ', evalutation_time)
print('total_time = ', final_time)
