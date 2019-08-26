################################################################################
# interceptor_statistics_kriging.py
#
# This file contains the script for evaluating the mean and standard deviation
# for time duration for the supersonic interceptor trajectory analysis using
# reduced stochastic collocation and ONLY the Kriging Surrogate.
#
################################################################################

import numpy as np
import cmath
import chaospy as cp
import time, sys, os
np.set_printoptions(precision=16)

from pystatreduce.new_stochastic_collocation import StochasticCollocation2
from pystatreduce.quantity_of_interest import QuantityOfInterest
from pystatreduce.dimension_reduction import DimensionReduction
from pystatreduce.active_subspace import ActiveSubspace
from pystatreduce.optimize.dymos_interceptor.quadratic_surrogate.interceptor_surrogate_qoi import InterceptorSurrogateQoI

systemsize = 45

# Create the distribution
mu = np.zeros(systemsize)
density_deviations =  np.array([0.1659134,  0.1659134, 0.16313925, 0.16080975, 0.14363596, 0.09014088, 0.06906912, 0.03601839, 0.0153984 , 0.01194864, 0.00705978, 0.0073889 , 0.00891946,
 0.01195811, 0.01263033, 0.01180144, 0.00912247, 0.00641914, 0.00624566, 0.00636504, 0.0064624 , 0.00639544, 0.0062501 , 0.00636687, 0.00650337, 0.00699955,
 0.00804997, 0.00844582, 0.00942114, 0.01080109, 0.01121497, 0.01204432, 0.0128207 , 0.01295824, 0.01307331, 0.01359864, 0.01408001, 0.01646131, 0.02063841,
 0.02250183, 0.02650464, 0.02733539, 0.02550976, 0.01783919, 0.0125073 , 0.01226541])
jdist = cp.MvNormal(mu, np.diag(density_deviations[:-1]))

# Create the surrogate object
fname = 'surrogate_samples_pseudo_random.npz'
surrogate_input_dict = {'surrogate info full path' : os.environ['HOME'] + '/UserApps/pyStatReduce/pystatreduce/optimize/dymos_interceptor/quadratic_surrogate/' + fname,
                        'surrogate_type' : 'kriging',
                        'kriging_theta' : 1.e-6,
                        'correlation function' : 'squar_exp',
                       }
surrogate_QoI = InterceptorSurrogateQoI(systemsize, surrogate_input_dict)

use_dominant_directions = False
use_active_subspace = True
if use_dominant_directions:
    # Get the dominant directions
    dominant_space = DimensionReduction(n_arnoldi_sample=systemsize+1,
                                        exact_Hessian=False,
                                        sample_radius=1.e-1)
    dominant_space.getDominantDirections(surrogate_QoI, jdist, max_eigenmodes=10)
    dominant_dir = dominant_space.dominant_dir # dominant_space.iso_eigenvecs[:,0:n_dominant_dir]
elif use_active_subspace:
    active_subspace = ActiveSubspace(surrogate_QoI,
                                    n_dominant_dimensions=20,
                                    n_monte_carlo_samples=1000,
                                    read_rv_samples=False,
                                    use_svd=True,
                                    use_iso_transformation=True)
    active_subspace.getDominantDirections(surrogate_QoI, jdist)
    n_dim = 2
    dominant_dir = active_subspace.iso_eigenvecs[:,0:n_dim]

# Create the stochastic collocation object
surrogate_QoI_dict = {'time_duration': {'QoI_func': surrogate_QoI.eval_QoI,
                                        'output_dimensions': 1,}
                     }
sc_obj = StochasticCollocation2(jdist, 3, 'MvNormal', surrogate_QoI_dict,
                                  reduced_collocation=True,
                                  dominant_dir=dominant_dir,
                                  include_derivs=False)
sc_obj.evaluateQoIs(jdist)

mu_j = sc_obj.mean(of=['time_duration'])
var_j = sc_obj.variance(of=['time_duration'])

# Monte Carlo, 10,000 samples result
mu_t_f = 323.898894360652
sigma_t_f = 5.56030668305056

err_mu = abs((mu_j['time_duration'][0] - mu_t_f) / mu_t_f) * 100
err_sigma = abs((np.sqrt(var_j['time_duration'][0,0]) - sigma_t_f) / sigma_t_f) * 100

print('surrogate type = ', surrogate_input_dict['surrogate_type'])
print('surrogate_sample = ', fname)
print('kriging_theta = ', surrogate_input_dict['kriging_theta'])
print('use_dominant_directions = ', use_dominant_directions)
print('use_active_subspace = ', use_active_subspace)

print('\nmean time duration = ', mu_j['time_duration'])
print('err mu = ', err_mu)
# print('variance time_duration = ', var_j['time_duration'])
print('standard deviation time_duration = ', np.sqrt(var_j['time_duration']))
print('err_sigma = ', err_sigma)
