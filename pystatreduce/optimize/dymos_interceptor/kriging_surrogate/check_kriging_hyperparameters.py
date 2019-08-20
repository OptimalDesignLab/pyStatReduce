from __future__ import division, print_function
import os, sys, errno, copy
import warnings

# pyStatReduce specific imports
import numpy as np
import chaospy as cp
import numdifftools as nd
from pystatreduce.new_stochastic_collocation import StochasticCollocation2
from pystatreduce.monte_carlo import MonteCarlo
from pystatreduce.quantity_of_interest import QuantityOfInterest
from pystatreduce.dimension_reduction import DimensionReduction
from pystatreduce.stochastic_arnoldi.arnoldi_sample import ArnoldiSampling
import pystatreduce.utils as utils
import pystatreduce.examples as examples
from pystatreduce.optimize.dymos_interceptor.quadratic_surrogate.interceptor_surrogate_qoi import InterceptorSurrogateQoI

hyperparam_arr = [1.e-6] # [10.0, 5.0, 3.0, 2.0, 1.0, 1.e-1, 1.e-2, 1.e-3, 1.e-4, 1.e-5]
systemsize = 45

# Create the distribution
mu = np.zeros(systemsize)
density_deviations =  np.array([0.1659134,  0.1659134, 0.16313925, 0.16080975, 0.14363596, 0.09014088, 0.06906912, 0.03601839, 0.0153984 , 0.01194864, 0.00705978, 0.0073889 , 0.00891946,
 0.01195811, 0.01263033, 0.01180144, 0.00912247, 0.00641914, 0.00624566, 0.00636504, 0.0064624 , 0.00639544, 0.0062501 , 0.00636687, 0.00650337, 0.00699955,
 0.00804997, 0.00844582, 0.00942114, 0.01080109, 0.01121497, 0.01204432, 0.0128207 , 0.01295824, 0.01307331, 0.01359864, 0.01408001, 0.01646131, 0.02063841,
 0.02250183, 0.02650464, 0.02733539, 0.02550976, 0.01783919, 0.0125073 , 0.01226541])
jdist = cp.MvNormal(mu, np.diag(density_deviations[:-1]))

mean_history = {}
std_dev_history = {}
for i in hyperparam_arr:
    input_dict = {'surrogate info full path' : os.environ['HOME'] + '/UserApps/pyStatReduce/pystatreduce/optimize/dymos_interceptor/quadratic_surrogate/surrogate_samples_latin_hypercube.npz',
                  'surrogate_type' : 'kriging',
                  'kriging_theta' : i,
                  'correlation function' : 'squar_exp', # 'abs_exp',
                 }
    # Create the QoI object
    QoI = InterceptorSurrogateQoI(systemsize, input_dict)
    QoI_dict = {'time_duration': {'QoI_func': QoI.eval_QoI,
                                  'output_dimensions': 1,}
                }
    # Do the monte carlo integration
    nsample = 10000
    mc_obj = MonteCarlo(nsample, jdist, QoI_dict, reduced_collocation=False,
                        include_derivs=False)
    mc_obj.getSamples(jdist, include_derivs=False)


    mu_j = mc_obj.mean(jdist, of=['time_duration'])
    var_j = mc_obj.variance(jdist, of=['time_duration'])

    mean_history[str(i)] = mu_j['time_duration'][0]
    std_dev_history[str(i)] = np.sqrt(var_j['time_duration'][0])

# Print the final stuff
for i in hyperparam_arr:
    print('theta0 = ', i)
    print('mu_j = ', mean_history[str(i)])
    print('std_dev_j = ', std_dev_history[str(i)])
