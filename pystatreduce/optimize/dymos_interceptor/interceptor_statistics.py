import numpy as np
import cmath
import chaospy as cp

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
              'use_for_collocation' : False,
              'n_collocation_samples': 20,
              'use_polynomial_control': False}
systemsize = input_dict['num_segments'] * input_dict['transcription_order']

mu = np.zeros(systemsize)
std_dev =  np.array([0.1659134,  0.1659134, 0.16313925, 0.16080975, 0.14363596, 0.09014088, 0.06906912, 0.03601839, 0.0153984 , 0.01194864, 0.00705978, 0.0073889 , 0.00891946,
 0.01195811, 0.01263033, 0.01180144, 0.00912247, 0.00641914, 0.00624566, 0.00636504, 0.0064624 , 0.00639544, 0.0062501 , 0.00636687, 0.00650337, 0.00699955,
 0.00804997, 0.00844582, 0.00942114, 0.01080109, 0.01121497, 0.01204432, 0.0128207 , 0.01295824, 0.01307331, 0.01359864, 0.01408001, 0.01646131, 0.02063841,
 0.02250183, 0.02650464, 0.02733539, 0.02550976, 0.01783919, 0.0125073 , 0.01226541])


# 0.04*np.eye(systemsize) # 0.04* np.random.rand(systemsize)
jdist = cp.MvNormal(mu, np.diag(std_dev[:-1]))

QoI = DymosInterceptorGlue(systemsize, input_dict)


QoI_dict = {'time_duration': {'QoI_func': QoI.eval_QoI,
                              'output_dimensions': 1,}
            }

dominant_dir = eigen_info.eigenvecs_atmos_dev[:,0:6]

# Create the stochastic collocation object
sc_obj = StochasticCollocation2(jdist, 3, 'MvNormal', QoI_dict,
                                  reduced_collocation=True,
                                  dominant_dir=dominant_dir,
                                  include_derivs=False)
sc_obj.evaluateQoIs(jdist)
mu_j = sc_obj.mean(of=['time_duration'])
var_j = sc_obj.variance(of=['time_duration'])
print('mean time duration = ', mu_j['time_duration'])
print('variance time_duration = ', var_j['time_duration'])
