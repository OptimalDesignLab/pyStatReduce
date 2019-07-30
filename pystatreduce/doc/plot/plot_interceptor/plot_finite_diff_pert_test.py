################################################################################
# run_finite_diff_pert_test.py
#
# The following file exists to run and plot the finite difference gradients that
# are generated using the different central difference perturbations considered.
#
################################################################################
import numpy as np
import chaospy as cp
import time, sys, pickle, cmath
import pprint

from mpl_toolkits import mplot3d
import matplotlib
import matplotlib.pyplot as plt

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

perturbation_arr = [1.e-1, 1.e-2, 1.e-3, 1.e-4, 1.e-5, 1.e-6]

generate_data = False
read_data = True
if generate_data == True:
    mu = np.zeros(systemsize)
    density_variations =  np.array([0.1659134,  0.1659134, 0.16313925, 0.16080975, 0.14363596, 0.09014088, 0.06906912, 0.03601839, 0.0153984 , 0.01194864, 0.00705978, 0.0073889 , 0.00891946,
     0.01195811, 0.01263033, 0.01180144, 0.00912247, 0.00641914, 0.00624566, 0.00636504, 0.0064624 , 0.00639544, 0.0062501 , 0.00636687, 0.00650337, 0.00699955,
     0.00804997, 0.00844582, 0.00942114, 0.01080109, 0.01121497, 0.01204432, 0.0128207 , 0.01295824, 0.01307331, 0.01359864, 0.01408001, 0.01646131, 0.02063841,
     0.02250183, 0.02650464, 0.02733539, 0.02550976, 0.01783919, 0.0125073 , 0.01226541])

    std_dev = density_variations[:-1]
    jdist = cp.MvNormal(mu, np.diag(std_dev))

    QoI = DymosInterceptorGlue(systemsize, input_dict)
    grad_dict = {} # dictionary in which gradients are stored
    for i in perturbation_arr:
        index_name = 'grad_' + str(i)
        grad_dict[index_name] = QoI.eval_QoIGradient(mu, np.zeros(systemsize), fd_pert=i)
        print('index_name = ', index_name)
        print(grad_dict[index_name])

    # Save the file
    pickle_out = open("gradients_dict.pickle", "wb")
    pickle.dump(grad_dict, pickle_out)
    pickle_out.close()

if read_data == True or 'grad_dict' not in locals():
    pickle_in = open("gradients_dict.pickle", "rb")
    grad_dict = pickle.load(pickle_in)

print(grad_dict)

grad_ind = range(1,systemsize+1)
fname = "gradient_scatter.pdf"
fig = plt.figure("gradients", figsize=(13,6))
ax = plt.axes()
sp1 = ax.plot(grad_ind, grad_dict['grad_' + str(perturbation_arr[0])], '^', mfc='none', label='1.e-1', lw=2)
sp2 = ax.plot(grad_ind, grad_dict['grad_' + str(perturbation_arr[1])], 'o', mfc='none', label='1.e-2', lw=2)
sp3 = ax.plot(grad_ind, grad_dict['grad_' + str(perturbation_arr[2])], 'v', mfc='none', label='1.e-3', ms=12, lw=2)
sp4 = ax.plot(grad_ind, grad_dict['grad_' + str(perturbation_arr[3])], 'D', mfc='none', label='1.e-4', ms=10, lw=2)
# sp5 = ax.plot(grad_ind, grad_dict['grad_' + str(perturbation_arr[4])], '>', label='1.e-5', ms=12, lw=2)
# sp6 = ax.plot(grad_ind, grad_dict['grad_' + str(perturbation_arr[5])], '<', label='1.e-6', ms=12, lw=2)
ax.set_xlabel('Gradient Index')
ax.set_ylabel('Gradient Value')
plt.xticks(np.arange(45, step=5))
ax.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
# plt.savefig(fname, format='pdf')
