from __future__ import division, print_function
import os, sys, errno, copy
import warnings

# pyStatReduce specific imports
import numpy as np
import chaospy as cp
import numdifftools as nd
from pystatreduce.new_stochastic_collocation import StochasticCollocation2
from pystatreduce.quantity_of_interest import QuantityOfInterest
from pystatreduce.dimension_reduction import DimensionReduction
from pystatreduce.stochastic_arnoldi.arnoldi_sample import ArnoldiSampling
import pystatreduce.utils as utils
import pystatreduce.examples as examples

#pyoptsparse sepecific imports
from scipy import sparse
import argparse
from pyoptsparse import Optimization, OPT, SNOPT

from openmdao.api import Problem, Group, IndepVarComp, pyOptSparseDriver, DirectSolver
from openmdao.utils.assert_utils import assert_rel_error

import dymos as dm
from smt.surrogate_models import QP, KRG # Surrogate Modeling

class InterceptorSurrogateQoI(QuantityOfInterest):
    """
    Class that creates a surrogate model for the dymos supersonic intercpetor problem
    for analysis
    """
    def __init__(self, systemsize, input_dict, data_type=np.float):
        QuantityOfInterest.__init__(self, systemsize, data_type=data_type)

        # Load the eigenmodes
        fname = input_dict['surrogate info full path']
        surrogate_info = np.load(fname)
        surrogate_samples = surrogate_info['input_samples']
        fval_arr = surrogate_info['fvals']

        # Create the surrogate
        self.surrogate_type = input_dict['surrogate_type']
        if self.surrogate_type == 'quadratic':
            self.surrogate = QP()
        elif self.surrogate_type == 'kriging':
            theta0 = input_dict['kriging_theta']
            self.surrogate = KRG(theta0=[theta0], corr=input_dict['correlation function'])
        else:
            raise NotImplementedError
        self.surrogate.set_training_values(surrogate_samples.T, fval_arr)
        self.surrogate.train()

    def eval_QoI(self, mu, xi):
        rv = mu + xi
        return self.surrogate.predict_values(np.expand_dims(rv, axis=0))

    def eval_QoIGradient(self, mu, xi):
        rv = np.expand_dims(mu + xi, axis=0)
        dfdrv = np.zeros(self.systemsize, dtype=self.data_type)
        for i in range(self.systemsize):
            dfdrv[i] = self.surrogate.predict_derivatives(rv, i)[0,0]

        return dfdrv

    def eval_QoIGradient_fd(self, mu, xi):
        # This function uses numdifftools to compute the gradients. Only use for
        # debugging.
        def func(xi):
            return self.eval_QoI(mu, xi)

        G = nd.Gradient(func)(xi)
        return G

if __name__ == '__main__':

    input_dict = {'surrogate info full path' : os.environ['HOME'] + '/UserApps/pyStatReduce/pystatreduce/optimize/dymos_interceptor/quadratic_surrogate/surrogate_samples_pseudo_random.npz',
                  'surrogate_type' : 'kriging',
                  'kriging_theta' : 1.e-4,
                  'correlation function' : 'squar_exp', # 'abs_exp',
                 }
    systemsize = 45

    # Create the distribution
    mu = np.zeros(systemsize)
    deviations =  np.array([0.1659134,  0.1659134, 0.16313925, 0.16080975, 0.14363596, 0.09014088, 0.06906912, 0.03601839, 0.0153984 , 0.01194864, 0.00705978, 0.0073889 , 0.00891946,
     0.01195811, 0.01263033, 0.01180144, 0.00912247, 0.00641914, 0.00624566, 0.00636504, 0.0064624 , 0.00639544, 0.0062501 , 0.00636687, 0.00650337, 0.00699955,
     0.00804997, 0.00844582, 0.00942114, 0.01080109, 0.01121497, 0.01204432, 0.0128207 , 0.01295824, 0.01307331, 0.01359864, 0.01408001, 0.01646131, 0.02063841,
     0.02250183, 0.02650464, 0.02733539, 0.02550976, 0.01783919, 0.0125073 , 0.01226541])
    jdist = cp.MvNormal(mu, np.diag(deviations[:-1]))

    # Create the QoI object
    QoI = InterceptorSurrogateQoI(systemsize, input_dict)
    QoI_dict = {'time_duration': {'QoI_func': QoI.eval_QoI,
                                  'output_dimensions': 1,}
                }

    # # Check Gradients
    # grad = QoI.eval_QoIGradient(std_dev[:-1], np.zeros(systemsize))
    # grad_fd = QoI.eval_QoIGradient_fd(std_dev[:-1], np.zeros(systemsize))
    # err = abs(grad - grad_fd)
    # print('grad = \n', grad)
    # print('grad_fd = \n', grad_fd)
    # print('err = \n', err)

    # Get the dominant directions
    dominant_space = DimensionReduction(n_arnoldi_sample=int(sys.argv[1]),
                                        exact_Hessian=False,
                                        sample_radius=1.e-6)
    dominant_space.getDominantDirections(QoI, jdist, max_eigenmodes=15)
    n_dominant_dir = int(sys.argv[2])
    dominant_dir = dominant_space.iso_eigenvecs[:,0:n_dominant_dir]

    # Compute statistics
    sc_obj = StochasticCollocation2(jdist, 3, 'MvNormal', QoI_dict,
                                      reduced_collocation=True,
                                      dominant_dir=dominant_dir,
                                      include_derivs=False)
    sc_obj.evaluateQoIs(jdist)

    mu_j = sc_obj.mean(of=['time_duration'])
    var_j = sc_obj.variance(of=['time_duration'])

    print('mean time duration = ', mu_j['time_duration'])
    print('standard deviation time = ', np.sqrt(var_j['time_duration']))
    # print('variance time_duration = ', var_j['time_duration'])
