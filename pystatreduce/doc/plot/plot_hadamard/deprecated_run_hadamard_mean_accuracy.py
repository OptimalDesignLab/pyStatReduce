################################################################################
#
# deprecated_run_hadamard_mean_accuracy.py
#
# This file is a DEPRECATED run script to generate the data usied in Aviation
# 2018. The data pertains to the accuracy of the mean and
# variance computed using reduced collocation for the Hadamard quadratic.
#
################################################################################

import os
import sys
import errno
# sys.path.insert(0, '../../src')

import numpy as np
import chaospy as cp

from pystatreduce.new_stochastic_collocation import StochasticCollocation2
from pystatreduce.stochastic_collocation import StochasticCollocation
from pystatreduce.quantity_of_interest import QuantityOfInterest
from pystatreduce.dimension_reduction import DimensionReduction
from pystatreduce.stochastic_arnoldi.arnoldi_sample import ArnoldiSampling
import pystatreduce.examples as examples

def run_hadamard(systemsize, eigen_decayrate, std_dev, n_eigenmodes):
    n_collocation_pts = 2

    # Create Hadmard Quadratic object
    QoI = examples.HadamardQuadratic(systemsize, eigen_decayrate)

    # # Create stochastic collocation object
    # collocation = StochasticCollocation(n_collocation_pts, "Normal")

    # Initialize chaospy distribution
    x = np.random.rand(QoI.systemsize)
    jdist = cp.MvNormal(x, np.diag(std_dev))

    threshold_factor = 0.5
    dominant_space = DimensionReduction(threshold_factor=threshold_factor,
                                        exact_Hessian=False,
                                        n_arnoldi_sample=71,
                                        min_eigen_accuracy=1.e-2)
    dominant_space.getDominantDirections(QoI, jdist, max_eigenmodes=n_eigenmodes)

    # if systemsize == 64:
        # print('x = \n', repr(x))
        # print('std_dev = \n', repr(std_dev))
        # print('iso_eigenvals = ', dominant_space.iso_eigenvals)
        # print("dominant_indices = ", dominant_space.dominant_indices)

    # Collocate
    # collocation = StochasticCollocation(n_collocation_pts, "Normal")
    # mu_j = collocation.normal.reduced_mean(QoI.eval_QoI, jdist, dominant_space)
    # print "mu_j = ", mu_j

    QoI_dict = {'Hadamard' : {'QoI_func' : QoI.eval_QoI,
                              'output_dimensions' : 1,
                              },
                }
    sc_obj = StochasticCollocation2(jdist, n_collocation_pts, 'MvNormal',
                                    QoI_dict, include_derivs=False,
                                    reduced_collocation=True,
                                    dominant_dir=dominant_space.dominant_dir)
    sc_obj.evaluateQoIs(jdist)
    mu_j_dict = sc_obj.mean(of=['Hadamard'])
    mu_j = mu_j_dict['Hadamard']

    # Evaluate the analytical value of the Hadamard Quadratic
    covariance = cp.Cov(jdist)
    mu_analytic = QoI.eval_analytical_QoI_mean(x, covariance)
    # print "mu_analytic = ", mu_analytic

    relative_error = np.linalg.norm((mu_j - mu_analytic) / mu_analytic)
    # print "relative_error = ", relative_error

    return relative_error


systemsize_arr = [16, 64, 256] # [16, 32, 64, 128, 256]
eigen_decayrate_arr = [2.0, 1.0, 0.5]
n_eigenmodes_arr = range(1,11)
n_stddev_samples = 10
n_e_sample = len(n_eigenmodes_arr)

err_mu_arr = np.zeros([n_e_sample, n_stddev_samples])
avg_mu_err = np.zeros(n_e_sample)
max_mu_err = np.zeros(n_e_sample)
min_mu_err = np.zeros(n_e_sample)

eigen_decayrate_arr_idx = 2

for i in systemsize_arr:
    for j in range(0, n_e_sample):
        print("systemsize = ", i, ", n_eigenmodes_arr[j] = ", n_eigenmodes_arr[j])
        for k in range(0, n_stddev_samples):
            std_dev = np.random.rand(i)
            # print "systemsize = ", i, ", n_eigenmodes_arr[j] = ", n_eigenmodes_arr[j], "std_dev.size = ", std_dev.size
            if i == 256:
                print("    k = ", k)
            err_mu_arr[j,k] = run_hadamard(i, eigen_decayrate_arr[eigen_decayrate_arr_idx], std_dev,
                                           n_eigenmodes_arr[j])

        avg_mu_err[j] = np.mean(err_mu_arr[j,:])
        max_mu_err[j] = np.max(err_mu_arr[j,:])
        min_mu_err[j] = np.min(err_mu_arr[j,:])

    dirname = ''.join(['./plot_data/mean_accuracy/', str(i), '/'])
    # Create the directory if it doesn't exist
    if not os.path.isdir(dirname):
        try:
            os.makedirs(dirname)
        except OSError as exc:  # Guard against race condition
            if exc.errno != errno.EEXIST:
                raise
    fname1 = ''.join([dirname, 'avg_err_decay', str(eigen_decayrate_arr[eigen_decayrate_arr_idx]), '.txt'])
    fname2 = ''.join([dirname, 'max_err_decay', str(eigen_decayrate_arr[eigen_decayrate_arr_idx]), '.txt'])
    fname3 = ''.join([dirname, 'min_err_decay', str(eigen_decayrate_arr[eigen_decayrate_arr_idx]), '.txt'])

    np.savetxt(fname1, avg_mu_err, delimiter=',')
    np.savetxt(fname2, max_mu_err, delimiter=',')
    np.savetxt(fname3, min_mu_err, delimiter=',')
