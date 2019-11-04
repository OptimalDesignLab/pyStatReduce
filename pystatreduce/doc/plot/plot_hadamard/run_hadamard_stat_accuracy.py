################################################################################
#
# run_hadamard_stat_accuracy.py
#
# This file is a run script to generate the data which is eventually plotted
# using a different script. The data pertains to the accuracy of the mean and
# variance computed using reduced collocation for the Hadamard quadratic.
#
################################################################################
import os
import sys
import errno

import numpy as np
import chaospy as cp

from pystatreduce.new_stochastic_collocation import StochasticCollocation2
from pystatreduce.stochastic_collocation import StochasticCollocation
from pystatreduce.quantity_of_interest import QuantityOfInterest
from pystatreduce.dimension_reduction import DimensionReduction
import pystatreduce.examples as examples

def run_hadamard(systemsize, eigen_decayrate, std_dev, n_eigenmodes):
    n_collocation_pts = 3

    # Create Hadmard Quadratic object
    QoI = examples.HadamardQuadratic(systemsize, eigen_decayrate)


    # Initialize chaospy distribution
    x = np.random.rand(QoI.systemsize)
    jdist = cp.MvNormal(x, np.diag(std_dev))

    threshold_factor = 1.0
    use_exact_Hessian = False
    if use_exact_Hessian:
        dominant_space = DimensionReduction(threshold_factor=threshold_factor,
                                            exact_Hessian=True)
        dominant_space.getDominantDirections(QoI, jdist)
        dominant_dir = dominant_space.iso_eigenvecs[:,0:n_eigenmodes]
    else:
        dominant_space = DimensionReduction(threshold_factor=threshold_factor,
                                            exact_Hessian=False,
                                            n_arnoldi_sample=71,
                                            min_eigen_accuracy=1.e-2)
        dominant_space.getDominantDirections(QoI, jdist, max_eigenmodes=n_eigenmodes)
        dominant_dir = dominant_space.dominant_dir
    # print "dominant_indices = ", dominant_space.dominant_indices

    # Create stochastic collocation object
    # collocation = StochasticCollocation(n_collocation_pts, "Normal")
    QoI_dict = {'Hadamard' : {'QoI_func' : QoI.eval_QoI,
                              'output_dimensions' : 1,
                              },
                }
    sc_obj = StochasticCollocation2(jdist, n_collocation_pts, 'MvNormal',
                                    QoI_dict, include_derivs=False,
                                    reduced_collocation=True,
                                    dominant_dir=dominant_dir)
    sc_obj.evaluateQoIs(jdist)

    # Collocate
    # mu_j = collocation.normal.reduced_mean(QoI, jdist, dominant_space)
    mu_j = sc_obj.mean(of=['Hadamard'])
    var_j = sc_obj.variance(of=['Hadamard'])
    std_dev_j = np.sqrt(var_j['Hadamard'])
    # print "mu_j = ", mu_j

    # Evaluate the analytical value of the Hadamard Quadratic
    covariance = cp.Cov(jdist)
    mu_analytic = QoI.eval_analytical_QoI_mean(x, covariance)
    var_analytic = QoI.eval_analytical_QoI_variance(x, covariance)
    std_dev_analytic = np.sqrt(var_analytic)
    # print "mu_analytic = ", mu_analytic

    relative_error_mu = np.linalg.norm((mu_j['Hadamard'] - mu_analytic) / mu_analytic)
    relative_err_var = np.linalg.norm((var_j['Hadamard'] - var_analytic) / var_analytic)
    relative_err_std_dev = np.linalg.norm((std_dev_j - std_dev_analytic) / std_dev_analytic)
    # print "relative_error = ", relative_error

    return relative_error_mu, relative_err_var, relative_err_std_dev


systemsize_arr = [16, 64, 256] # [16, 32, 64, 128, 256]
eigen_decayrate_arr = [2.0, 1.0, 0.5]
n_eigenmodes_arr = range(1,11)
n_stddev_samples = 10
n_e_sample = len(n_eigenmodes_arr)

err_mu_arr = np.zeros([n_e_sample, n_stddev_samples])
avg_mu_err = np.zeros(n_e_sample)
max_mu_err = np.zeros(n_e_sample)
min_mu_err = np.zeros(n_e_sample)

err_var_arr = np.zeros([n_e_sample, n_stddev_samples])
avg_var_err = np.zeros(n_e_sample)
max_var_err = np.zeros(n_e_sample)
min_var_err = np.zeros(n_e_sample)

err_std_dev_arr = np.zeros([n_e_sample, n_stddev_samples])
avg_std_dev_err = np.zeros(n_e_sample)
max_std_dev_err = np.zeros(n_e_sample)
min_std_dev_err = np.zeros(n_e_sample)

eigen_decayrate_arr_idx = 0
print('eigen decayrate = ', eigen_decayrate_arr[eigen_decayrate_arr_idx])

for i in systemsize_arr:
    for j in range(0, n_e_sample):
        print("systemsize = ", i, ", n_eigenmodes_arr[j] = ", n_eigenmodes_arr[j])
        for k in range(0, n_stddev_samples):
            std_dev = abs(np.random.rand(i))
            # print "systemsize = ", i, ", n_eigenmodes_arr[j] = ", n_eigenmodes_arr[j], "std_dev.size = ", std_dev.size
            # if i == 256:
            #     print("    k = ", k)
            err_mu_arr[j,k], err_var_arr[j,k], err_std_dev_arr[j,k] = run_hadamard(i, eigen_decayrate_arr[eigen_decayrate_arr_idx], std_dev,
                                           n_eigenmodes_arr[j])

        avg_mu_err[j] = np.mean(err_mu_arr[j,:])
        max_mu_err[j] = np.max(err_mu_arr[j,:])
        min_mu_err[j] = np.min(err_mu_arr[j,:])

        avg_var_err[j] = np.mean(err_var_arr[j,:])
        max_var_err[j] = np.max(err_var_arr[j,:])
        min_var_err[j] = np.min(err_var_arr[j,:])

        avg_std_dev_err[j] = np.mean(err_std_dev_arr[j,:])
        max_std_dev_err[j] = np.max(err_std_dev_arr[j,:])
        min_std_dev_err[j] = np.min(err_std_dev_arr[j,:])

    dirname_mu = ''.join(['./plot_data/mean_accuracy/', str(i), '/'])
    dirname_var = ''.join(['./plot_data/variance_accuracy/', str(i), '/'])
    dirname_std_dev = ''.join(['./plot_data/std_dev_accuracy/', str(i), '/'])
    # Create the directory if it doesn't exist
    if not os.path.isdir(dirname_mu):
        try:
            os.makedirs(dirname_mu)
        except OSError as exc:  # Guard against race condition
            if exc.errno != errno.EEXIST:
                raise

    if not os.path.isdir(dirname_var):
        try:
            os.makedirs(dirname_var)
        except OSError as exc:  # Guard against race condition
            if exc.errno != errno.EEXIST:
                raise

    fname1 = ''.join([dirname_mu, 'avg_err_decay', str(eigen_decayrate_arr[eigen_decayrate_arr_idx]), '.txt'])
    fname2 = ''.join([dirname_mu, 'max_err_decay', str(eigen_decayrate_arr[eigen_decayrate_arr_idx]), '.txt'])
    fname3 = ''.join([dirname_mu, 'min_err_decay', str(eigen_decayrate_arr[eigen_decayrate_arr_idx]), '.txt'])

    fname4 = ''.join([dirname_var, 'avg_err_decay', str(eigen_decayrate_arr[eigen_decayrate_arr_idx]), '.txt'])
    fname5 = ''.join([dirname_var, 'max_err_decay', str(eigen_decayrate_arr[eigen_decayrate_arr_idx]), '.txt'])
    fname6 = ''.join([dirname_var, 'min_err_decay', str(eigen_decayrate_arr[eigen_decayrate_arr_idx]), '.txt'])

    fname7 = ''.join([dirname_std_dev, 'avg_err_decay', str(eigen_decayrate_arr[eigen_decayrate_arr_idx]), '.txt'])
    fname8 = ''.join([dirname_std_dev, 'max_err_decay', str(eigen_decayrate_arr[eigen_decayrate_arr_idx]), '.txt'])
    fname9 = ''.join([dirname_std_dev, 'min_err_decay', str(eigen_decayrate_arr[eigen_decayrate_arr_idx]), '.txt'])

    np.savetxt(fname1, avg_mu_err, delimiter=',')
    np.savetxt(fname2, max_mu_err, delimiter=',')
    np.savetxt(fname3, min_mu_err, delimiter=',')

    # np.savetxt(fname4, avg_var_err, delimiter=',')
    # np.savetxt(fname5, max_var_err, delimiter=',')
    # np.savetxt(fname6, min_var_err, delimiter=',')

    # print('avg_mu_err = ', avg_mu_err)
    # print('max_mu_err = ', max_mu_err)
    # print('min_mu_err = ', min_mu_err, '\n')
    # print('avg_var_err = ', avg_var_err)
    # print('max_var_err = ', max_var_err)
    # print('min_var_err = ', min_var_err)

    np.savetxt(fname7, avg_std_dev_err, delimiter=',')
    np.savetxt(fname8, max_std_dev_err, delimiter=',')
    np.savetxt(fname9, min_std_dev_err, delimiter=',')
