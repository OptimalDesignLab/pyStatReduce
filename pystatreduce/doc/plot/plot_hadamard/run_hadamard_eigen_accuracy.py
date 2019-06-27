# run_hadamard_eigen_accuracy
import os
import sys
import errno

import numpy as np
import chaospy as cp

from pystatreduce.stochastic_collocation import StochasticCollocation
from pystatreduce.quantity_of_interest import QuantityOfInterest
from pystatreduce.dimension_reduction import DimensionReduction
import pystatreduce.examples as examples

def run_hadamard(systemsize, eigen_decayrate, std_dev, n_sample):
    # n_collocation_pts = 2

    # Create Hadmard Quadratic object
    QoI = examples.HadamardQuadratic(systemsize, eigen_decayrate)

    # Create stochastic collocation object
    # collocation = StochasticCollocation(n_collocation_pts, "Normal")

    # Initialize chaospy distribution
    x = np.random.randn(QoI.systemsize)
    jdist = cp.MvNormal(x, np.diag(std_dev))

    threshold_factor = 0.5
    dominant_space_exact = DimensionReduction(threshold_factor=threshold_factor,
                                              exact_Hessian=True)
    dominant_space = DimensionReduction(threshold_factor=threshold_factor,
                                        exact_Hessian=False,
                                        n_arnoldi_sample=n_sample)

    dominant_space.getDominantDirections(QoI, jdist, max_eigenmodes=20)
    dominant_space_exact.getDominantDirections(QoI, jdist)

    # Sort the exact eigenvalues in descending order
    sort_ind = dominant_space_exact.iso_eigenvals.argsort()[::-1]

    # Compare the eigenvalues of the 10 most dominant spaces
    lambda_exact = dominant_space_exact.iso_eigenvals[sort_ind]
    error_arr = dominant_space.iso_eigenvals[0:10] - lambda_exact[0:10]
    # print 'error_arr = ', error_arr
    rel_error_norm = np.linalg.norm(error_arr) / np.linalg.norm(lambda_exact[0:10])

    return rel_error_norm

systemsize_arr = [64, 128, 256]
eigen_decayrate_arr = [2.0, 1.0, 0.5]
n_arnoldi_samples_arr = [11, 21, 31, 41, 51]
n_stddev_samples = 10

eigen_decayrate_arr_idx = 0

err_arr = np.zeros([len(n_arnoldi_samples_arr), n_stddev_samples])
avg_err = np.zeros(len(n_arnoldi_samples_arr))
max_err = np.zeros(len(n_arnoldi_samples_arr))
min_err = np.zeros(len(n_arnoldi_samples_arr))

for eigen_decayrate_arr_idx in range(0, len(eigen_decayrate_arr)):
    for i in systemsize_arr:
        for j in range(0, len(n_arnoldi_samples_arr)):
            print('decay rate = ', eigen_decayrate_arr[eigen_decayrate_arr_idx]
                    ,', systemsize = ', i, ', arnoldi samples = ', n_arnoldi_samples_arr[j])
            for k in range(0, n_stddev_samples):
                std_dev = abs(np.random.randn(i))
                err_arr[j,k] = run_hadamard(i, eigen_decayrate_arr[eigen_decayrate_arr_idx],
                                std_dev, n_arnoldi_samples_arr[j])
                # print 'error_norm = ', error_norm
                # sys.exit()
            avg_err[j] = np.mean(err_arr[j,:])
            max_err[j] = np.max(err_arr[j,:])
            min_err[j] = np.min(err_arr[j,:])

        dirname = ''.join(['./plot_data/eigen_accuracy/', str(i), '/'])
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

        np.savetxt(fname1, avg_err, delimiter=',')
        np.savetxt(fname2, max_err, delimiter=',')
        np.savetxt(fname3, min_err, delimiter=',')
