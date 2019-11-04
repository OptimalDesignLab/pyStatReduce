################################################################################
# plot_dominant_direction_eigenvals.py
#
# The following file plots the eigenvalues generated from the dominant directions
# of the Hessian
#
################################################################################

import os, sys
import numpy as np
import chaospy as cp
import numdifftools as nd
from pystatreduce.new_stochastic_collocation import StochasticCollocation2
from pystatreduce.monte_carlo import MonteCarlo
from pystatreduce.quantity_of_interest import QuantityOfInterest
from pystatreduce.dimension_reduction import DimensionReduction
from pystatreduce.active_subspace import ActiveSubspace
from pystatreduce.stochastic_arnoldi.arnoldi_sample import ArnoldiSampling
import pystatreduce.utils as utils
import pystatreduce.examples as examples
from pystatreduce.optimize.dymos_interceptor.quadratic_surrogate.interceptor_surrogate_qoi import InterceptorSurrogateQoI

# Plotting imports
from mpl_toolkits import mplot3d
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

systemsize = 45
mu = np.zeros(systemsize)
deviations =  np.array([0.1659134,  0.1659134, 0.16313925, 0.16080975, 0.14363596, 0.09014088, 0.06906912, 0.03601839, 0.0153984 , 0.01194864, 0.00705978, 0.0073889 , 0.00891946,
 0.01195811, 0.01263033, 0.01180144, 0.00912247, 0.00641914, 0.00624566, 0.00636504, 0.0064624 , 0.00639544, 0.0062501 , 0.00636687, 0.00650337, 0.00699955,
 0.00804997, 0.00844582, 0.00942114, 0.01080109, 0.01121497, 0.01204432, 0.0128207 , 0.01295824, 0.01307331, 0.01359864, 0.01408001, 0.01646131, 0.02063841,
 0.02250183, 0.02650464, 0.02733539, 0.02550976, 0.01783919, 0.0125073 , 0.01226541])

generate_data = False
plot_kriging_eignevals = False # Only plots the eigenvalues for the Kriging surrogate where theta = 1e-4
plot_kriging_theta_angles = True

if generate_data:
    # Create the atmospheric joint distribution
    jdist = cp.MvNormal(mu, np.diag(deviations[:-1]))
    # Create the surrogate object
    fname = 'surrogate_samples_pseudo_random.npz' # 'surrogate_samples_pseudo_random_0.1.npz'
    surrogate_input_dict = {'surrogate info full path' : os.environ['HOME'] + '/UserApps/pyStatReduce/pystatreduce/optimize/dymos_interceptor/quadratic_surrogate/' + fname,
                            'surrogate_type' : 'kriging',
                            'kriging_theta' : 1.e-4,
                            'correlation function' : 'squar_exp',
                           }
    surrogate_QoI = InterceptorSurrogateQoI(systemsize, surrogate_input_dict)

    # Get the dominant directions
    dominant_space = DimensionReduction(n_arnoldi_sample=systemsize+1,
                                        exact_Hessian=False,
                                        sample_radius=1.e-1)
    dominant_space.getDominantDirections(surrogate_QoI, jdist, max_eigenmodes=20)

    if plot_kriging_theta_angles:
        surrogate_input_dict2 = {'surrogate info full path' : os.environ['HOME'] + '/UserApps/pyStatReduce/pystatreduce/optimize/dymos_interceptor/quadratic_surrogate/' + fname,
                                'surrogate_type' : 'kriging',
                                'kriging_theta' : 1.e-6,
                                'correlation function' : 'squar_exp',
                               }

        surrogate_QoI2 = InterceptorSurrogateQoI(systemsize, surrogate_input_dict)

        # Get the dominant directions
        dominant_space2 = DimensionReduction(n_arnoldi_sample=systemsize+1,
                                            exact_Hessian=False,
                                            sample_radius=1.e-1)
        dominant_space2.getDominantDirections(surrogate_QoI, jdist, max_eigenmodes=20)


if plot_kriging_eignevals:
    if generate_data:
        eigenvals = dominant_space.iso_eigenvals[0:20]
        print('eigenvals = ', repr(eigenvals))
    else:
        iso_eigenvals = np.array([ 0.62606465, -0.54819897,  0.29980389,  0.180946  ,  0.15643205, -0.1506419 ,  0.10568389,  0.08748472, -0.07912507, -0.05031196, -0.04065365,
                                0.03385065, -0.03184812,  0.02992081, -0.02396936,  0.02360027,  0.01894736,  0.01288077, -0.01128102,  0.01104412,  0.00913591,  0.00698749,
                               -0.00608776, -0.00524267,  0.0049631 , -0.00336075,  0.00303844,  0.00278134, -0.00273262,  0.00254415,  0.00222251, -0.00203982,  0.00190023,
                                0.00175404,  0.00140449,  0.0010735 ,  0.00101895,  0.00090106,  0.00084618, -0.00065268,  0.00062987, -0.00044691,  0.0004279 ,  0.00014922,
                               -0.00003856])

        eigenvals = abs(iso_eigenvals[0:20])

    idx = range(1, eigenvals.size+1)
    fname = 'interceptor_kriging_dominant_eigenvals.pdf'
    fig = plt.figure('eigenvalues', figsize=(7,4))
    gca_var = fig.gca()
    ax = plt.axes()
    s = ax.scatter(idx, eigenvals)
    ax.set_xlabel('eigenvalue index')
    ax.set_ylabel('eigenvalue')
    gca_var.xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.yscale('log')
    plt.ylim(1.e-2, 1.e0)
    plt.tight_layout()
    # plt.show()
    plt.savefig(fname, format='pdf')

if plot_kriging_theta_angles:
    n_bases = 6
    if generate_data:
        eigenvecs_1e_4 = dominant_space.iso_eigenvecs
        eigenvecs_1e_6 = dominant_space2.iso_eigenvecs
        # print('eigenvecs_1e_4 = \n', repr(eigenvecs_1e_4))
        # print('eigenvecs_1e_6 = \n', repr(eigenvecs_1e_6))
        np.savetxt('eigenvecs_1e_4.txt', eigenvecs_1e_4)
        np.savetxt('eigenvecs_1e_6.txt', eigenvecs_1e_6)
    else:
        # Read from file
        eigenvecs_1e_4 = np.loadtxt('eigenvecs_1e_4.txt')
        eigenvecs_1e_6 = np.loadtxt('eigenvecs_1e_4.txt')

    angles_radians = utils.compute_subspace_angles(eigenvecs_1e_4[:,0:n_bases], eigenvecs_1e_6[:,0:n_bases])
    angles_degrees = np.degrees(angles_radians)
    print(angles_degrees)
    """
    # Finally plot
    xvals = range(1, n_bases+1)
    fname = "kriging_angles_1e_4_1e_6.pdf"
    fig = plt.figure('scatter', figsize=(6,5))
    ax = plt.axes()
    s = ax.scatter(xvals, angles_degrees, marker='o')
    ax.set_xlabel('subspace indices')
    ax.set_ylabel('angles (degrees)')
    plt.xticks(xvals)
    plt.tight_layout()
    # plt.show()
    plt.savefig(fname, format='pdf')
    """
