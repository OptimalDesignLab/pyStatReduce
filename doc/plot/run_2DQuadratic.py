# run_hadamard
import os
import sys
import errno
sys.path.insert(0, '../../src')

import numpy as np
import chaospy as cp

from stochastic_collocation import StochasticCollocation
from quantity_of_interest import QuantityOfInterest
from dimension_reduction import DimensionReduction
from stochastic_arnoldi.arnoldi_sample import ArnoldiSampling
import examples


def run2DQuadratic(theta, std_dev, nx):

    systemsize = 2
    # nx = 100
    xlow = -2*np.ones(systemsize)
    xupp = 2*np.ones(systemsize)
    x1 = np.linspace(xlow[0], xupp[0], num=nx)
    x2 = np.linspace(xlow[1], xupp[1], num=nx)
    error_mu_j = np.zeros((nx,nx))

    threshold_factor = 0.9
    tuple = (theta,)

    for i in xrange(0, nx):
        for j in xrange(0, nx):
            x = np.array([x1[i], x2[j]])

            # Create necessary objects
            collocation = StochasticCollocation(3, "Normal")
            QoI = examples.Paraboloid2D(systemsize, tuple)
            jdist = cp.MvNormal(x, np.diag(std_dev))
            dominant_space = DimensionReduction(threshold_factor, exact_Hessian=True)

            # Get dominant directions and perform reduced collocation
            dominant_space.getDominantDirections(QoI, jdist)
            mu_j_bar = collocation.normal.reduced_mean(QoI, jdist, dominant_space)

            # Check agaisnt full stochastic collocation
            mu_j = collocation.normal.mean(x, std_dev, QoI)

            error_mu_j[i,j] = abs((mu_j_bar - mu_j)/mu_j)

    max_err = np.amax(error_mu_j)

    return max_err

def test_orientation():
    n_theta = 11
    n_samples = 100
    max_err = np.zeros(n_theta)
    theta = np.linspace(0, 90, num=n_theta)

    # same standard deviation
    std_dev = np.array([0.1, 0.1])
    print "standard deviation = ", std_dev
    for i in xrange(0, n_theta): # xrange(0, n_theta):
        theta_rad = theta[i]*np.pi/180
        print "  theta = ", theta[i]
        max_err[i] = run2DQuadratic(theta_rad, std_dev, n_samples)

    print "max_err = ", max_err

    fname = "max_err_01_01.txt"
    np.savetxt(fname, max_err, delimiter=',')


def test_sigmaRatio():
    n_theta = 6
    n_samples = 50
    n_ratios = 16
    max_err = np.zeros((n_theta, n_ratios))
    theta = np.linspace(0, 90, num=n_theta)
    std_dev_ratios = np.linspace(1, 20, num=n_ratios)
    sigma2 = 0.1
    for i in xrange(0, n_theta):
        theta_rad = theta[i]*np.pi/180
        print "theta = ", theta[i]
        for j in xrange(0, n_ratios):
            sigma1 = std_dev_ratios[j]*sigma2
            std_dev = np.array([sigma1, sigma2])
            print "  std_dev = ", std_dev
            # max_err[i,j] = run2DQuadratic(theta_rad, std_dev, n_samples)

    # np.savetxt("max_err_sigma_ratio.txt", max_err)

test_sigmaRatio()
# test_orientation()
