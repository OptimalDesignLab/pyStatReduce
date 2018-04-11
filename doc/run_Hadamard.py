import sys
sys.path.insert(0, '../src')

import numpy as np
import chaospy as cp

from stochastic_collocation import StochasticCollocation
from quantity_of_interest import QuantityOfInterest
from dimension_reduction import DimensionReduction
from stochastic_arnoldi.arnoldi_sample import ArnoldiSampling
import examples

def run_hadamard(systemsize, eigen_decayrate, std_dev):
    n_collocation_pts = 2

    # Create Hadmard Quadratic object
    QoI = examples.HadamardQuadratic(systemsize, eigen_decayrate)

    # Create stochastic collocation object
    collocation = StochasticCollocation(n_collocation_pts, "Normal")

    # Initialize chaospy distribution
    x = np.random.rand(QoI.systemsize)
    jdist = cp.MvNormal(x, np.diag(std_dev))

    threshold_factor = 0.9
    dominant_space = DimensionReduction(threshold_factor, exact_Hessian=False)
    dominant_space.getDominantDirections(QoI, jdist)

    print "dominant_indices = ", dominant_space.dominant_indices

    # Collocate
    mu_j = collocation.normal.reduced_mean(QoI, jdist, dominant_space)
    print "mu_j = ", mu_j

    # Evaluate the analytical value of the Hadamard Quadratic
    covariance = cp.Cov(jdist)
    mu_analytic = QoI.eval_analytical_QoI_mean(x, covariance)
    print "mu_analytic = ", mu_analytic

    relative_error = np.linalg.norm((mu_j - mu_analytic) / mu_analytic)
    print "relative_error = ", relative_error


systemsize_arr = [16, 32, 64, 128, 256]
eigen_decayrate_arr = [2.0, 1.0, 0.5]

std_dev = np.random.rand(systemsize_arr[0])
run_hadamard(systemsize_arr[0], eigen_decayrate_arr[2], std_dev)
