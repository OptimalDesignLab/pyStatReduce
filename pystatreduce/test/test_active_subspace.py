# test_active_subspace.py
import numpy as np
import chaospy as cp
import copy
from pystatreduce.new_stochastic_collocation import StochasticCollocation2
from pystatreduce.quantity_of_interest import QuantityOfInterest
from pystatreduce.dimension_reduction import DimensionReduction
from pystatreduce.active_subspace import ActiveSubspace
from pystatreduce.stochastic_arnoldi.arnoldi_sample import ArnoldiSampling
import pystatreduce.examples as examples
import pystatreduce.optimize.OAS_ScanEagle.oas_scaneagle_opt as scaneagle_opt

systemsize = 4
eigen_decayrate = 2.0

# Create Hadmard Quadratic object
QoI = HadamardQuadratic(systemsize, eigen_decayrate)

# Create the joint distribution
mu = np.random.rand(systemsize)
std_dev = np.diag(np.random.rand(systemsize))
jdist = cp.MvNormal(mu, std_dev)

# Create the active subspace object
active_subspace = ActiveSubspace(n_dominant_dimensions=2, QoI, n_monte_carlo_samples=1000)
active_subspace.getDominantDirections(QoI, jdist)
