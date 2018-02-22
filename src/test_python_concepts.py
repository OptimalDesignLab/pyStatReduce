# Arbitrary runs:
# The purspose of this file is to test arbitrary python scripts/packages/and
# functions


import numpy as np
import chaospy as cp
from stochastic_collocation import StochasticCollocation
from quantity_of_interest import QuantityOfInterest
import examples

systemsize = 2
x = np.zeros(systemsize)
theta = 0
sigma = np.array([0.2, 0.1])
tuple = (theta,)

# Create a Stochastic collocation object
collocation = StochasticCollocation(3, "Normal")

# Create a QoI object using ellpsoid
QoI = examples.Paraboloid2D(systemsize, tuple)

mu_j = collocation.normal.mean(x, sigma, QoI)
print("mu_j = ", mu_j)

# Analytical value
sys_mat = np.diag([50,1])
cov_mat = np.diag([sigma[0]*sigma[0], sigma[1]*sigma[1]])
mu_j_analytical = np.trace(np.matmul(sys_mat, cov_mat)) + x.dot(sys_mat.dot(x))
print("mu_j_analytical = ", mu_j_analytical)
