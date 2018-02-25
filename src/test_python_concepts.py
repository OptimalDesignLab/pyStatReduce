# Arbitrary runs:
# The purspose of this file is to test arbitrary python scripts/packages/and
# functions


import numpy as np
import chaospy as cp
from stochastic_collocation import StochasticCollocation
from quantity_of_interest import QuantityOfInterest
import examples

systemsize = 2
x = np.random.rand(systemsize) # np.zeros(systemsize)
theta = 0
sigma = np.array([0.2, 0.1])
tuple = (theta,)

# Create a Stochastic collocation object
collocation = StochasticCollocation(3, "Normal")

# Create a QoI object using ellpsoid
QoI = examples.Paraboloid2D(systemsize, tuple)

mu_j = collocation.normal.mean(x, sigma, QoI)
print("mu_j = ", mu_j)

# Analytical mean value
sys_mat = np.diag([50,1])
cov_mat = np.diag([sigma[0]*sigma[0], sigma[1]*sigma[1]])
print "cov_mat", cov_mat
mu_j_analytical = np.trace(np.matmul(sys_mat, cov_mat)) + x.dot(sys_mat.dot(x))
print("mu_j_analytical = ", mu_j_analytical)

jdist = cp.MvNormal(x, np.diag(sigma))
sigma_j = collocation.normal.variance(QoI, jdist, mu_j)
# Analytical variance value
mat1 = np.matmul(sys_mat, cov_mat)
mat2 = np.matmul(sys_mat, np.matmul(cov_mat, sys_mat.dot(x)))
variance_analytical = 2*np.trace(np.matmul(mat1, mat1)) + 4*x.dot(mat2)
print "variance_analytical = ", variance_analytical
print "variance numerical = ", sigma_j
