# Arbitrary runs:
# The purspose of this file is to test arbitrary python scripts/packages/and
# functions


import numpy as np
import chaospy as cp
import examples
from stochastic_collocation import StochasticCollocation
from quantity_of_interest import QuantityOfInterest
from dimension_reduction import DimensionReduction
from stochastic_arnoldi.arnoldi_sample import ArnoldiSampling

# 1. Check the quadrature
degree = 3
q_hermite, w_hermite = np.polynomial.legendre.leggauss(degree) #np.polynomial.hermite.hermgauss(degree)
univariate_normal = cp.Normal() # STandard normal
nodes, weights = cp.generate_quadrature(2, univariate_normal, rule="G")
print "q_hermite = ", q_hermite
print "nodes = ", nodes
print "w_hermite = ", w_hermite
print "weights = ", weights







"""
systemsize = 2
x = np.random.rand(systemsize) # np.zeros(systemsize)
theta = 0
sigma = np.ones(systemsize) # np.array([0.2, 0.1])
tuple = (theta,)

# Create a Stochastic collocation object
collocation = StochasticCollocation(3, "Normal")

# Create a QoI object using ellpsoid
QoI = examples.Paraboloid2D(systemsize, tuple)

# Create a joint distribution
jdist = cp.MvNormal(x, np.diag(sigma))
"""
