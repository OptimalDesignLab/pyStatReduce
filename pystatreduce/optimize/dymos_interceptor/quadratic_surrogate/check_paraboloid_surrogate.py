import numpy as np
import cmath
import chaospy as cp
import time
import sys
import os

from smt.surrogate_models import QP # Surrogate Modeling

from pystatreduce.new_stochastic_collocation import StochasticCollocation2
from pystatreduce.stochastic_collocation import StochasticCollocation
from pystatreduce.quantity_of_interest import QuantityOfInterest
from pystatreduce.dimension_reduction import DimensionReduction
import pystatreduce.examples as examples
import pystatreduce.utils as utils

# Create the actual QoI
systemsize = 3
mu = np.random.randn(3)
std_dev = abs(np.diag(np.random.randn(3)))
jdist = cp.MvNormal(mu, std_dev)
QoI = examples.Paraboloid3D(systemsize)

# Generate the samples for surrogate
n_surrogate_samples = int(0.5 * (systemsize + 1) * (systemsize + 2))
surrogate_samples = jdist.sample(5000, rule='R')

std_dev_vec = np.diagonal(std_dev)
new_samples = utils.check_std_dev_violation(surrogate_samples, mu, std_dev_vec, scale=1.0)
print('orig_shape = ', new_samples.shape)
new_samples = new_samples[:,0:1000]
print('new_shape = ', new_samples.shape)

"""
fval_arr = np.zeros(n_surrogate_samples)
for i in range(n_surrogate_samples):
    fval_arr[i] = QoI.eval_QoI(surrogate_samples[:,i], np.zeros(systemsize))

# Construct the surrogate
sm = QP() # Surrogate model object
sm.set_training_values(surrogate_samples.T, fval_arr)
sm.train()

# Check the gradient values at a random
check_point = np.random.randn(3)
print('check_point = ', check_point)

fval = QoI.eval_QoI(check_point, np.zeros(3))
fval_surrogate = sm.predict_values(np.expand_dims(check_point, axis=0))

grad = QoI.eval_QoIGradient(check_point, np.zeros(3))
grad_surrogate = np.zeros(systemsize)
for i in range(systemsize):
    grad_surrogate[i] = sm.predict_derivatives(np.expand_dims(check_point, axis=0), i)[0,0]

# Print the values
print('fval = ', fval)
print('fval_surrogate = ', fval_surrogate)
print('grad = ', grad)
print('grad_surrogate = ', grad_surrogate)
"""
