# plot normal quadratic distribution overlay
# Plots the effects of theta in the 2D quadratic by showing the normal
# distribution on top of the regular quadratic

import numpy as np
from mpl_toolkits import mplot3d
import matplotlib
import matplotlib.pyplot as plt

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

# Quadratic object
mu = np.zeros(2)
sigma = np.array([2, 0.1]) # np.ones(2)
theta = np.pi/3
tuple = (theta,)
jdist = cp.MvNormal(mu, np.diag(sigma))
QoI = examples.Paraboloid2D(2, tuple)

# Plot the normal distribution
nx = 100
xi_1 = np.linspace(-3,3,nx)
xi_2 = np.linspace(-3,3,nx)
xi = np.zeros(2)
probability_density = np.zeros([nx,nx])
J_xi = np.zeros([nx, nx])
for i in xrange(0, nx):
    for j in xrange(0,nx):
        xi[:] = np.array([xi_1[i], xi_2[j]])
        probability_density[j,i] = jdist.pdf(xi)
        J_xi[j,i] = QoI.eval_QoI(mu, xi)


# Plot the distribution
# fname = "./pdfs/stadard_quadratic-distribution_0.pdf"
fname = "./pdfs/2_01_quadratic-distribution_60.pdf"
plt.rc('text', usetex=True)
matplotlib.rcParams['mathtext.fontset'] = 'cm'
fig = plt.figure("probability_distribution", figsize=(6,6))
ax = plt.axes()
cp1 = ax.contour(xi_1, xi_2, probability_density, colors="red", linewidths=0.5)
cp2 = ax.contour(xi_1, xi_2, J_xi, levels=[2,4,8,16,32,64,128,256, 512], colors="black", linewidths=0.5)
# ax.clabel(cp, inline=1, fmt='%1.1f', fontsize=8)
lines = [cp1.collections[0], cp2.collections[0]]
labels = ['Probability density', 'Quantity of Interest']
ax.set_xlabel(r'$\xi_1$', fontsize=16)
ax.set_ylabel(r'$\xi_2$', fontsize=16)
ax.tick_params(axis='both', labelsize=16)
plt.legend(lines, labels, fontsize=16)
plt.tight_layout()
# plt.show()
fig.savefig(fname, format="pdf")
