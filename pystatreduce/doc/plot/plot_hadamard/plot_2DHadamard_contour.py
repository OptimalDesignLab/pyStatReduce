# Plot the contours of a 2D Hadamard Quadratic. This figure has 3 subplots,
# which describe the effect of different eigenvalue decay rates on the contours
# of the Hadamard quadratic

import os
import sys
import errno
sys.path.insert(0, '../../src')

import numpy as np
from mpl_toolkits import mplot3d
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import chaospy as cp

from stochastic_collocation import StochasticCollocation
from quantity_of_interest import QuantityOfInterest
from dimension_reduction import DimensionReduction
from stochastic_arnoldi.arnoldi_sample import ArnoldiSampling
import examples

systemsize = 2
decay_rate_arr = np.array([2,1,0.5])
QoI1 = examples.HadamardQuadratic(systemsize, decay_rate_arr[0])
QoI2 = examples.HadamardQuadratic(systemsize, decay_rate_arr[1])
QoI3 = examples.HadamardQuadratic(systemsize, decay_rate_arr[2])

nx = 100
xlow = -2*np.ones(systemsize)
xupp = 2*np.ones(systemsize)
x1 = np.linspace(xlow[0], xupp[0], num=nx)
x2 = np.linspace(xlow[1], xupp[1], num=nx)
J_xi1 = np.zeros([nx,nx])
J_xi2 = np.zeros([nx,nx])
J_xi3 = np.zeros([nx,nx])
pert = np.zeros(systemsize)

for i in xrange(0, nx):
    for j in xrange(0, nx):
        x = np.array([x1[i], x2[j]])
        J_xi1[j,i] = QoI1.eval_QoI(x, pert)
        J_xi2[j,i] = QoI2.eval_QoI(x, pert)
        J_xi3[j,i] = QoI3.eval_QoI(x, pert)

# Plot
plt.rc('text', usetex=True)
matplotlib.rcParams['mathtext.fontset'] = 'cm'
props = dict(boxstyle='round', facecolor='white')

fname = "./pdfs/hadamard_contours.pdf"
f, axes = plt.subplots(1,3, sharey=True , figsize=(10,4))
plt.setp(axes, yticks=[-2,-1,0,1,2])
cp1 = axes[0].contour(x1, x2, J_xi1, cmap="coolwarm", linewidths=0.5)
# axes[0].clabel(cp, inline=1, fmt='%1.1f', fontsize=8)
axes[0].set_xlabel(r'$\xi_1$', fontsize=16)
axes[0].set_ylabel(r'$\xi_2$', fontsize=16)
axes[0].text(0.5,1,r'$\lambda_i = \frac{1}{i^2}$', size=18, bbox=props, \
              transform=axes[0].transAxes, horizontalalignment='center', \
              verticalalignment='center')

cp2 = axes[1].contour(x1, x2, J_xi2, cmap="coolwarm", linewidths=0.5)
# axes[1].clabel(cp, inline=1, fmt='%1.1f', fontsize=8)
axes[1].set_xlabel(r'$\xi_1$', fontsize=16)
axes[1].set_ylabel(r'$\xi_2$', fontsize=16)
axes[1].text(0.5,1,r'$\lambda_i = \frac{1}{i}$', size=18, bbox=props, \
              transform=axes[1].transAxes, horizontalalignment='center', \
              verticalalignment='center')

cp3 = axes[2].contour(x1, x2, J_xi3, cmap="coolwarm", linewidths=0.5)
# axes[2].clabel(cp, inline=1, fmt='%1.1f', fontsize=8)
axes[2].set_xlabel(r'$\xi_1$', fontsize=16)
axes[2].set_ylabel(r'$\xi_2$', fontsize=16)
axes[2].text(0.5,1,r'$\lambda_i = \frac{1}{\sqrt{i}}$', size=18, bbox=props, \
              transform=axes[2].transAxes, horizontalalignment='center', \
              verticalalignment='center')

f.savefig(fname, format='pdf')
