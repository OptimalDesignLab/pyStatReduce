# run_hadamard
import numpy as np
from mpl_toolkits import mplot3d
import matplotlib
import matplotlib.pyplot as plt

import os
import sys
import errno
sys.path.insert(0, '../../src')

import chaospy as cp

from stochastic_collocation import StochasticCollocation
from quantity_of_interest import QuantityOfInterest
from dimension_reduction import DimensionReduction
from stochastic_arnoldi.arnoldi_sample import ArnoldiSampling
import examples


nx = 100
x1 = np.linspace(-5,5, num=nx)
x2 = np.linspace(-5,5, num=nx)
val = np.zeros([nx,nx])
QoI = examples.Rosenbrock(2)
xi = np.zeros(2)
std_dev_xi = np.ones(2)
collocation_QoI = StochasticCollocation(3, "Normal")
for i in xrange(0, nx):
    for j in xrange(0, nx):
        mu = np.array([x1[i], x2[j]])
        QoI_func = QoI.eval_QoI
        val[j,i] = collocation_QoI.normal.mean(mu, std_dev_xi, QoI_func)
        # val[j,i] = QoI.eval_QoI(mu, xi)


fname = "rosenbrock_exact.pdf"
plt.rc('text', usetex=True)
matplotlib.rcParams['mathtext.fontset'] = 'cm'
fig = plt.figure("rosenbrock", figsize=(6,6))
ax = plt.axes()
# cp = ax.contour(x1, x2, val, levels=[2,4,8,16,32,64,128,256, 512], cmap="coolwarm", linewidths=0.5)
cp = ax.contour(x1, x2, val, cmap="coolwarm", levels=[500, 1000, 5000, 10000, 15000, 30000, 45000, 90000], linewidths=0.5)
ax.clabel(cp, inline=1, fmt='%1.1f', fontsize=8)
ax.set_xlabel(r'$\xi_1$', fontsize=16)
ax.set_ylabel(r'$\xi_2$', fontsize=16)
plt.tight_layout()
plt.show()
# fig.savefig(fname, format="pdf")
