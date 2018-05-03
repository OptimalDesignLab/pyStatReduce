# plot isocontours for a standard deviation [0.2, 0.1]
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
theta = np.pi/3
tuple = (theta,)
QoI = examples.Paraboloid2D(2, tuple)
sigma = np.array([2, 0.1]) # np.ones(2)
jdist = cp.MvNormal(mu, np.diag(sigma))
covariance = cp.Cov(jdist)
sqrt_Sigma = np.sqrt(covariance)
inv_Jac = np.linalg.inv(sqrt_Sigma)

n_xi = 150
xi_bar_min = [-3.0, -3.0]
xi_bar_max = [3.0, 3.0]
xi_bar_1 = np.linspace(xi_bar_min[0], xi_bar_max[0],n_xi)
xi_bar_2 = np.linspace(xi_bar_min[1], xi_bar_max[1],n_xi)
xi_bar = np.zeros(2)
xi = np.zeros(2)
J_xi = np.zeros([n_xi, n_xi])
for i in xrange(0, n_xi):
    for j in xrange(0, n_xi):
        xi_bar[:] = np.array([xi_bar_1[i], xi_bar_2[j]])
        xi[:] = np.dot(sqrt_Sigma, xi_bar) # np.dot(inv_Jac, xi_bar)
        J_xi[j,i] = QoI.eval_QoI(mu, xi)

# dominant directions
Hessian = QoI.eval_QoIHessian(mu, np.zeros(2))
Hessian_Product = np.matmul(sqrt_Sigma, np.matmul(Hessian, sqrt_Sigma))
eigenvalues, V_orig = np.linalg.eig(Hessian_Product)
sort_ind = eigenvalues.argsort()
V_orig[:,:] = V_orig[:,sort_ind]
V2 = V_orig[:,1]
theta_bar = np.arctan(V2[1]/V2[0])

# plot dominant axis
xi_hat = np.zeros(n_xi)
for i in xrange(0, n_xi):
    xi_hat[i] = np.tan(theta_bar) * xi_bar_1[i]

# Plot quiver
quiver_scale = np.diag([1.0, 3.0])
scaled_vec = np.matmul(V_orig, quiver_scale)
qX = np.zeros(2)
qY = np.zeros(2)
qU = scaled_vec[0, :]
qV = scaled_vec[1, :]

# Actual plotting

# fname = "2D_quadratic_iso_contour_std_normal.pdf"
fname = "2D_quadratic_iso_contour_diff_sigma.pdf"

plt.rc('text', usetex=True)
matplotlib.rcParams['mathtext.fontset'] = 'cm'
fig = plt.figure("reduced_quadratic", figsize=(6,6))
ax = plt.axes()
cp = ax.contour(xi_bar_1, xi_bar_2, J_xi, levels=[0.5,2,4,8,16,64,256, 512], cmap="coolwarm", linewidths=0.5)
ax.clabel(cp, inline=1, fmt='%1.1f', fontsize=8)
lp = ax.plot(xi_bar_1, xi_hat, linestyle="--", dashes=(5,5), linewidth=1.5, color="gray", label="Dominant Direction")
qp = ax.quiver(qX, qY, qU, qV, units="xy", width=0.03, scale=1.8)
ax.set_xlim(xi_bar_min[0], xi_bar_max[0])
# ax.set_ylim(-2.0, 2.0)
ax.set_ylim(xi_bar_min[1], xi_bar_max[1])
ax.set_xlabel(r'$\tilde{\xi}_1$', fontsize=16)
ax.set_ylabel(r'$\tilde{\xi}_2$', fontsize=16)
# ax.annotate(r'$\mathsf{V}_{1}$', xy=[-0.8,-0.5], fontsize=16) # same sigma
# ax.annotate(r'$\mathsf{V}_{2}$', xy=[0.3,-1.5], fontsize=16)  # same sigma
ax.annotate(r'$\mathsf{V}_{1}$', xy=[-0.2,0.6], fontsize=16) # different sigma
ax.annotate(r'$\mathsf{V}_{2}$', xy=[1.3,-0.6], fontsize=16)  # different sigma
ax.legend(loc="upper right", fancybox="true", prop={'size': 16})
ax.tick_params(axis='both', labelsize=16)
plt.tight_layout()
fig.savefig(fname, format="pdf")
