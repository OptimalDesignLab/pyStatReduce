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

# # Both random variables have the same standard deviation
# V_orig = np.array([[-np.sqrt(3)/2, -0.5],[-0.5, np.sqrt(3)/2]])
# V2 = np.array([-0.5, np.sqrt(3)/2])
# theta_bar = theta

# random variables have different standard deviation
sigma = np.array([2, 0.2])
jdist = cp.MvNormal(mu, np.diag(sigma))
Hessian = QoI.eval_QoIHessian(mu, np.zeros(2))
covariance = cp.Cov(jdist)
sqrt_Sigma = np.sqrt(covariance)
Hessian_Product = np.matmul(sqrt_Sigma, np.matmul(Hessian, sqrt_Sigma))
eigenvalues, V_orig = np.linalg.eig(Hessian_Product)
V2 = V_orig[:,1]
# V_orig[:,0] = -V_orig[:,0]
print 'V1 = ', V_orig[:,0]
print 'V2 = ', V_orig[:,1]
theta_bar = np.arctan(V2[1]/V2[0])
# contour plot
n_xi = 100
xi_min = [-3.0, -3.0]
xi_max = [3.0, 3.0]
xi_1 = np.linspace(xi_min[0], xi_max[0],n_xi)
xi_2 = np.linspace(xi_min[1], xi_max[1],n_xi)
xi = np.zeros(2)
J_xi = np.zeros([n_xi, n_xi])
for i in xrange(0, n_xi):
    for j in xrange(0, n_xi):
        xi[:] = np.array([xi_1[i], xi_2[j]])
        J_xi[j,i] = QoI.eval_QoI(mu, xi)

# plot dominant axis
xi_hat = np.zeros(n_xi)
for i in xrange(0, n_xi):
    xi_hat[i] = np.tan(-theta_bar) * xi_1[i]

# Plot quiver
quiver_scale = np.diag([1.0, 3.0])
scaled_vec = np.matmul(quiver_scale, V_orig)
qX = np.zeros(2)
qY = np.zeros(2)
qU = scaled_vec[:, 0]
qV = scaled_vec[:, 1]

# if std dev not equal
qU[0] = -qU[0]
qV[0] = -qV[0]

# Quadrature points
n_quadrature_points = 5
q, w = np.polynomial.hermite.hermgauss(n_quadrature_points)
points = np.zeros([n_quadrature_points, 2])
for i in xrange(0, n_quadrature_points):
    points[i,:] = np.sqrt(2)*V2*q[i]

plt.rc('text', usetex=True)
matplotlib.rcParams['mathtext.fontset'] = 'cm'

fig = plt.figure("reduced_quadratic", figsize=(6,6))
ax = plt.axes()
cp = ax.contour(xi_1, xi_2, J_xi, levels=[2,4,6,8,16,32,64,128,256, 512], cmap="coolwarm", linewidths=0.5)
ax.clabel(cp, inline=1, fontsize=8)
lp = ax.plot(xi_1, xi_hat, linestyle="--", dashes=(5,5), linewidth=1.5, color="gray", label="Dominant Direction")
qp = ax.quiver(qX, qY, qU, qV, units="xy", width=0.03, scale=1.8)
# sp = ax.scatter(points[:,0], points[:,1], color="black", label="Collocation Points")
ax.set_xlim(xi_min[0], xi_max[0])
# ax.set_ylim(-2.0, 2.0)
ax.set_ylim(xi_min[1], xi_max[1])
ax.set_xlabel(r'$\xi_1$', fontsize=16)
ax.set_ylabel(r'$\xi_2$', fontsize=16)
ax.annotate(r'$\mathsf{V}_{1}$', xy=[-0.8,-0.6], fontsize=16)
ax.annotate(r'$\mathsf{V}_{2}$', xy=[-0.6,1.2], fontsize=16)
ax.legend(loc="upper right", fancybox="true", prop={'size': 16})
ax.tick_params(axis='both', labelsize=16)
plt.tight_layout()
# ax.grid("on")

fig.savefig("2D_quadratic_contour.pdf", format="pdf")
