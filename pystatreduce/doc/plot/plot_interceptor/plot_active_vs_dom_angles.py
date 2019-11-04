################################################################################
# plot_active_vs_dom_angles.py
#
# This file plots the subspace angles between the dominant directions and the
# active subspace basis vectors
#
################################################################################

# Matplotlib imports
from mpl_toolkits import mplot3d
import matplotlib
import matplotlib.pyplot as plt

# pyStatReduce specific imports
import os, sys
import numpy as np
import chaospy as cp
import numdifftools as nd
from pystatreduce.new_stochastic_collocation import StochasticCollocation2
from pystatreduce.monte_carlo import MonteCarlo
from pystatreduce.quantity_of_interest import QuantityOfInterest
from pystatreduce.dimension_reduction import DimensionReduction
from pystatreduce.active_subspace import ActiveSubspace
from pystatreduce.stochastic_arnoldi.arnoldi_sample import ArnoldiSampling
import pystatreduce.utils as utils
import pystatreduce.examples as examples
from pystatreduce.optimize.dymos_interceptor.quadratic_surrogate.interceptor_surrogate_qoi import InterceptorSurrogateQoI

# Create the atmospheric joint distribution
systemsize = 45
mu = np.zeros(systemsize)
deviations =  np.array([0.1659134,  0.1659134, 0.16313925, 0.16080975, 0.14363596, 0.09014088, 0.06906912, 0.03601839, 0.0153984 , 0.01194864, 0.00705978, 0.0073889 , 0.00891946,
 0.01195811, 0.01263033, 0.01180144, 0.00912247, 0.00641914, 0.00624566, 0.00636504, 0.0064624 , 0.00639544, 0.0062501 , 0.00636687, 0.00650337, 0.00699955,
 0.00804997, 0.00844582, 0.00942114, 0.01080109, 0.01121497, 0.01204432, 0.0128207 , 0.01295824, 0.01307331, 0.01359864, 0.01408001, 0.01646131, 0.02063841,
 0.02250183, 0.02650464, 0.02733539, 0.02550976, 0.01783919, 0.0125073 , 0.01226541])
jdist = cp.MvNormal(mu, np.diag(deviations[:-1]))

# Create the surrogate object
fname = 'surrogate_samples_pseudo_random.npz' # 'surrogate_samples_pseudo_random_0.1.npz'
surrogate_input_dict = {'surrogate info full path' : os.environ['HOME'] + '/UserApps/pyStatReduce/pystatreduce/optimize/dymos_interceptor/quadratic_surrogate/' + fname,
                        'surrogate_type' : 'kriging',
                        'kriging_theta' : 1.e-4,
                        'correlation function' : 'squar_exp',
                       }
surrogate_QoI = InterceptorSurrogateQoI(systemsize, surrogate_input_dict)

# Get the dominant directions
dominant_space = DimensionReduction(n_arnoldi_sample=systemsize+1,
                                    exact_Hessian=False,
                                    sample_radius=1.e-1)
dominant_space.getDominantDirections(surrogate_QoI, jdist, max_eigenmodes=10)

# Get the Active Subspace
active_subspace = ActiveSubspace(surrogate_QoI,
                                 n_dominant_dimensions=20,
                                 n_monte_carlo_samples=1000,
                                 read_rv_samples=False,
                                 use_svd=True,
                                 use_iso_transformation=True)
active_subspace.getDominantDirections(surrogate_QoI, jdist)

# Now get the two angles
n_bases = 2
eigenvecs_dom = dominant_space.iso_eigenvecs[:,0:n_bases]
eigenvecs_active = active_subspace.iso_eigenvecs[:,0:n_bases]
angles_radians = utils.compute_subspace_angles(eigenvecs_dom, eigenvecs_active)
angles_degrees = np.degrees(angles_radians)
print(angles_degrees)
"""
# Finally plot
xvals = range(1, n_bases+1)
fname = "dominant_vs_active_angles.pdf"
fig = plt.figure('scatter', figsize=(6,5))
ax = plt.axes()
s = ax.scatter(xvals, angles_degrees, marker='o')
ax.set_xlabel('subspace indices')
ax.set_ylabel('angles (degrees)')
plt.xticks(xvals)
plt.tight_layout()
# plt.show()
plt.savefig(fname, format='pdf')
"""
