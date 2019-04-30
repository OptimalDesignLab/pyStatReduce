# active_subspace.py
# This file contains the dimension reduction method presented by Constantine in
# the paper "Active subspace methods in theory and practice: applications to
# Kriging Surfaces"
import numpy as np
import chaospy as cp
from pystatreduce.monte_carlo import MonteCarlo

class ActiveSubspace(object):

    def __init__(self, QoI, n_dominant_dimensions=1, n_monte_carlo_samples=1000):
        """
        This file contains the dimension reduction method presented by
        Constantine in the paper "Active subspace methods in theory and
        practice: applications to Kriging Surfaces". The Kriging surrogate is
        not implemented.
        """
        self.n_dominant_dimensions = n_dominant_dimensions
        self.n_monte_carlo_samples = n_monte_carlo_samples

    def getDominantDirections(self, QoI, jdist):
        systemsize = QoI.systemsize

        # Get C_tilde using Monte Carlo
        grad = np.zeros(systemsize)
        C_tilde = np.zeros([systemsize, systemsize])
        rv_arr = jdist.sample(self.n_monte_carlo_samples) # points for computing the uncentered covariance matrix
        pert = np.zeros(systemsize)
        for i in range(0, self.n_monte_carlo_samples):
            grad[:] = QoI.eval_QoIGradient(rv_arr[:,i], pert)
            C_tilde[:,:] += np.outer(grad, grad)
        C_tilde[:,:] = C_tilde[:,:]/self.n_monte_carlo_samples

        # Factorize C_tilde that
        self.iso_eigenvals, self.iso_eigenvecs = np.linalg.eig(C_tilde)
        # get the indices of dominant eigenvalues in descending order
        sort_ind = self.iso_eigenvals.argsort()[::-1]
        self.iso_eigenvecs[:,:] = self.iso_eigenvecs[:,sort_ind]
        self.iso_eigenvals[:] = self.iso_eigenvals[sort_ind]
        self.dominant_indices = np.arange(0,self.n_dominant_dimensions)
        self.dominant_dir = self.iso_eigenvecs[:, self.dominant_indices]
