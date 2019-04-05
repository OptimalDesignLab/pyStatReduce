# active_subspace.py
# This file contains the dimension reduction method presented by Constantine in
# the paper "Active subspace methods in theory and practice: applications to
# Kriging Surfaces"
import numpy as np
import chaospy as cp
from monte_carlo import MonteCarlo

class ActiveSubspace(object):

    def __init__(self, n_dominant_dimensions=1, n_monte_carlo_samples=1000):
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
        C_tilde = np.zeros([systemsize, systemsize])

        # Get C_tilde using Monte Carlo
        grad = np.zeros(systemsize)
        C_tilde = np.zeros([systemsize, systemsize])
        mu = np.zeros(2)
        for i in xrange(0, self.n_monte_carlo_samples):
            rv = jdist.sample()
            grad[:] = QoI.eval_QoIGradient(mu, rv)
            C_tilde[:,:] = np.outer(grad, grad)
        C_tilde[:,:] = C_tilde[:,:]/self.n_monte_carlo_samples

        # Factorize C_tilde that
        self.iso_eigenvals, self.iso_eigenvecs = np.linalg.eig(C_tilde)
        # get the indices of dominant eigenvalues in descending order
        sort_ind = self.iso_eigenvals.argsort()[::-1]
        self.iso_eigenvecs[:,:] = self.iso_eigenvecs[:,sort_ind]
        self.iso_eigenvals[:] = self.iso_eigenvals[sort_ind]
        self.dominant_indices = np.arange(0,self.n_dominant_dimensions)
