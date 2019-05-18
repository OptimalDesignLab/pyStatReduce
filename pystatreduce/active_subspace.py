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

        **Constructor Arguments**
        * `n_dominant_dimensions` : Number of dominant directions expected by the user.
        * `n_monte_carlo_samples` : Number of monte carlo samples needed to construct
                                    the uncentered covariance of the gradient vector.
        """
        self.n_dominant_dimensions = n_dominant_dimensions
        self.n_monte_carlo_samples = n_monte_carlo_samples

    def getDominantDirections(self, QoI, jdist):
        systemsize = QoI.systemsize

        # Get C_tilde using Monte Carlo
        grad = np.zeros(systemsize)
        self.C_tilde = np.zeros([systemsize, systemsize])

        read_file = False # Debug
        write_file = False
        if read_file == True:
            rv_arr = np.loadtxt('rv_arr.txt')
            np.testing.assert_equal(self.n_monte_carlo_samples, rv_arr.shape[1])
        else:
            rv_arr = jdist.sample(self.n_monte_carlo_samples) # points for computing the uncentered covariance matrix

        if write_file == True:
            np.savetxt('rv_arr.txt', rv_arr)

        pert = np.zeros(systemsize)
        for i in range(0, self.n_monte_carlo_samples):
            grad[:] = QoI.eval_QoIGradient(rv_arr[:,i], pert)
            self.C_tilde[:,:] += np.outer(grad, grad)
        self.C_tilde[:,:] = self.C_tilde[:,:]/self.n_monte_carlo_samples

        # Factorize C_tilde that
        self.iso_eigenvals, self.iso_eigenvecs = np.linalg.eig(self.C_tilde)
        # print('Unsorted')
        # print('eigenvals = ', self.iso_eigenvals)
        # print('eigenvecs = \n', self.iso_eigenvecs)
        # get the indices of dominant eigenvalues in descending order
        sort_ind = self.iso_eigenvals.argsort()[::-1]
        self.iso_eigenvecs[:,:] = self.iso_eigenvecs[:,sort_ind]
        self.iso_eigenvals[:] = self.iso_eigenvals[sort_ind]
        self.dominant_indices = np.arange(0,self.n_dominant_dimensions)
        self.dominant_dir = self.iso_eigenvecs[:, self.dominant_indices]
