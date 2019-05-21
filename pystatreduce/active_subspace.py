# active_subspace.py
# This file contains the dimension reduction method presented by Constantine in
# the paper "Active subspace methods in theory and practice: applications to
# Kriging Surfaces"
import numpy as np
import chaospy as cp
from pystatreduce.monte_carlo import MonteCarlo

class ActiveSubspace(object):

    def __init__(self, QoI, n_dominant_dimensions=1, n_monte_carlo_samples=1000,
                 use_svd=False, read_rv_samples=False, write_rv_samples=False):
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
        self.use_svd = use_svd
        self.read_file = read_rv_samples
        self.write_file = write_rv_samples

    def getDominantDirections(self, QoI, jdist):
        systemsize = QoI.systemsize

        if self.read_file == True:
            rv_arr = np.loadtxt('rv_arr.txt')
            np.testing.assert_equal(self.n_monte_carlo_samples, rv_arr.shape[1])
        else:
            use_truncated_samples = True
            if use_truncated_samples:
                rv_arr = self.get_truncated_samples(jdist)
            else:
                rv_arr = jdist.sample(self.n_monte_carlo_samples) # points for computing the uncentered covariance matrix
        if self.write_file == True:
            np.savetxt('rv_arr.txt', rv_arr)

        pert = np.zeros(systemsize)

        new_samples = self.check_3sigma_violation(rv_arr, jdist)

        if self.use_svd == False:
            # Get C_tilde using Monte Carlo
            grad = np.zeros(systemsize)
            self.C_tilde = np.zeros([systemsize, systemsize])
            for i in range(0, self.n_monte_carlo_samples):
                grad[:] = QoI.eval_QoIGradient(rv_arr[:,i], pert)
                self.C_tilde[:,:] += np.outer(grad, grad)
            self.C_tilde[:,:] = self.C_tilde[:,:]/self.n_monte_carlo_samples

            # Factorize C_tilde that
            self.iso_eigenvals, self.iso_eigenvecs = np.linalg.eig(self.C_tilde)
        else:
            # Grad matrix
            grad = np.zeros([systemsize, self.n_monte_carlo_samples])
            for i in range(0, self.n_monte_carlo_samples):
                grad[:,i] = QoI.eval_QoIGradient(rv_arr[:,i], pert)
            grad[:,:] = grad / np.sqrt(self.n_monte_carlo_samples)

            # Perform SVD
            W, s, _ = np.linalg.svd(grad)

            self.iso_eigenvals = s ** 2
            self.iso_eigenvecs = W

        # print('Unsorted')
        # print('eigenvals = ', self.iso_eigenvals)
        # print('eigenvecs = \n', self.iso_eigenvecs)
        # get the indices of dominant eigenvalues in descending order
        sort_ind = self.iso_eigenvals.argsort()[::-1]
        self.iso_eigenvecs[:,:] = self.iso_eigenvecs[:,sort_ind]
        self.iso_eigenvals[:] = self.iso_eigenvals[sort_ind]
        self.dominant_indices = np.arange(0,self.n_dominant_dimensions)
        self.dominant_dir = self.iso_eigenvecs[:, self.dominant_indices]

    def get_truncated_samples(self, jdist):
        mu = cp.E(jdist)
        sigma = cp.Std(jdist)
        upper_bound = mu + 3 * sigma
        lower_bound = mu - 3 * sigma
        # print('mu = ', mu)
        # print('sigma = ', sigma)
        # print('lower_bound = ', lower_bound)
        # print('upper_bound = ', upper_bound)
        dist0 = cp.Truncnorm(lo=lower_bound[0], up=upper_bound[0], mu=mu[0], sigma=sigma[0])
        dist1 = cp.Truncnorm(lo=lower_bound[1], up=upper_bound[1], mu=mu[1], sigma=sigma[1])
        dist2 = cp.Truncnorm(lo=lower_bound[2], up=upper_bound[2], mu=mu[2], sigma=sigma[2])
        dist3 = cp.Truncnorm(lo=lower_bound[3], up=upper_bound[3], mu=mu[3], sigma=sigma[3])
        dist4 = cp.Truncnorm(lo=lower_bound[4], up=upper_bound[4], mu=mu[4], sigma=sigma[4])
        dist5 = cp.Truncnorm(lo=lower_bound[5], up=upper_bound[5], mu=mu[5], sigma=sigma[5])
        dist6 = cp.Truncnorm(lo=lower_bound[6], up=upper_bound[6], mu=mu[6], sigma=sigma[6])

        trunc_jdist = cp.J(dist0, dist1, dist2, dist3, dist4, dist5, dist6)

        rv_arr = trunc_jdist.sample(self.n_monte_carlo_samples)
        return rv_arr

    def check_3sigma_violation(self, rv_arr, jdist):
        mu = cp.E(jdist)
        sigma = cp.Std(jdist) # Standard deviation
        upper_bound = mu + 3 * sigma
        lower_bound = mu - 3 * sigma
        ctr = 0
        idx_list = [] # List of all the violating entries
        for i in range(self.n_monte_carlo_samples):
            sample = rv_arr[:,i]
            if all(sample > lower_bound) == False or all(sample < upper_bound) == False:
                idx_list.append(i)

        print("number of violations = ", len(idx_list))
        print(idx_list)
        # Delete the arrays from the idx_list
        new_samples = np.delete(rv_arr, idx_list, axis=1)

        return new_samples
