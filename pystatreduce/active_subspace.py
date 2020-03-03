# active_subspace.py
# This file contains the dimension reduction method presented by Constantine in
# the paper "Active subspace methods in theory and practice: applications to
# Kriging Surfaces"
import numpy as np
import chaospy as cp
import copy
from pystatreduce.monte_carlo import MonteCarlo

class ActiveSubspace(object):

    def __init__(self, QoI, n_dominant_dimensions=1, n_monte_carlo_samples=1000,
                 use_svd=False, read_rv_samples=False, write_rv_samples=False,
                 read_gradient_samples=False, gradient_array=None,
                 use_iso_transformation=False, use_truncated_samples=False):
        """
        This file contains the dimension reduction method presented by
        Constantine in the paper "Active subspace methods in theory and
        practice: applications to Kriging Surfaces". The Kriging surrogate is
        not implemented.

        **Constructor Arguments**
        * `n_dominant_dimensions` : Number of dominant directions expected by the user.
        * `n_monte_carlo_samples` : Number of monte carlo samples needed to construct
                                    the uncentered covariance of the gradient vector.
        * `use_svd` : Bool for whether to use SVD of eigen decomposition to
                      identify the active subspace. Default is False.
        * `read_rv_samples` : read a text file which loads in the random variables
                              where the gradients are to be evaluated. The
                              filename should be `rv_arr.txt`. Default = False
        * `write_rv_samples` : write the random variables generated from a distribution
                               to file. The filename is `rv_arr.txt`, Default is
                               False.
        * `read_gradient_samples` : Reads gradient samples from an array. The
                                    gradients should be supplied using the
                                    keyword argument `gradient_array`.
        * `gradient_array` : Numpy array which is the collection of gradients
        * `use_iso_transformation` : Bool for using an isoprobabilistic
                                     transformation for the uncentered
                                     covariance of the gradient matrix. Default
                                     is False.
        * `use_truncated_samples` : Generates samples that lie within 3 standard
                                    deviations. Default is False
        """
        self.n_dominant_dimensions = n_dominant_dimensions
        self.n_monte_carlo_samples = n_monte_carlo_samples
        self.use_svd = use_svd
        self.use_iso_transformation = use_iso_transformation
        self.read_gradient_samples = read_gradient_samples
        if read_gradient_samples:
            assert gradient_array is not None
            assert gradient_array.shape == (self.n_monte_carlo_samples, QoI.systemsize)
            self.gradient_array = gradient_array

        # Debug flags
        self.read_file = read_rv_samples
        self.write_file = write_rv_samples
        self.check_std_dev_violation = False
        self.use_truncated_samples = use_truncated_samples

    def getDominantDirections(self, QoI, jdist):
        systemsize = QoI.systemsize

        if self.read_file == True:
            rv_arr = np.loadtxt('rv_arr.txt')
            np.testing.assert_equal(self.n_monte_carlo_samples, rv_arr.shape[1])
        else:
            if self.use_truncated_samples:
                rv_arr = self.get_truncated_samples(jdist)
            else:
                rv_arr = jdist.sample(self.n_monte_carlo_samples) # points for computing the uncentered covariance matrix
        if self.write_file == True:
            np.savetxt('rv_arr.txt', rv_arr)

        pert = np.zeros(systemsize)

        if self.use_truncated_samples:
            new_samples = self.check_3sigma_violation(rv_arr, jdist)

        if self.use_svd == False:
            # Get C_tilde using Monte Carlo
            self.C_tilde = np.zeros([systemsize, systemsize])
            if self.read_gradient_samples:
                for i in range(0, self.n_monte_carlo_samples):
                    self.C_tilde[:,:] += np.outer(self.gradient_array[i,:], self.gradient_array[i,:])
            else:
                grad = np.zeros(systemsize)
                for i in range(0, self.n_monte_carlo_samples):
                    grad[:] = QoI.eval_QoIGradient(rv_arr[:,i], pert)
                    self.C_tilde[:,:] += np.outer(grad, grad)
            self.C_tilde[:,:] = self.C_tilde[:,:]/self.n_monte_carlo_samples

            # Factorize C_tilde that
            self.iso_eigenvals, self.iso_eigenvecs = np.linalg.eig(self.C_tilde)
        else:
            if self.read_gradient_samples:
                grad = copy.deepcopy(self.gradient_array.T)
            else:
            # Grad matrix
                grad = np.zeros([systemsize, self.n_monte_carlo_samples])
                for i in range(0, self.n_monte_carlo_samples):
                    grad[:,i] = QoI.eval_QoIGradient(rv_arr[:,i], pert)
            grad[:,:] = grad / np.sqrt(self.n_monte_carlo_samples)
            # Perform SVD
            W, s, _ = np.linalg.svd(grad)

            if self.use_iso_transformation:
                # Do the isoprobabilistic transformation
                covariance = cp.Cov(jdist)
                sqrt_Sigma = np.sqrt(covariance)
                iso_grad = np.dot(grad.T, sqrt_Sigma).T
                # Perform SVD
                W, s, _ = np.linalg.svd(iso_grad)

            self.iso_eigenvals = s ** 2
            self.iso_eigenvecs = W

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

        # print("number of violations = ", len(idx_list))
        # print(idx_list)
        # Delete the arrays from the idx_list
        new_samples = np.delete(rv_arr, idx_list, axis=1)

        return new_samples

class MultifidelityActiveSubspace(object):

    def __init__(self, n_rv, n_dominant_dimensions, n_monte_carlo_samples=[50, 100],
                 read_rv_samples=False, write_rv_samples=False,
                 use_iso_transformation=False, use_truncated_samples=False):

        """
        Implementation of a bifidelity active subspace paper as demonstrated in
        the paper by Lam. arXiv:1809.05567
        """

        self.systemsize = n_rv
        self.n_dominant_dimensions = n_dominant_dimensions
        self.n_monte_carlo_samples = n_monte_carlo_samples
        self.use_svd = use_svd
        self.use_iso_transformation = use_iso_transformation

        # Debug flags
        self.read_file = read_rv_samples
        self.write_file = write_rv_samples
        self.check_std_dev_violation = False
        self.use_truncated_samples = use_truncated_samples

    def getDominantDirections(self, jdist, QoI_dict):
        systemsize = QoI.systemsize
        self.constructGradientCovarianceMatrix()
        self.iso_eigenvals, self.iso_eigenvecs = np.linalg.eig(self.C_tilde)

        # get the indices of dominant eigenvalues in descending order
        sort_ind = self.iso_eigenvals.argsort()[::-1]
        self.iso_eigenvecs[:,:] = self.iso_eigenvecs[:,sort_ind]
        self.iso_eigenvals[:] = self.iso_eigenvals[sort_ind]
        self.dominant_indices = np.arange(0,self.n_dominant_dimensions)
        self.dominant_dir = self.iso_eigenvecs[:, self.dominant_indices]


    def constructGradientCovarianceMatrix(self, jdist, QoI_dict):
        self.C_tilde = np.zeros([self.systemsize, self.systemsize])
        rv_arr = jdist.sample(sum(self.n_monte_carlo_samples))

        # Evaluate high fidelity contribution
        pert = np.zeros(self.systemsize)
        # grad_lf = np.zeros(self.n_monte_carlo_samples[0], self.systemsize)
        # int_mat = np.zeros(self.systemsize, self.systemsize)
        for i in range(self.n_monte_carlo_samples[0]):
            grad_hf = QoI_dict['high_fidelity'].eval_QoIGradient(rv_arr[:,i], pert)
            grad_lf = QoI_dict['low_fidelity'].eval_QoIGradient(rv_arr[:,i], pert)
            self.C_tilde[:,:] += np.outer(grad_hf, grad_hf) - np.outer(grad_lf[i,:], grad_lf[i,:])
            # int_mat += np.outer(grad_lf[i,:], grad_lf[i,:])
        self.C_tilde[:,:] = self.C_tilde[:,:]/self.n_monte_carlo_samples[0]

        # Evaluate the low fidelity contribution
        int_mat = np.zeros([self.systemsize, self.systemsize])
        for i in range(self.n_monte_carlo_samples[0], sum(self.n_monte_carlo_samples)):
            grad = QoI_dict['low_fidelity'].eval_QoIGradient(rv_arr[:,i], pert)
            int_mat[:,:] += np.outer(grad, grad)
        int_mat[:,:] = int_mat[:,:] / self.n_monte_carlo_samples[1]

        # Construct the full C_tilde
        self.C_tilde += int_mat

if __name__ == '__main__':
    pass
