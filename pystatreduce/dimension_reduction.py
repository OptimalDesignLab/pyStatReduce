# Dimension Reduction

import numpy as np
import chaospy as cp
from pystatreduce.stochastic_arnoldi.arnoldi_sample import ArnoldiSampling

class DimensionReduction(object):
    """
    Base class that comprises of all the dimension reduction strategies that are
    provided by this package. Dominant directions for approximating expected
    values of QoI can be obtained using member functions of this class.

    **Class Members**

    * `threshold_factor` : fraction of threshold energy that MUST be had by an
                           approximation of the expected value of QoI.
    * `iso_eigenvals` : Eigenvalues of the Hessian of the QoI in the
                         isoprobabilistic space
    * `iso_eigenvecs` : Eigenvectors of the Hessian of the QoI in the
                            isoprobabilistic space
    * `dominant_indices` : array of indices that correspond to the column number
                           in iso_eigenvecs, of the largest eigen values
                           needed for approximation.

    * `num_sample` : Number of arnoldi samples. When using the exact Hessian,
                        num_sample = QoI.systemsize

    **Constructor Arguments**

    * `threshold_factor` : fraction of threshold energy that MUST be had by an
                           approximation of the expected value of QoI.

    """

    def __init__(self, **kwargs):
        # self.threshold_factor = threshold_factor

        # Decide between using arnoldi-iteration or exact Hessian
        if kwargs['exact_Hessian'] == False:
            self.use_exact_Hessian = False

            if 'min_eigen_accuracy' in kwargs:
                self.min_eigen_accuracy = kwargs.get('min_eigen_accuracy')
            else:
                self.min_eigen_accuracy = 1.e-3

            if 'n_arnoldi_sample' in kwargs:
                self.num_sample = kwargs.get('n_arnoldi_sample')
            else:
                self.num_sample = 51

            if 'sample_radius' in kwargs:
                self.sample_radius = kwargs.get('sample_radius')
            else:
                self.sample_radius = 1.e-6

            if 'eigen_discard_ratio' in kwargs:
                self.discard_ratio = kwargs.get('eigen_discard_ratio')
            else:
                self.discard_ratio = 1.e-10
        else:
            self.use_exact_Hessian = True

            if 'threshold_factor' in kwargs:
                self.threshold_factor = kwargs.get('threshold_factor')
            else:
                self.threshold_factor = 0.9
        # TODO: Check if isoprobabilistic eigen modes can be initialized here

    def getDominantDirections(self, QoI, jdist, **kwargs):
        """
        Given a Quantity of Interest and a jpint distribution of the random
        variables, this function computes the directions along which the
        function is most nonlinear. We call these nonlinear directions as
        dominant directions.

        **Arguments**

        * `QoI` : QuantityOfInterest object
        * `jdist` : Joint distribution object

        """

        mu = cp.E(jdist)
        covariance = cp.Cov(jdist)
        # Check if variance covariance matrix is diagonal
        if np.count_nonzero(covariance - np.diag(np.diagonal(covariance))) == 0:
            sqrt_Sigma = np.sqrt(covariance)
        else:
            raise NotImplementedError
        # TODO: Break the long if blocks into functions
        if self.use_exact_Hessian == True:

            # Get the Hessian of the QoI
            xi = np.zeros(QoI.systemsize)
            Hessian = QoI.eval_QoIHessian(mu, xi)

            # Compute the eigen modes of the Hessian of the Hadamard Quadratic
            Hessian_Product = np.matmul(sqrt_Sigma, np.matmul(Hessian, sqrt_Sigma))
            self.iso_eigenvals, self.iso_eigenvecs = np.linalg.eig(Hessian_Product)

            self.num_sample = QoI.systemsize

            # Next,
            # Get the system energy of Hessian_Product
            system_energy = np.sum(self.iso_eigenvals)
            ind = []

            # get the indices of dominant eigenvalues in descending order
            sort_ind = self.iso_eigenvals.argsort()[::-1]

            # Check the threshold
            for i in range(0, self.num_sample):
                dominant_eigen_val_ind = sort_ind[0:i+1]
                reduced_energy = np.sum(self.iso_eigenvals[dominant_eigen_val_ind])
                if reduced_energy <= self.threshold_factor*system_energy:
                    ind.append(dominant_eigen_val_ind[i])
                else:
                    break

            if len(ind) == 0:
                ind.append(np.argmax(self.iso_eigenvals))

            self.dominant_indices = ind

        else:
            # approximate the hessian of the QoI in the isoprobabilistic space
            # 1. Initialize ArnoldiSampling object
            # Check if the system size is smaller than the number of arnoldi sample - 1
            if QoI.systemsize < self.num_sample-1:
                self.num_sample = QoI.systemsize+1

            arnoldi = ArnoldiSampling(self.sample_radius, self.num_sample)

            # 2. Declare arrays
            # 2.1 iso-eigenmodes
            self.iso_eigenvals = np.zeros(self.num_sample-1)
            self.iso_eigenvecs = np.zeros([QoI.systemsize, self.num_sample-1])
            # 2.2 solution and function history array
            mu_iso = np.zeros(QoI.systemsize)
            gdata0 = np.zeros([QoI.systemsize, self.num_sample])

            # 3. Approximate eigenvalues using ArnoldiSampling
            # 3.1 Convert x into a standard normal distribution
            mu_iso[:] = 0.0
            gdata0[:,0] = np.dot(QoI.eval_QoIGradient(mu, np.zeros(QoI.systemsize)),
                                sqrt_Sigma)
            dim, error_estimate = arnoldi.arnoldiSample(QoI, jdist, mu_iso, gdata0,
                                                        self.iso_eigenvals, self.iso_eigenvecs)

            # Check how many eigen pairs are good. We will exploit the fact that
            # the eigen pairs are already sorted in a descending order. We will
            # only use "good" eigen pairs to estimate the dominant directions.
            ctr = 0
            for i in range(0, dim):
                if error_estimate[i] < self.min_eigen_accuracy:
                    ctr += 1
                else:
                    break
            # print('error_estimate = ', error_estimate)
            # Compute the accumulated energy
            # acc_energy = np.sum(np.square(self.iso_eigenvals[0:ctr]))

            # Compute the magnitude of the eigenvalues w.r.t the largest eigenvalues
            relative_ratio = self.iso_eigenvals[0:ctr] / self.iso_eigenvals[0]

            # We will only use eigenpairs which whose relative size > discard_ratio
            # Since we are considering the size we consider the absolute magnitude
            usable_pairs = np.where(np.abs(relative_ratio) > self.discard_ratio)

            if 'max_eigenmodes' in kwargs:
                max_eigenmodes = kwargs['max_eigenmodes']
                if usable_pairs[0].size < max_eigenmodes:
                    self.dominant_indices = usable_pairs[0]
                else:
                    self.dominant_indices = usable_pairs[0][0:max_eigenmodes]
            else:
                self.dominant_indices = usable_pairs[0]

        # For either of the two cases of Exact hessian or inexact Hessian, we
        # specify the dominant indices
        self.dominant_dir = self.iso_eigenvecs[:, self.dominant_indices]

    #--------------------------------------------------------------------------#
    # Experimental Section
    #--------------------------------------------------------------------------#

    def calcMarginals(self, jdist):
        """
        Compute the marginal density object for the dominant space. The current
        implementation is only for Gaussian distribution.
        """

        marginal_size = len(self.dominant_indices)
        orig_mean = cp.E(jdist)
        orig_covariance = cp.Cov(jdist)

        # Step 1: Rotate the mean & covariance matrix of the the original joint
        # distribution along the eigenve
        dominant_vecs = self.iso_eigenvecs[:,self.dominant_indices]
        marginal_mean = np.dot(dominant_vecs.T, orig_mean)
        marginal_covariance = np.matmul(dominant_vecs.T,np.matmul(orig_covariance, dominant_vecs))

        # Step 2: Create the new marginal distribution
        if marginal_size == 1: # Univariate distributions have to be treated separately
            marginal_std_dev = np.sqrt(np.asscalar(marginal_covariance))
            self.marginal_distribution = cp.Normal(np.asscalar(marginal_mean),
                                         marginal_std_dev)
        else:
            self.marginal_distribution = cp.MvNormal(marginal_mean,
                                         marginal_covariance)
