# Dimension Reduction

import numpy as np
import chaospy as cp
from stochastic_arnoldi.arnoldi_sample import ArnoldiSampling

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

    def __init__(self, threshold_factor, **kwargs):
        self.threshold_factor = threshold_factor

        # Decide between using arnoldi-iteration or exact Hessian
        if kwargs['exact_Hessian'] == False:
            self.use_exact_Hessian = False
            self.min_eigen_accuracy = 1.e-3
            if 'n_arnoldi_sample' in kwargs:
                self.num_sample = kwargs.get('n_arnoldi_sample')
            else:
                self.num_sample = 51

            if 'sample_radius' in kwargs:
                self.sample_radius = kwargs.get('sample_radius')
            else:
                self.sample_radius = 1.e-6
        else:
            self.use_exact_Hessian = True
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
            for i in xrange(0, self.num_sample):
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

            # print "gdata0 = ", '\n', gdata0
            # print "dim = ", dim
            # print "iso_eigenvecs.size = ", self.iso_eigenvecs.shape
            # print "error_estimate = ", error_estimate

            # Check how many eigen pairs are good. We will exploit the fact that
            # the eigen pairs are already sorted in a descending order. We will
            # only use "good" eigen pairs to estimate the dominant directions.
            ctr = 0
            for i in xrange(0, dim):
                if error_estimate[i] < self.min_eigen_accuracy:
                    ctr += 1
                else:
                    break
            # print "ctr = ", ctr
            # print "self.iso_eigenvals[0:ctr] = ", self.iso_eigenvals[0:ctr]

            # Compute the accumulated energy
            # acc_energy = np.sum(np.square(self.iso_eigenvals[0:ctr]))

            # Compute the magnitude of the eigenvalues w.r.t the largest eigenvalues
            relative_ratio = self.iso_eigenvals[0:ctr] / self.iso_eigenvals[0]

            discard_ratio = 1.e-10 # 0.1
            # We will only use eigenpairs which whose relative size > discard_ratio
            usable_pairs = np.where(relative_ratio > discard_ratio)
            # print "usable_pairs[0].size = ", usable_pairs[0].size

            if 'max_eigenmodes' in kwargs:
                max_eigenmodes = kwargs['max_eigenmodes']
                if usable_pairs[0].size < max_eigenmodes:
                    self.dominant_indices = usable_pairs[0]
                else:
                    self.dominant_indices = usable_pairs[0][0:max_eigenmodes]
            else:
                self.dominant_indices = usable_pairs[0]


            # # Now use only a portion of the accumulated energy
            # ind = []
            # for i in usable_pairs[0]:
            #     dominant_eigen_val_ind = usable_pairs[0][0:i+1]
            #     # print "dominant_eigen_val_ind = ", usable_pairs[0][0:i+1]
            #     reduced_energy = np.sum(np.square(self.iso_eigenvals[dominant_eigen_val_ind]))
            #     if reduced_energy <= self.threshold_factor*acc_energy:
            #             ind.append(dominant_eigen_val_ind[i])
            #     else:
            #         break
            # if len(ind) == 0:
            #     ind.append(np.argmax(self.iso_eigenvals))
            # self.dominant_indices = ind

        # # Next,
        # # Get the system energy of Hessian_Product
        # system_energy = np.sum(self.iso_eigenvals)
        # ind = []

        # # get the indices of dominant eigenvalues in descending order
        # sort_ind = self.iso_eigenvals.argsort()[::-1]

        # # Check the threshold
        # for i in xrange(0, self.num_sample):
        #     dominant_eigen_val_ind = sort_ind[0:i+1]
        #     reduced_energy = np.sum(self.iso_eigenvals[dominant_eigen_val_ind])
        #     if reduced_energy <= self.threshold_factor*system_energy:
        #         ind.append(dominant_eigen_val_ind[i])
        #     else:
        #         break

        # if len(ind) == 0:
        #     ind.append(np.argmax(self.iso_eigenvals))

        # self.dominant_indices = ind
