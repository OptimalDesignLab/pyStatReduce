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

    **Constructor Arguments**

    * `threshold_factor` : fraction of threshold energy that MUST be had by an
                           approximation of the expected value of QoI.

    """

    def __init__(self, threshold_factor, **kwargs):
        self.threshold_factor = threshold_factor

        # Decide between using arnoldi-iteration or exact Hessian
        if kwargs['exact_Hessian'] == False:
            self.use_exact_Hessian = False
        else:
            self.use_exact_Hessian = True
        # TODO: Check if isoprobabilistic eigen modes can be initialized here

    def getDominantDirections(self, QoI, jdist):
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

        if self.use_exact_Hessian == True:

            # Get the Hessian of the QoI
            xi = np.zeros(QoI.systemsize)
            Hessian = QoI.eval_QoIHessian(mu, xi)

            # Compute the eigen modes of the Hessian of the Hadamard Quadratic
            Hessian_Product = np.matmul(sqrt_Sigma, np.matmul(Hessian, sqrt_Sigma))
            self.iso_eigenvals, self.iso_eigenvecs = np.linalg.eig(Hessian_Product)

            num_sample = QoI.systemsize
        else:
            # mu_iso = jdist.fwd(mu)
            # approximate the hessian of the QoI in the isoprobabilistic space
            # 1. Initialize ArnoldiSampling object
            perturbation_size = 1.e-6
            if QoI.systemsize < 20:
                num_sample = QoI.systemsize+1
            else:
                num_sample = 21
            arnoldi = ArnoldiSampling(perturbation_size, num_sample)

            # 2. Declare arrays
            # 2.1 iso-eigenmodes
            self.iso_eigenvals = np.zeros(num_sample-1)
            self.iso_eigenvecs = np.zeros([QoI.systemsize, num_sample-1])
            # 2.2 solution and function history array
            xdata = np.zeros([QoI.systemsize, num_sample])
            fdata = np.zeros(num_sample)
            gdata = np.zeros([QoI.systemsize, num_sample])
            grad_red = np.zeros(QoI.systemsize)

            # 3. Approximate eigenvalues using ArnoldiSampling
            # 3.1 Convert x into a standard normal distribution
            xdata[:,0] = 0.0
            gdata[:,0] = np.dot(QoI.eval_QoIGradient(mu, np.zeros(QoI.systemsize)),
                                sqrt_Sigma)
            dim, error_estimate = arnoldi.arnoldiSample_2_test(QoI, jdist, xdata, fdata, gdata,
                                                        self.iso_eigenvals, self.iso_eigenvecs,
                                                        grad_red)

        # Next,
        # Get the system energy of Hessian_Product
        system_energy = np.sum(self.iso_eigenvals)
        ind = []

        # get the indices of dominant eigenvalues in descending order
        sort_ind = self.iso_eigenvals.argsort()[::-1]

        # Check the threshold
        # for i in xrange(0, QoI.systemsize):
        for i in xrange(0, num_sample):
            dominant_eigen_val_ind = sort_ind[0:i+1]
            reduced_energy = np.sum(self.iso_eigenvals[dominant_eigen_val_ind])
            if reduced_energy <= self.threshold_factor*system_energy:
                ind.append(dominant_eigen_val_ind[i])
            else:
                break

        if len(ind) == 0:
            ind.append(np.argmax(self.iso_eigenvals))

        self.dominant_indices = ind


    """
    def getDominantDirections2(self, QoI, jdist):

        # Convert the joint distribution to an isoprobabilistic space
        mu = cp.E(jdist)
        mu_iso = jdist.fwd(mu)

        # approximate the hessian of the QoI in the isoprobabilistic space
        # 1. Initialize ArnoldiSampling object
        perturbation_size = 0.2
        if QoI.systemsize < 20:
            num_sample = QoI.systemsize+1
        else:
            num_sample = 20
        arnoldi = ArnoldiSampling(perturbation_size, num_sample)

        # 2. Declare arrays for iso-eigenmodes
        self.iso_eigenvals = np.zeros(QoI.systemsize)
        self.iso_eigenvecs = np.zeros([QoI.systemsize, QoI.systemsize])

        # 3. Approximate eigenvalues using ArnoldiSampling
        xdata = np.zeros([QoI.systemsize, QoI.systemsize+1])
        fdata = np.zeros(QoI.systemsize+1)
        gdata = np.zeros([QoI.systemsize, QoI.systemsize+1])
        grad_red = np.zeros(QoI.systemsize)
        xdata[:,0] = mu_iso[:]
        gdata[:,0] = QoI.eval_QoIGradient(mu, np.zeros(QoI.systemsize))
        dim, error_estimate = arnoldi.arnoldiSample_2_test(QoI, jdist, xdata, fdata, gdata,
                                                    self.iso_eigenvals, self.iso_eigenvecs,
                                                    grad_red)

        # Get the system energy of Hessian_Product
        system_energy = np.sum(self.iso_eigenvals)
        ind = []

        # get the indices of dominant eigenvalues in descending order
        sort_ind = self.iso_eigenvals.argsort()[::-1]

        # Check the threshold
        for i in xrange(0, QoI.systemsize):
            dominant_eigen_val_ind = sort_ind[0:i+1]
            reduced_energy = np.sum(self.iso_eigenvals[dominant_eigen_val_ind])
            if reduced_energy <= self.threshold_factor*system_energy:
                ind.append(dominant_eigen_val_ind[i])
            else:
                break

        if len(ind) == 0:
            ind.append(np.argmax(self.iso_eigenvals))

        self.dominant_indices = ind
    """
