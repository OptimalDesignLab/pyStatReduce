# Dimension Reduction

import numpy as np
import chaospy as cp

class DimensionReduction(object):

    def __init__(self, threshold_factor):
        self.threshold_factor = threshold_factor
        # TODO: Check if isoprobabilistic eigen modes can be initialized here

    def getDominantDirections(self, QoI, jdist):

        mu = cp.E(jdist)
        covariance = cp.Cov(jdist)

        # Check if variance covariance matrix is diagonal
        if np.count_nonzero(covariance - np.diag(np.diagonal(covariance))) == 0:
            sqrt_Sigma = np.sqrt(covariance)
        else:
            raise NotImplementedError

        # Get the Hessian of the QoI
        xi = np.zeros(QoI.systemsize)
        Hessian = QoI.eval_QoIHessian(mu, xi)

        # Compute the eigen modes of the Hessian of the Hadamard Quadratic
        Hessian_Product = np.matmul(sqrt_Sigma, np.matmul(Hessian, sqrt_Sigma))
        self.iso_eigen_vals, self.iso_eigen_vectors = np.linalg.eig(Hessian_Product)

        # Get the system energy of Hessian_Product
        system_energy = np.sum(self.iso_eigen_vals)
        ind = []

        # get the indices of dominant eigenvalues in descending order
        sort_ind = self.iso_eigen_vals.argsort()[::-1]

        # Check the threshold
        for i in xrange(0, QoI.systemsize):
            dominant_eigen_val_ind = sort_ind[0:i+1]
            reduced_energy = np.sum(self.iso_eigen_vals[dominant_eigen_val_ind])
            if reduced_energy <= self.threshold_factor*system_energy:
                ind.append(dominant_eigen_val_ind[i])
            else:
                break

        if len(ind) == 0:
            ind.append(np.argmax(self.iso_eigen_vals))

        self.dominant_indices = ind
