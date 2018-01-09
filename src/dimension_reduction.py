# Dimension Reduction

import numpy as np
import chaospy as cp

class DimensionReduction(object):

    def __init__(self, threshold_factor):
        self.threshold_factor = threshold_factor

    def getDominantDistribution(self, QoI, jdist):

        mu = cp.E(jdist)
        covariance = cp.Cov(jdist)

        # Check if variance covariance matrix is diagonal
        if np.count_nonzero(covariance - np.diag(np.diagonal(covariance))) == 0:
            sqrt_Sigma = np.sqrt(Sigma)
        else:
            raise NotImplementedError

        # Get the Hessian of the QoI
        xi = np.zeros(QoI.systemsize)
        Hessian = QoI.eval_QoIHessian(mu, xi)

        # Compute the eigen modes of the Hessian of the Hadamard Quadratic
        Hessian_Product = np.matmul(sqrt_Sima, np.matmul(H, sqrt_Sigma))
        eigen_vals, eigen_vectors = np.linalg.eig(Hessian_Product)

        # Get the system energy of Hessian_Product
        system_energy = np.sum(eigen_vals)
        ind = []

        # get the indices of dominant eigenvalues in descending order
        sort_ind = np.argsort(eigen_vals)

        # Check the threshold
        for i in xrange(0, QoI.systemsize):
            dominant_eigen_val_ind = sort_ind[0:i]
            reduced_energy = np.sum(eigen_vals[dominant_eigen_val_ind])
            if reduced_energy <= self.threshold_factor*system_energy:
                # ind = np.append(ind, dominant_eigen_val_ind[i])
                ind.append(dominant_eigen_val_ind[i])
            else:
                break

        if len(ind) == 0:
            ind.append(max(eigen_vals))

        return eigen_vals, eigen_vectors, ind
