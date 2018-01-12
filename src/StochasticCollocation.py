# StochasticCollocation.py
import numpy as np
import chaospy as cp

class StochasticCollocation(object):

    def __init__(self, degree, distribution_type):
        assert degree > 0, "Need at least 1 collocation point for \
                                        uncertainty propagation"
        if distribution_type == "Normal" or distribution_type == "MvNormal":
            self.q, self.w = np.polynomial.hermite.hermgauss(degree)
        elif distribution_type == "Uniform":
            self.q, self.w = np.polynomial.legendre.leggauss(degree)
        else:
            raise NotImplementedError

    def normal(self, x, sigma, QoI):

        systemsize = x.size
        ref_collocation_pts = self.q
        ref_collocation_w = self.w
        idx = 0
        colloc_xi_arr = np.zeros(systemsize)
        colloc_w_arr = np.zeros(systemsize)
        mu_j = np.zeros(1)
        idx = self.doNormalCollocation(x, sigma, mu_j, ref_collocation_pts,
                                       ref_collocation_w, QoI, colloc_xi_arr,
                                       colloc_w_arr, idx)
        assert idx == -1
        mu_j[0] = mu_j[0]/(np.sqrt(np.pi)**systemsize)

        return mu_j[0]

    def normalReduced(self, QoI, jdist, dominant_space):

        x = cp.E(jdist)
        covariance = cp.Cov(jdist)
        n_quadrature_loops = len(dominant_space.dominant_indices)
        dominant_dir = dominant_space.iso_eigen_vectors[:, dominant_space.dominant_indices]
        ref_collocation_pts = self.q
        ref_collocation_w = self.w
        idx = 0
        colloc_xi_arr = np.zeros(n_quadrature_loops)
        colloc_w_arr = np.zeros(n_quadrature_loops)
        mu_j = np.zeros(1)

        idx = self.doReducedNormalCollocation(x, covariance, dominant_dir, mu_j,
                                              ref_collocation_pts,
                                              ref_collocation_w, QoI, colloc_xi_arr,
                                              colloc_w_arr, idx)

        assert idx == -1
        mu_j[0] = mu_j[0]/(np.sqrt(np.pi)**n_quadrature_loops)

        return mu_j[0]

    def uniform():
        raise NotImplementedError

    def doNormalCollocation(self, x, sigma, mu_j, xi, w, QoI, colloc_xi_arr,
                            colloc_w_arr, idx):

        if idx == x.size-1:
            sqrt2 = np.sqrt(2)
            for i in xrange(0, xi.size):
                colloc_w_arr[idx] = w[i] # Get the array of all the weights needed
                colloc_xi_arr[idx] = xi[i] # Get the array of all the locations needed
                fval = QoI.eval_QoI(x, sqrt2*sigma*colloc_xi_arr)
                mu_j[0] += np.prod(colloc_w_arr)*fval
            return idx-1
        else:
            for i in xrange(0, xi.size):
                colloc_xi_arr[idx] = xi[i]
                colloc_w_arr[idx] = w[i]
                idx = self.doNormalCollocation(x, sigma, mu_j, xi, w, QoI, colloc_xi_arr,
                                    colloc_w_arr, idx+1)
            return idx-1


    def doReducedNormalCollocation(self, x, covariance, dominant_dir, mu_j, xi, w, QoI,
                                   colloc_xi_arr, colloc_w_arr, idx):

        if idx == np.size(dominant_dir,1)-1:  # x.size-1: # TODO: Change to nquadrature loops
            sqrt2 = np.sqrt(2)
            sqrt_Sigma = np.sqrt(covariance)
            for i in xrange(0, xi.size):
                colloc_w_arr[idx] = w[i] # Get the array of all the weights needed
                colloc_xi_arr[idx] = xi[i] # Get the array of all the locations needed
                q = sqrt2* np.matmul(sqrt_Sigma, dominant_dir.dot(colloc_xi_arr))
                # fval = QoI.eval_QoI(x, sqrt2*sigma*dominant_dir.dot(colloc_xi_arr))
                fval = QoI.eval_QoI(x, q)
                mu_j[0] += np.prod(colloc_w_arr)*fval
            return idx-1
        else:
            for i in xrange(0, xi.size):
                colloc_xi_arr[idx] = xi[i]
                colloc_w_arr[idx] = w[i]
                idx = self.doReducedNormalCollocation(x, covariance,
                      dominant_dir, mu_j, xi, w, QoI, colloc_xi_arr,
                      colloc_w_arr, idx+1)
            return idx-1
