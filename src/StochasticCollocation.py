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

    def normal(self, x, sigma, QoI, collocation_obj):

        systemsize = x.size
        ref_collocation_pts = collocation_obj.q
        ref_collocation_w = collocation_obj.w
        idx = 0
        colloc_xi_arr = np.zeros(systemsize)
        colloc_w_arr = np.zeros(systemsize)
        mu_j = np.zeros(1)
        idx = self.doNormalCollocation(x, sigma, mu_j, ref_collocation_pts,
                                       ref_collocation_w, QoI, colloc_xi_arr,
                                       colloc_w_arr, idx)
        # print(idx)
        assert idx == -1
        mu_j[0] = mu_j[0]/(np.sqrt(np.pi)**systemsize)

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
