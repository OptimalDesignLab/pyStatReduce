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

    def normal(x, sigma, QoI, collocation_obj):

        systemsize = x.size
        ref_collocation_pts = collocation_obj.q
        ref_collocation_w = collocation_obj.w
        idx = 1
        colloc_xi_arr = np.zeros(systemsize)
        colloc_w_arr = np.zeros(systemsize)
        mu_j = 0.0
        idx = doCollocation(x, sigma, mu_j, ref_collocation_pts, ref_collocation_w,
                            QoI, colloc_xi_arr, colloc_w_arr, idx)
        assert idx == 0
        mu_j = mu_j/(sqrt(pi)^systemsize)

        return mu_j


    def uniform():
        raise NotImplementedError

    def doNormalCollocation(x, sigma, mu_j, xi, w, QoI, colloc_xi_arr,
                            colloc_w_arr, idx):

        if idx == x.size-1:
            for i in xrange(0, xi.size):
                colloc_w_arr[idx] = w[i] # Get the array of all the weights needed
                colloc_xi_arr[idx] = xi[i] # Get the array of all the locations needed
                fval = QoI.eval_QoI(x, sqrt(2)*sigma*colloc_xi_arr)
                mu_j += np.prod(colloc_w_arr)*fval
        else:
            # for i = 0:(xi.size-1)
            for i in xrange(0, xi.size):
                colloc_xi_arr[idx-1] = xi[i]
                colloc_w_arr[idx-1] = w[i]
                idx = doCollocation(x, sigma, mu_j, xi, w, QoI, colloc_xi_arr,
                                    colloc_w_arr, idx+1)
