# New stochastic collocation implementation
import numpy as np
import chaospy as cp
import pystatreduce.utils

class StochasticCollocation2(object):
    """
    Newer implementation of stochastic collocation that can handle multiple
    QoI's at once
    """
    def __init__(self, jdist, quadrature_degree, distribution_type, QoI_dict,
                 include_derivs=False, reduced_collocation=False,
                 data_type=np.float):
        assert quadrature_degree > 0, "Need at least 1 collocation point for \
                                        uncertainty propagation"
        self.n_rv = cp.E(jdist).size
        self.QoI_dict = QoI_dict
        self.distribution_type = distribution_type
        self.data_type = data_type

        # Get the 1D quadrature points based on the distribution being
        # approximated
        if distribution_type == "Normal" or distribution_type == "MvNormal":
            self.q, self.w = np.polynomial.hermite.hermgauss(quadrature_degree)
            self.getQuadratureInfo(jdist, quadrature_degree, self.q, self.w)
            self.allocateQoISpace(include_derivs)
        else:
            raise NotImplementedError

    def getQuadratureInfo(self, jdist, quadrature_degree, ref_collocation_pts,
    					  ref_collocation_w):
        """
        Generates the integration points at which the deterministic QoI needs
        to be evaluated, and also the corresponding quadrature weights.
        """
        self.n_points = quadrature_degree**self.n_rv
        covariance = cp.Cov(jdist)
        idx = 0
        ctr = 0
        colloc_xi_arr = np.zeros(self.n_rv)
        colloc_w_arr = np.zeros(self.n_rv)
        self.points = np.zeros([self.n_points, self.n_rv], dtype=self.data_type)
        self.quadrature_weights = np.zeros(self.n_points, dtype=self.data_type)
        sqrt_Sigma = np.sqrt(covariance)
        idx, ctr = self.__compute_quad(sqrt_Sigma, ref_collocation_pts,
                                       ref_collocation_w, colloc_xi_arr,
                                       colloc_w_arr, self.points,
                                       self.quadrature_weights, idx, ctr)
        assert idx == -1
        self.points += cp.E(jdist)

    def allocateQoISpace(self, include_derivs):
        """
        Allocates space for the output QoIs at the quadrature points.
        """
        for i in self.QoI_dict:
            self.QoI_dict[i]['fvals'] = np.zeros([self.n_points,
                                        self.QoI_dict[i]['output_dimensions']],
                                        dtype=self.data_type)
        if include_derivs == True:
            for i in self.QoI_dict:
                for j in self.QoI_dict[i]['deriv_dict']:
                    self.QoI_dict[i]['deriv_dict'][j]['fvals'] =  np.zeros([self.n_points,
                                                self.QoI_dict[i]['deriv_dict'][j]['output_dimensions']],
                                                dtype=self.data_type)

    def evaluateQoIs(self, jdist, include_derivs=False):
        pert = np.zeros(self.n_rv, dtype=self.data_type)
        for i in range(0, self.n_points):
            for j in self.QoI_dict:
                QoI_func = self.QoI_dict[j]['QoI_func']
                self.QoI_dict[j]['fvals'][i,:] = QoI_func(self.points[i,:], pert)
                if include_derivs == True:
                    for k in self.QoI_dict[j]['deriv_dict']:
                        dQoI_func = self.QoI_dict[j]['deriv_dict'][k]['dQoI_func']
                        self.QoI_dict[j]['deriv_dict'][k]['fvals'][i,:] =\
                        					dQoI_func(self.points[i,:], pert)

    def mean(self, of=None):
        """
        Compute the mean of the specified quantites of interest based on
        precomputed function values.
        """
        mean_val = {}
        for i in of:
            if i in self.QoI_dict:
                mean_val[i] = np.zeros(self.QoI_dict[i]['output_dimensions'])
                for j in range(0, self.n_points):
                    mean_val[i] += self.QoI_dict[i]['fvals'][j,:] * self.quadrature_weights[j]
                mean_val[i] = mean_val[i] / (np.sqrt(np.pi)**self.n_rv)
        return mean_val

    def variance(self, of=None):
        """
        Compute the variance of the specified quantites of interest based on
        precomputed function values.
        """
        variance_val = {}
        mu = self.mean(of=of) # np.sum(self.QoI_dict[i]['fvals'] * self.quadrature_weights)
        for i in of:
            if i in self.QoI_dict:
                qoi_dim = self.QoI_dict[i]['output_dimensions']
                variance_val[i] = np.zeros([qoi_dim, qoi_dim])
                for j in range(0, self.n_points):
                    val = self.QoI_dict[i]['fvals'][j,:] - mu[i]
                    variance_val[i] += self.quadrature_weights[j]*np.outer(val, val)
                variance_val[i] = variance_val[i] / (np.sqrt(np.pi)**self.n_rv)
        return variance_val

    def __compute_quad(self, sqrt_Sigma, ref_collocation_pts, ref_collocation_w,
    				   colloc_xi_arr, colloc_w_arr, actual_location,
    				   quadrature_weights, idx, ctr):
        """
        The recursive function that actually does the heavy lifting for
        getQuadratureInfo.
        """
        if idx == colloc_xi_arr.size-1:
            sqrt2 = np.sqrt(2)
            for i in range(0, ref_collocation_pts.size):
                colloc_xi_arr[idx] = ref_collocation_pts[i] # Get the array of all the locations needed
                colloc_w_arr[idx] = ref_collocation_w[i] # Get the array of all the weights needed
                actual_location[ctr,:] = sqrt2*np.dot(sqrt_Sigma, colloc_xi_arr)
                quadrature_weights[ctr] = np.prod(colloc_w_arr)#  / (np.sqrt(np.pi)**self.n_rv)
                ctr += 1
            return idx-1, ctr
        else:
            for i in range(0, ref_collocation_pts.size):
                colloc_xi_arr[idx] = ref_collocation_pts[i]
                colloc_w_arr[idx] = ref_collocation_w[i]
                idx, ctr = self.__compute_quad(sqrt_Sigma, ref_collocation_pts,
                		   					   ref_collocation_w, colloc_xi_arr,
                		   					   colloc_w_arr, actual_location,
    				   						   quadrature_weights, idx+1, ctr)
            return idx-1, ctr

    """
    def __compute_perturbation(self, sqrt_Sigma, ref_collocation_pts, colloc_xi_arr,
                             actual_location, idx, ctr):

    # The recursive function that actually does all of the heavylifting for
    # the function getQuadPts.

        if idx == colloc_xi_arr.size-1:
            sqrt2 = np.sqrt(2)
            for i in range(0, ref_collocation_pts.size):
                colloc_xi_arr[idx] = ref_collocation_pts[i] # Get the array of all the locations needed
                actual_location[ctr,:] = sqrt2*np.dot(sqrt_Sigma, colloc_xi_arr)
                ctr += 1
            return idx-1, ctr
        else:
            for i in range(0, ref_collocation_pts.size):
                colloc_xi_arr[idx] = ref_collocation_pts[i]
                idx, ctr = self.__compute_perturbation(sqrt_Sigma, ref_collocation_pts,
                                        colloc_xi_arr,
                                        actual_location,
                                        idx+1, ctr)
            return idx-1, ctr

    def __compute_weights(self, ref_collocation_w, colloc_w_arr,
    					  quadrature_weights, idx, ctr):

    # Recursive function that computes the specific weights corresponding to
    # the multidimensional quadrature point. This code is specific to normal
    # distribution.

    	 if idx == colloc_w_arr.size-1:
            for i in xrange(0, ref_collocation_w.size):
                colloc_w_arr[idx] = ref_collocation_w[i] # Get the array of all the weights needed
                quadrature_weights[ctr] = np.prod(colloc_w_arr)
                ctr += 1
            return idx-1, ctr
        else:
            for i in xrange(0, ref_collocation_w.size):
                colloc_w_arr[idx] = ref_collocation_w[i]
                idx, ctr = self.compute_weights(ref_collocation_w, colloc_w_arr,
                                                quadrature_weights,
                                                idx+1, ctr)
    return idx-1, ctr
    """
