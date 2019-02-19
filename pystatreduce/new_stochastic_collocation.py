# New stochastic collocation implementation
import copy
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
                 dominant_dir=None, data_type=np.float):
        assert quadrature_degree > 0, "Need at least 1 collocation point for \
                                        uncertainty propagation"
        self.n_rv = cp.E(jdist).size
        self.QoI_dict = copy.copy(QoI_dict) # We don't
        self.distribution_type = distribution_type
        self.data_type = data_type

        # Get the 1D quadrature points based on the distribution being
        # approximated
        if distribution_type == "Normal" or distribution_type == "MvNormal":
            self.q, self.w = np.polynomial.hermite.hermgauss(quadrature_degree)
            if reduced_collocation == False:
                self.getQuadratureInfo(jdist, quadrature_degree, self.q, self.w)
            else:
                self.getReducedQuadratureInfo(jdist, dominant_dir, quadrature_degree, self.q, self.w)

            self.allocateQoISpace(include_derivs)
        else:
            raise NotImplementedError

    def getQuadratureInfo(self, jdist, quadrature_degree, ref_collocation_pts,
    					  ref_collocation_w):
        """
        Generates the integration points at which the deterministic QoI needs
        to be evaluated, and also the corresponding quadrature weights.
        """
        self.n_quadrature_loops = self.n_rv
        self.n_points = quadrature_degree**self.n_quadrature_loops
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

    def getReducedQuadratureInfo(self, jdist, dominant_dir, quadrature_degree,
                                 ref_collocation_pts, ref_collocation_w):
        """
        Generates the integration points at which the deterministic QoI needs
        to be evaluated for the reduced stochastic space, and also the
        corresponding quadrature weights.
        """
        self.n_quadrature_loops = dominant_dir.shape[1]
        self.n_points = quadrature_degree**self.n_quadrature_loops
        covariance = cp.Cov(jdist)
        idx = 0
        ctr = 0
        colloc_xi_arr = np.zeros(self.n_quadrature_loops)
        colloc_w_arr = np.zeros(self.n_quadrature_loops)
        self.points = np.zeros([self.n_points, self.n_rv], dtype=self.data_type)
        self.quadrature_weights = np.zeros(self.n_points, dtype=self.data_type)
        sqrt_Sigma = np.sqrt(covariance)
        idx, ctr = self.__compute_reduced_quad(sqrt_Sigma, dominant_dir, ref_collocation_pts,
                                   ref_collocation_w, colloc_xi_arr, colloc_w_arr,
                                   self.points, self.quadrature_weights, idx, ctr)
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
                                                self.QoI_dict[i]['output_dimensions'],
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
                mean_val[i] = np.zeros(self.QoI_dict[i]['output_dimensions'], dtype=self.data_type)
                for j in range(0, self.n_points):
                    mean_val[i] += self.QoI_dict[i]['fvals'][j,:] * self.quadrature_weights[j]
                mean_val[i] = mean_val[i]
        return mean_val

    def variance(self, of=None):
        """
        Compute the variance of the specified quantites of interest based on
        precomputed function values.
        """
        variance_val = {}
        mu = self.mean(of=of)
        for i in of:
            if i in self.QoI_dict:
                qoi_dim = self.QoI_dict[i]['output_dimensions']
                variance_val[i] = np.zeros([qoi_dim, qoi_dim], dtype=self.data_type)
                for j in range(0, self.n_points):
                    val = self.QoI_dict[i]['fvals'][j,:] - mu[i]
                    variance_val[i] += self.quadrature_weights[j]*np.outer(val, val)
                variance_val[i] = variance_val[i]#  / (np.sqrt(np.pi)**self.n_rv)
        return variance_val

    def dmean(self, of=None, wrt=None):
        """
        Compute the derivative of the mean of a given QoI w.r.t an input variable.
        It doesn't necessarily have to be a random variable. It can be any
        independent parameter.
        """
        dmean_val = {}
        for i in of:
            if i in self.QoI_dict:
                dmean_val[i] = {}
                for j in wrt:
                    if j in self.QoI_dict[i]['deriv_dict']:
                        dmean_val[i][j] = np.zeros([self.QoI_dict[i]['output_dimensions'],
                                                    self.QoI_dict[i]['deriv_dict'][j]['output_dimensions']],
                                                   dtype=self.data_type)
                        for k in range(0, self.n_points):
                            dmean_val[i][j] += self.QoI_dict[i]['deriv_dict'][j]['fvals'][k,:] *\
                                                self.quadrature_weights[k]
                        dmean_val[i][j] = dmean_val[i][j]
        return dmean_val

    def dvariance(self, of=None, wrt=None):
        """
        Compute the derivative of the variance of a given QoI w.r.t an input variable.
        It doesn't necessarily have to be a random variable. It can be any
        independent parameter.
        **This implementation is ONLY for scalar QoI**
        """
        dvariance_val = {}
        mu = self.mean(of=of)
        for i in of:
            if i in self.QoI_dict:
                dvariance_val[i] = {}
                dmu_j = self.dmean(of=of, wrt=wrt)
                for j in wrt:
                    if j in self.QoI_dict[i]['deriv_dict']:
                        intarr = mu[i]*dmu_j[i][j]
                        val = np.zeros([self.QoI_dict[i]['output_dimensions'],
                                        self.QoI_dict[i]['deriv_dict'][j]['output_dimensions']],
                                       dtype=self.data_type)
                        for k in range(0, self.n_points):
                            fval = self.QoI_dict[i]['fvals'][k,:]
                            dfval = self.QoI_dict[i]['deriv_dict'][j]['fvals'][k,:]
                            val += self.quadrature_weights[k]*fval*dfval
                        dvariance_val[i][j] = 2*(val + mu[i]*dmu_j[i][j]*(np.sum(self.quadrature_weights)-2))
        return dvariance_val

    def dStdDev(self, of=None, wrt=None):
        """
        Compute the derivative of the standard deviation of a given QoI w.r.t an
        input variable. It doesn't necessarily have to be a random variable. It
        can be any independent parameter.
        **This implementation is ONLY for scalar QoI**
        """
        dstd_dev_val = {}
        var = self.variance(of=of)
        dvar = self.dvariance(of=of, wrt=wrt)
        for i in of:
            if i in self.QoI_dict:
                dstd_dev_val[i] = {}
                for j in wrt:
                    if j in self.QoI_dict[i]['deriv_dict']:
                        dstd_dev_val[i][j] = 0.5 * dvar[i][j] / np.sqrt(var[i])
        return dstd_dev_val

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
                quadrature_weights[ctr] = np.prod(colloc_w_arr) / (np.sqrt(np.pi)**self.n_quadrature_loops)
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

    def __compute_reduced_quad(self, sqrt_Sigma, dominant_dir, ref_collocation_pts,
                               ref_collocation_w, colloc_xi_arr, colloc_w_arr,
                               actual_location, quadrature_weights, idx, ctr):
        """
        The recursive function that actually does the heavy lifting for
        getReducedQuadratureInfo.
        """
        if idx == self.n_quadrature_loops - 1:
            sqrt2 = np.sqrt(2)
            for i in range(0, ref_collocation_pts.size):
                colloc_xi_arr[idx] = ref_collocation_pts[i] # Get the array of all the locations needed
                colloc_w_arr[idx] = ref_collocation_w[i] # Get the array of all the weights needed
                actual_location[ctr,:] = sqrt2*np.dot(sqrt_Sigma, np.dot(dominant_dir,colloc_xi_arr))
                quadrature_weights[ctr] = np.prod(colloc_w_arr) / (np.sqrt(np.pi)**self.n_quadrature_loops)
                ctr += 1
            return idx-1, ctr
        else:
            for i in range(0, ref_collocation_pts.size):
                colloc_xi_arr[idx] = ref_collocation_pts[i]
                colloc_w_arr[idx] = ref_collocation_w[i]
                idx, ctr = self.__compute_reduced_quad(sqrt_Sigma, dominant_dir,
                                ref_collocation_pts, ref_collocation_w,
                                colloc_xi_arr, colloc_w_arr, actual_location,
                                quadrature_weights, idx+1, ctr)
            return idx-1, ctr
