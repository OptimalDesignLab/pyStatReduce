# New stochastic collocation implementation
import copy
import numpy as np
import chaospy as cp
import pystatreduce.utils

class StochasticCollocation3(object):
    """
    3rd implementation of how to do stochastic collocation, This is to handle
    multiple QoI that have different dominant directions, and/or do full
    collocation.

    Different set of random variables MUST HAVE diffrent collocation objects.
    """
    def __init__(self, jdist, distribution_type, QoI_dict, data_type=np.float):
        self.data_type = data_type
        self.n_rv = cp.E(jdist).size
        self.collocation_dict = {} # Dictionary that will hold all the
                                   # information pertaining to the different QoI
        if distribution_type == "Normal" or distribution_type == "MvNormal":
            self.collocation_dict = copy.copy(QoI_dict)
            for i in self.collocation_dict:
                quadrature_degree = self.collocation_dict[i]['quadrature_degree']
                q, w = np.polynomial.hermite.hermgauss(quadrature_degree)
                if self.collocation_dict[i]['reduced_collocation'] == False:
                    points, weights = self.getQuadratureInfo(jdist,
                                                        quadrature_degree, q, w)
                else:
                    dominant_dir = self.collocation_dict[i]['dominant_dir']
                    points, weights = self.getReducedQuadratureInfo(jdist,
                                          dominant_dir, quadrature_degree, q, w)
                self.collocation_dict[i]['points'] = points
                self.collocation_dict[i]['quadrature_weights'] = weights
                self.allocateQoISpace(self.collocation_dict[i])
        else:
            raise NotImplementedError

    def getQuadratureInfo(self, jdist, quadrature_degree, ref_collocation_pts,
    					  ref_collocation_w):
        """
        Generates the integration points at which the deterministic QoI needs
        to be evaluated, and also the corresponding quadrature weights.
        """
        n_quadrature_loops = self.n_rv
        n_points = quadrature_degree ** n_quadrature_loops
        covariance = cp.Cov(jdist)
        idx = 0
        ctr = 0
        colloc_xi_arr = np.zeros(self.n_rv)
        colloc_w_arr = np.zeros(self.n_rv)
        points = np.zeros([n_points, self.n_rv], dtype=self.data_type)
        quadrature_weights = np.zeros(n_points, dtype=self.data_type)
        sqrt_Sigma = np.sqrt(covariance)
        idx, ctr = self.__compute_quad(sqrt_Sigma, ref_collocation_pts,
                                       ref_collocation_w, colloc_xi_arr,
                                       colloc_w_arr, points,
                                       quadrature_weights, idx, ctr)
        assert idx == -1
        points += cp.E(jdist)

        return points, quadrature_weights

    def getReducedQuadratureInfo(self, jdist, dominant_dir, quadrature_degree,
                                 ref_collocation_pts, ref_collocation_w):
        """
        Generates the integration points at which the deterministic QoI needs
        to be evaluated for the reduced stochastic space, and also the
        corresponding quadrature weights.
        """
        n_quadrature_loops = dominant_dir.shape[1]
        n_points = quadrature_degree ** n_quadrature_loops
        covariance = cp.Cov(jdist)
        idx = 0
        ctr = 0
        colloc_xi_arr = np.zeros(n_quadrature_loops)
        colloc_w_arr = np.zeros(n_quadrature_loops)
        points = np.zeros([n_points, self.n_rv], dtype=self.data_type)
        quadrature_weights = np.zeros(n_points, dtype=self.data_type)
        sqrt_Sigma = np.sqrt(covariance)
        idx, ctr = self.__compute_reduced_quad(sqrt_Sigma, dominant_dir, ref_collocation_pts,
                                   ref_collocation_w, colloc_xi_arr, colloc_w_arr,
                                   points, quadrature_weights, idx, ctr)
        assert idx == -1
        points += cp.E(jdist)

        return points, quadrature_weights

    def allocateQoISpace(self, QoI_dict):
        """
        Allocates space for the output QoIs at the quadrature points.
        """
        n_points = np.size(QoI_dict['points'], 0)
        QoI_dict['fvals'] = np.zeros([n_points,
                                      QoI_dict['output_dimensions']],
                                      dtype=self.data_type)
        if QoI_dict['include_derivs'] == True:
                for j in QoI_dict['deriv_dict']:
                    QoI_dict['deriv_dict'][j]['fvals'] =  np.zeros([n_points,
                                                QoI_dict['output_dimensions'],
                                                QoI_dict['deriv_dict'][j]['output_dimensions']],
                                                dtype=self.data_type)
        return None

    def evaluateQoIs(self, jdist):
        pert = np.zeros(self.n_rv, dtype=self.data_type)

        # for i in range(0, self.n_points):
        #     for j in self.QoI_dict:
        #         QoI_func = self.QoI_dict[j]['QoI_func']
        #         self.QoI_dict[j]['fvals'][i,:] = QoI_func(self.points[i,:], pert)
        #         if include_derivs == True:
        #             for k in self.QoI_dict[j]['deriv_dict']:
        #                 dQoI_func = self.QoI_dict[j]['deriv_dict'][k]['dQoI_func']
        #                 self.QoI_dict[j]['deriv_dict'][k]['fvals'][i,:] =\
        #                 					dQoI_func(self.points[i,:], pert)


        for i in self.collocation_dict:
            QoI_func = self.collocation_dict[i]['QoI_func']
            points = self.collocation_dict[i]['points']
            n_points = np.size(points, 0)
            for j in range(0, n_points):
                self.collocation_dict[i]['fvals'][j,:] = QoI_func(points[j,:], pert)
            # Now compute derivative values if needed
            if self.collocation_dict[i]['include_derivs'] == True:
                for k in self.collocation_dict[i]['deriv_dict']:
                    dQoI_func = self.collocation_dict[i]['deriv_dict'][k]['dQoI_func']
                    for j in range(0, n_points):
                        # This implementation assumes that the derivatives and
                        # and the QoI use the same collocation order.
                        self.collocation_dict[i]['deriv_dict'][k]['fvals'][j,:] =\
                            dQoI_func(points[j,:], pert)

        return None

    def mean(self, of=None):
        """
        Compute the mean of the specified quantites of interest based on
        precomputed function values.
        """
        mean_val = {}
        for i in of:
            if i in self.collocation_dict:
                mean_val[i] = np.zeros(self.collocation_dict[i]['output_dimensions'], dtype=self.data_type)
                n_points = np.size(self.collocation_dict[i]['points'],0)
                for j in range(0, n_points):
                    mean_val[i] += self.collocation_dict[i]['fvals'][j,:] *\
                                   self.collocation_dict[i]['quadrature_weights'][j]
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
            if i in self.collocation_dict:
                qoi_dim = self.collocation_dict[i]['output_dimensions']
                variance_val[i] = np.zeros([qoi_dim, qoi_dim], dtype=self.data_type)
                n_points = np.size(self.collocation_dict[i]['points'],0)
                for j in range(0, n_points):
                    val = self.collocation_dict[i]['fvals'][j,:] - mu[i]
                    variance_val[i] += self.collocation_dict[i]['quadrature_weights'][j] *\
                                       np.outer(val, val)
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
            if i in self.collocation_dict:
                dmean_val[i] = {}
                n_points = np.size(self.collocation_dict[i]['points'],0)
                for j in wrt:
                    if j in self.collocation_dict[i]['deriv_dict']:
                        dmean_val[i][j] = np.zeros([self.collocation_dict[i]['output_dimensions'],
                                                    self.collocation_dict[i]['deriv_dict'][j]['output_dimensions']],
                                                   dtype=self.data_type)
                        for k in range(0, n_points):
                            dmean_val[i][j] += self.collocation_dict[i]['deriv_dict'][j]['fvals'][k,:] *\
                                               self.collocation_dict[i]['quadrature_weights'][k]
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
            if i in self.collocation_dict:
                dvariance_val[i] = {}
                dmu_j = self.dmean(of=of, wrt=wrt)
                n_points = np.size(self.collocation_dict[i]['points'],0)
                quadrature_weights = self.collocation_dict[i]['quadrature_weights']
                for j in wrt:
                    if j in self.collocation_dict[i]['deriv_dict']:
                        intarr = mu[i]*dmu_j[i][j]
                        val = np.zeros([self.collocation_dict[i]['output_dimensions'],
                                        self.collocation_dict[i]['deriv_dict'][j]['output_dimensions']],
                                       dtype=self.data_type)
                        for k in range(0, n_points):
                            fval = self.collocation_dict[i]['fvals'][k,:]
                            dfval = self.collocation_dict[i]['deriv_dict'][j]['fvals'][k,:]
                            val += quadrature_weights[k]*fval*dfval
                        dvariance_val[i][j] = 2*(val + mu[i]*dmu_j[i][j]*(np.sum(quadrature_weights)-2))
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
            if i in self.collocation_dict:
                dstd_dev_val[i] = {}
                for j in wrt:
                    if j in self.collocation_dict[i]['deriv_dict']:
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
            n_quadrature_loops = self.n_rv
            sqrt2 = np.sqrt(2)
            for i in range(0, ref_collocation_pts.size):
                colloc_xi_arr[idx] = ref_collocation_pts[i] # Get the array of all the locations needed
                colloc_w_arr[idx] = ref_collocation_w[i] # Get the array of all the weights needed
                actual_location[ctr,:] = sqrt2*np.dot(sqrt_Sigma, colloc_xi_arr)
                # quadrature_weights[ctr] = np.prod(colloc_w_arr) / (np.sqrt(np.pi)**self.n_quadrature_loops)
                quadrature_weights[ctr] = np.prod(colloc_w_arr) / (np.sqrt(np.pi)**n_quadrature_loops)
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
        n_quadrature_loops = dominant_dir.shape[1]
        if idx == n_quadrature_loops - 1: # self.n_quadrature_loops - 1:
            sqrt2 = np.sqrt(2)
            for i in range(0, ref_collocation_pts.size):
                colloc_xi_arr[idx] = ref_collocation_pts[i] # Get the array of all the locations needed
                colloc_w_arr[idx] = ref_collocation_w[i] # Get the array of all the weights needed
                actual_location[ctr,:] = sqrt2*np.dot(sqrt_Sigma, np.dot(dominant_dir,colloc_xi_arr))
                quadrature_weights[ctr] = np.prod(colloc_w_arr) / (np.sqrt(np.pi) ** n_quadrature_loops)
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
