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
				 include_derivs=False, reduced_collocation=True,
				 data_type=np.float):
		assert quadrature_degree > 0, "Need at least 1 collocation point for \
                                        uncertainty propagation"
        self.n_rv = cp.E(jdist).size
        self.QoI_dict = QoI_dict
        self.distribution_type = distribution_type

        # Get the 1D quadrature points based on the distribution being 
        # approximated
        if distribution_type == "Normal" or distribution_type == "MvNormal":
        	self.q, self.w = np.polynomial.hermite.hermgauss(quadrature_degree)
        	self.getQuadPts(jdist, quadrature_degree, self.q)
        else:
        	raise NotImplementedError

    def getQuadPts(self, jdist, quadrature_degree, ref_collocation_pts):
    	"""
		Generates the integration points at which the deterministic QoI needs
		to be evaluated.
    	"""
    	self.n_points = quadrature_degree**self.n_rv
    	covariance = cp.Cov(jdist)
    	idx = 0
    	ctr = 0
    	colloc_xi_arr = np.zeros(self.n_rv)
    	self.points = np.zeros([self.n_points, self.n_rv])
    	sqrt_Sigma = np.sqrt(covariance)
    	idx, ctr = self.__compute_perturbation(sqrt_Sigma, ref_collocation_pts,
    										   colloc_xi_arr, self.points, idx,
    										   ctr)
        assert idx == -1

    def evaluateQoIs(self, jdist, include_derivs=False):
    	pass

    def __compute_perturbation(self, sqrt_Sigma, ref_collocation_pts, colloc_xi_arr,
                             actual_location, idx, ctr):
    	"""
		The recursive function that actually does all of the heavylifting for 
		the function getQuadPts.
    	"""
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