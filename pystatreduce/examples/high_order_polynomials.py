# 2D 4th order nonlinear function
import math
import numpy as np
import numdifftools as nd
from pystatreduce.quantity_of_interest import QuantityOfInterest

class Poly2DOrder4(QuantityOfInterest):

    def __init__(self, systemsize=2):
        QuantityOfInterest.__init__(self, systemsize)
        self.local_minima = np.zeros(2)
        self.global_minima = 0.5*np.array([-3-np.sqrt(7),-3-np.sqrt(7)])
        self.saddle_point = 0.5*np.array([-3+np.sqrt(7),-3+np.sqrt(7)])

    def eval_QoI(self, mu, xi):
        rv = mu + xi
        fval = 1.5*(rv[0]**2) + rv[1]**2 - 2*rv[0]*rv[1] + 2*(rv[0]**3) + 0.5*(rv[0]**4)
        return fval

    def eval_QoIGradient(self, mu, xi):
        rv = mu + xi
        grad = np.zeros(2)
        grad[0] = 3*rv[0] - 2*rv[1] + 6*(rv[0]**2) + 2*(rv[0]**3)
        grad[1] = 2*(rv[1] - rv[0])
        return grad

    def eval_QoIHessian(self, mu, xi):
        rv = mu + xi
        hess = np.zeros([2,2])
        hess[0,0] = 3.0 + 12*rv[0] + 6*(rv[0]**2)
        hess[0,1] = -2.0
        hess[1,0] = -2.0
        hess[1,1] = 2.0
        return hess

class PolyRVDV(QuantityOfInterest):
    """
    Nonlinear polynomial (Poly) that has random variables (RV) and independent
    parameters (DV). This serves as an example case for unittests.
    """
    def __init__(self, systemsize=2, n_parameters=2, data_type=np.float):
        QuantityOfInterest.__init__(self, systemsize)
        self.data_type = data_type
        self.n_parameters = n_parameters
        self.dv = np.ones(n_parameters, dtype=self.data_type)

    def eval_QoI(self, mu, xi):
        rv = mu + xi
        fval = 50 * (rv[0]**2) * (self.dv[0]**2) + 2 * (rv[1]**2) * (self.dv[1]**2)
        return fval

    def eval_QoIGradient(self, mu, xi):
        rv = mu + xi
        grad = np.zeros(self.systemsize, dtype=self.data_type)
        grad[0] = 100 * rv[0] * (self.dv[0]**2)
        grad[1] = 4 * rv[1] * (self.dv[1]**2)

    def eval_QoIHessian(self, mu, xi):
        rv = mu + xi
        hess = np.zeros([self.systemsize, self.systemsize], dtype=self.data_type)
        hess[0,0] = 100 * (self.dv[0]**2)
        hess[1,1] = 4 * (self.dv[1]**2)
        return hess

    def set_dv(self, new_dv):
        self.dv[:] = new_dv

    def eval_QoIGradient_dv(self, mu, xi):
        rv = mu + xi
        grad = np.zeros(self.n_parameters, dtype=self.data_type)
        grad[0] = 100 * (rv[0]**2) * self.dv[0]
        grad[1] = 4 * (rv[1]**2) * self.dv[1]
        return grad
