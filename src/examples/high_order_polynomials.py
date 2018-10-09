# 2D 4th order nonlinear function
import math
import numpy as np
import numdifftools as nd
from quantity_of_interest import QuantityOfInterest

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
