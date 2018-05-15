# 2D quadratic function
import math
import numpy as np
import numdifftools as nd
from quantity_of_interest import QuantityOfInterest

class Paraboloid2D(QuantityOfInterest):

    def __init__(self, systemsize, tuple):
        QuantityOfInterest.__init__(self, systemsize)
        self.extra_args = tuple
        self.quadratic_matrix = np.diag([50,1])

    def eval_QoI(self, mu, xi):
        theta = self.extra_args[0]
        # rotation_mat = [cos(theta) -sin(theta);
        #                 sin(theta) cos(theta)]
        rotation_mat = np.array([[np.cos(theta), -np.sin(theta)],
                                 [np.sin(theta), np.cos(theta)]])
        xi_hat = rotation_mat.dot(xi)
        fval = 50*(mu[0]+xi_hat[0])**2 + (mu[1] + xi_hat[1])**2
        return fval

    def eval_QoIHessian(self, mu, xi):
        def func(xi) :
            return self.eval_QoI(mu, xi)

        H = nd.Hessian(func)(xi)
        return H

class Paraboloid3D(QuantityOfInterest):

    def __init__(self, systemsize):
        QuantityOfInterest.__init__(self, systemsize)
        self.quadratic_matrix = np.diag([50, 25, 1])

    def eval_QoI(self, mu, xi):
        return 50*(mu[0]+xi[0])**2 + 25*(mu[1] + xi[1])**2 + (mu[2] + xi[2])**2

    def eval_QoIGradient(self, mu, xi):
        rv = mu + xi
        grad = np.array([100*rv[0], 50*rv[1], 2*rv[2]])
        return grad


class Paraboloid5D(QuantityOfInterest):

    def __init__(self, systemsize):
        QuantityOfInterest.__init__(self, systemsize)
        self.quadratic_matrix = np.diag([50, 40, 30, 20, 1])

    def eval_QoI(self, mu, xi):

        fval = 50*(mu[0]+xi[0])**2 + 40*(mu[1]+xi[1])**2 + 30*(mu[2]+xi[2])**2 \
               + 20*(mu[3]+xi[3])**2 + (mu[4]+xi[4])**2
        return fval
