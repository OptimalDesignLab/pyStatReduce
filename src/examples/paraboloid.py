# 2D quadratic function
import math
import numpy as np
import numdifftools as nd
from QuantityOfInterest import QuantityOfInterest

class Paraboloid2D(QuantityOfInterest):

    def __init__(self, tuple):
        self.extra_args = tuple

    def eval_QoI(self, mu, xi):
        theta = self.extra_args[0]
        # rotation_mat = [cos(theta) -sin(theta);
        #                 sin(theta) cos(theta)]
        rotation_mat = np.array([[math.cos(theta), -math.sin(theta)],
                                 [math.sin(theta), math.cos(theta)]])
        xi_hat = rotation_mat.dot(xi)
        fval = 50*(mu[0]+xi_hat[0])**2 + (mu[1] + xi_hat[1])**2
        return fval

    def eval_QoIHessian(self, mu, xi):
        def func(xi) :
            return self.eval_QoI(mu, xi)

        H = nd.Hessian(func)(xi)
        return H
