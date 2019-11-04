# exponential_function.py
import numpy as np
import numdifftools as nd
from pystatreduce.quantity_of_interest import QuantityOfInterest

class ExponentialFunction(QuantityOfInterest):

    def __init__(self, systemsize):
        """
        Compute an exponential function of the form

            f = exp(a^T * a)

        where `a` is computed as

            x = [ a[0], a[1]/2, a[2]/3, ..., a[systemsize-1]/systemsize ]

        """
        QuantityOfInterest.__init__(self, systemsize)

    def eval_QoI(self, mu, xi):
        x = mu + xi
        xs = x/np.arange(1, self.systemsize+1)
        f = np.exp(np.dot(xs,xs))

        return f

    def eval_QoIGradient(self, mu, xi):
        x = mu + xi
        dfdx = np.zeros(self.systemsize)
        xs = x/np.arange(1, self.systemsize+1)
        f = np.exp(np.dot(xs,xs))
        for i in range(0, self.systemsize):
            dfdx[i] = 2.0*x[i]*f / ((i+1)*(i+1))

        return dfdx

    def eval_QoIHessian(self, mu, xi):
        """
        Use finite difference to compute Hessian.
        """
        def func(xi) :
            return self.eval_QoI(mu, xi)

        H = nd.Hessian(func)(xi)
        return H

class Exp_07xp03y(QuantityOfInterest):

    def __init__(self, systemsize):
        QuantityOfInterest.__init__(self, systemsize)
        self.a = np.array([0.7, 0.3])

    def eval_QoI(self, mu, xi):
        rv = mu + xi
        fval = np.exp(0.7*rv[0] + 0.3*rv[1])
        return fval

    def eval_QoIGradient(self, mu, xi):
        rv = mu + xi
        fval = self.eval_QoI(mu, xi)
        dfval = fval * np.array([0.7, 0.3])
        return dfval

    def eval_QoIHessian(self, mu, xi):
        def func(xi):
            return self.eval_QoI(mu, xi)

        H = nd.Hessian(func)(xi)
        return H
