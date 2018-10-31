# exponential_function.py
import numpy as np
import numdifftools as nd
from pystatreduce.quantity_of_interest import QuantityOfInterest

class ExponentialFunction(QuantityOfInterest):

    def __init__(self, systemsize):
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
