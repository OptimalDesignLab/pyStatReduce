import math
import numpy as np
import numdifftools as nd
from pystatreduce.quantity_of_interest import QuantityOfInterest

class ConstantFunction(QuantityOfInterest):

    def __init__(self, systemsize):
        QuantityOfInterest.__init__(self, systemsize)

    def eval_QoI(self, mu, xi):
        return 5

    def eval_QoIGradient(self, mu, xi):
        return 0.0

    def eval_QoIHessian(self, mu, xi):
        return 0.0

class LinearFunction(QuantityOfInterest):

    def __init__(self, systemsize):
        QuantityOfInterest.__init__(self, systemsize)
        self.A = np.random.rand(systemsize)

    def eval_QoI(self, mu, xi):
        x = mu + xi
        return self.A.dot(x)

    def eval_QoIGradient(self, mu, xi):
        return self.A
