# QuantityOfInterest.py

class QuantityOfInterest(object):

    def __init__(self, nvariables):
        assert nvariables > 0, "Quantity of Interest cannot have zero inputs"
        self.systemsize = nvariables

    def eval_QoI(self, mu, xi):
        raise NotImplementedError

    def eval_QoIHessian(self, mu, xi):
        raise NotImplementedError
