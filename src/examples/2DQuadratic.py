# 2D quadratic function
import numpy as np
import numdifftools as nd
from QuantityOfInterest import QuantityOfInterest

class ellipsoid(QuantityOfInterest):

    def __init__(self, tuple):
        self.extra_args = tuple

    def eval_QoI(self, μ, ξ):
        θ = self.tuple[0]
        rotation_mat = [cos(θ) -sin(θ);
                        sin(θ) cos(θ)]
        ξ_hat = rotation_mat*ξ
        fval = 50*(x[0]+ξ_hat[0])^2 + (x[1] + ξ_hat[1])^2
        return fval

    def eval_QoIHessian(self, μ, ξ):
        func(ξ) = self.eval_QoI(μ, ξ)
        H = nd.Hessian(func)(ξ)
        return H
