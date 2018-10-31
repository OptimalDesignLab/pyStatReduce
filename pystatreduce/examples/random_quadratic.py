import numpy as np
import numdifftools as nd
from pystatreduce.quantity_of_interest import QuantityOfInterest

class RandomQuadratic(QuantityOfInterest):
    """
    Class for computing an arbitrary quadratic of a defined size which is of
    the form

            f(x) = 0.5* x^T *V*E*V^T * x

    where V is generated using a QR factorization of a randomly generated
    matrix, E is a randomly generated vector and, depending on the kwarg
    'positive_definite' has its first index set to zero. Finally, x is a random
    variable that can be decomposed into its mean value, mu, and pertrurbation
    about its mean, xi.

    """

    def __init__(self, systemsize, **kwargs):
        QuantityOfInterest.__init__(self, systemsize)
        self.V, self.R = np.linalg.qr(np.random.rand(systemsize, systemsize))
        self.E = np.sort(np.random.rand(systemsize))

        if kwargs['positive_definite'] == True:
            pass
        elif kwargs['positive_definite'] == False:
            self.E[0] = 0.0

    def eval_QoI(self, mu, xi):
        x = mu + xi
        Vx = self.V.transpose().dot(x)
        g = np.zeros(self.systemsize)
        f = 0.0
        for i in range(0, self.systemsize):
            for j in range(0, self.systemsize):
                g[i] += self.V[i,j]*self.E[j]*Vx[j]
            f += 0.5*g[i]*x[i]

        return f

    def eval_QoIGradient(self, mu, xi):
        x = mu + xi
        Vx = self.V.transpose().dot(x)
        g = np.zeros(self.systemsize)
        for i in range(0, self.systemsize):
            for j in range(0, self.systemsize):
                g[i] += self.V[i,j]*self.E[j]*Vx[j]

        return g

"""
mat

[0.669963 0.227424 0.871246 0.910015 0.403449 0.491179 0.255712 0.510808 0.303675 0.747969;
 0.124521 0.0629285 0.375275 0.663709 0.895946 0.0326626 0.190279 0.613175 0.349628 0.342436;
 0.232099 0.307988 0.200296 0.857899 0.220059 0.850967 0.10221 0.635863 0.0287286 0.366659;
 0.147472 0.644024 0.713723 0.0570229 0.162096 0.684449 0.967385 0.14996 0.948119 0.679125;
 0.881156 0.189921 0.0847771 0.727253 0.497293 0.367176 0.234273 0.116565 0.0987549 0.110819;
 0.470129 0.422179 0.619725 0.940541 0.904302 0.370071 0.34143 0.156813 0.909265 0.126349;
 0.139064 0.508614 0.900505 0.469566 0.349258 0.922739 0.343383 0.125692 0.288524 0.358557;
 0.350387 0.738279 0.999603 0.327494 0.540892 0.916665 0.664266 0.466364 0.0855699 0.744292;
 0.477974 0.00499078 0.482415 0.951108 0.732742 0.497042 0.949754 0.410182 0.940132 0.942812;
 0.210511 0.433537 0.829527 0.136132 0.975876 0.960629 0.618523 0.832769 0.563914 0.241556]

E

[0.0561565,0.180111,0.187385,0.348511,0.474983,0.674096,0.787899,0.860523,0.916701,0.948724]
"""
