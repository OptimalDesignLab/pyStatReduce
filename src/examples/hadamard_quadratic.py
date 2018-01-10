# Hadamard quadratic
import numpy as np
from QuantityOfInterest import QuantityOfInterest

class HadamardQuadratic(QuantityOfInterest):

    def __init__(self, systemsize, eigen_decayrate):
        QuantityOfInterest.__init__(self, systemsize)
        self.eigen_decayrate = eigen_decayrate
        self.eigen_vals = np.zeros(systemsize)
        self.eigen_vectors = np.zeros((systemsize, systemsize))
        self.getSyntheticEigenValues()
        self.getSyntheticEigenVectors()

    def eval_QoI(self, mu, xi):

        # getSyntheticEigenValues()
        # getSyntheticEigenVectors()
        random_variable = mu + xi
        xi_hat = np.zeros(self.systemsize)
        self.applyHadamard(random_variable, xi_hat)

        fval = xi_hat.dot((self.eigen_vals*xi_hat))
        return fval

    def eval_QoIHessian(self, mu, xi):
        Hessian = np.dot(self.eigen_vectors,
                         (self.eigen_vals*self.eigen_vectors.T).T)
        return Hessian

    def eval_analytical_QoI_mean(self, mu, covariance_mat):

        sysmat = np.dot(self.eigen_vectors,
                        (self.eigen_vals*self.eigen_vectors.T).T)
        mu_fval = np.trace(np.matmul(sysmat, covariance_mat))  \
                  + mu.dot(sysmat.dot(mu))

        return mu_fval

    def applyHadamard(self, x, y):
        n = np.size(x,0)
        assert n % 2  == 0, "size of x must be a power of 2"

        # Convert to 2D because numpy complains
        if x.ndim == 1:
            x_2D = x[:, np.newaxis]
            y_2D = y[:, np.newaxis]
        else:
            x_2D = x
            y_2D = y

        fac = 1.0/np.sqrt(2)
        if n == 2:
            y[0,:] = fac*(x[0,:] + x[1,:])
            y[1,:] = fac*(x[0,:] - x[1,:])
        else:
            n2 = n // 2
            Hx1 = np.zeros((n2, np.size(x_2D,1)))
            Hx2 = np.zeros((n2, np.size(x_2D,1)))
            self.applyHadamard(x_2D[0:n2,:], Hx1)
            self.applyHadamard(x_2D[n2:n,:], Hx2)
            y_2D[0:n2,:] = fac*(Hx1 + Hx2)
            y_2D[n2:n,:] = fac*(Hx1 - Hx2)

    def getSyntheticEigenValues(self):
        for i in xrange (0, self.systemsize):
            self.eigen_vals[i] = 1/(i+1)**self.eigen_decayrate

    def getSyntheticEigenVectors(self):
        iden = np.eye(self.systemsize)
        self.applyHadamard(iden, self.eigen_vectors)