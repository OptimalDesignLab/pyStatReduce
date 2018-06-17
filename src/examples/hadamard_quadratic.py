# Hadamard quadratic
import numpy as np
from quantity_of_interest import QuantityOfInterest

class HadamardQuadratic(QuantityOfInterest):

    def __init__(self, systemsize, eigen_decayrate):
        QuantityOfInterest.__init__(self, systemsize)
        self.eigen_decayrate = eigen_decayrate
        self.eigen_vals = np.zeros(systemsize)
        self.eigen_vectors = np.zeros((systemsize, systemsize))
        self.getSyntheticEigenValues()
        self.getSyntheticEigenVectors()

    def eval_QoI(self, mu, xi):

        random_variable = mu + xi
        xi_hat = np.zeros(self.systemsize)
        self.applyHadamard(random_variable, xi_hat)

        fval = xi_hat.dot((self.eigen_vals*xi_hat))
        return fval

    def eval_QoIGradient(self, mu, xi):
        rv = mu + xi
        xi_hat = np.zeros(self.systemsize)
        dfdrv = np.zeros(self.systemsize)
        self.applyHadamard(rv, xi_hat)
        xi_hat = 2*self.eigen_vals*xi_hat
        self.applyHadamard(xi_hat, dfdrv)

        return dfdrv

    def eval_QoIHessian(self, mu, xi):
        Hessian = 2*np.dot(self.eigen_vectors,
                         (self.eigen_vals*self.eigen_vectors.T).T)
        return Hessian

    def eval_analytical_QoI_mean(self, mu, covariance_mat):

        sysmat = np.dot(self.eigen_vectors,
                        (self.eigen_vals*self.eigen_vectors.T).T)
        mu_fval = np.trace(np.matmul(sysmat, covariance_mat))  \
                  + mu.dot(sysmat.dot(mu))

        return mu_fval

    def applyHadamard(self, x, y):
        """
        Multiplies `x` by a scaled orthonormal Hadamard matrix and returns `y`.
        This method uses Sylvester's construction and thus results in a
        symmetric Hadamrad matrix with trace zero.
        """

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
            y_2D[0,:] = fac*(x_2D[0,:] + x_2D[1,:])
            y_2D[1,:] = fac*(x_2D[0,:] - x_2D[1,:])
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
