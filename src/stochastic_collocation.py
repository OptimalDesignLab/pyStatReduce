# StochasticCollocation.py
import numpy as np
import chaospy as cp
import utils

class StochasticCollocation(object):
    """
    Base class for uncertainty propagation using stochastic collocation. The
    user must create an object of this class in order to propagate uncertainty
    using stochastic collocation. Because of its close dependence on the package
    `chaospy`, only two types of distribution (univariate and multivariate) are
    intended to be supported, viz, normal and uniform. An object of this class
    can be initialied using the folloqing arguments.

    NOTE: As of 23 January 2018, only stochastic collocation for normal
    distribution is supported. In addition the collocation method uses a simple
    tensor product collocation grid with the same number of collocation point in
    every dimension.

    **Class Members**

    * `q` : Sample points for a quadrature scheme
    * `w` : Sample weights for a quadrature scheme

    **Constructor Arguments**

    * `degree` : Number of collocation point in a particular dimension
    * `distribution_type` : String value that indicates the type of distribution
                            is expected to be used.
    * `QoI_dimensions` : Dimension of the quantity of interest. Default = 1

    """

    def __init__(self, degree, distribution_type, QoI_dimensions=1):
        # TODO: Check the term degree
        assert degree > 0, "Need at least 1 collocation point for \
                                        uncertainty propagation"
        # TODO: get rid of normal and uniform
        if distribution_type == "Normal" or distribution_type == "MvNormal":
            self.normal = NormalDistribution(degree, QoI_dimensions)
        elif distribution_type == "Uniform":
            self.uniform = UniformDistribution(degree, QoI_dimensions)
        else:
            raise NotImplementedError

# ------------------------------------------------------------------------------

class NormalDistribution(StochasticCollocation):

    def __init__(self, degree, QoI_dimensions):
        self.q, self.w = np.polynomial.hermite.hermgauss(degree)
        self.QoI_dimensions = QoI_dimensions

    def mean(self, x, sigma, QoI_func):
        """
        Public member of `StochasticCollocation` class that should be used for
        performing stochastic collocation for univariate and multivariate
        normal distributions.

        **Inputs**

        * `x` : Mean value of the random variables
        * `sigma` : Standard deviation of the random variables
        * `QoI` : Quantity of Interest Object

        **Output**

        * `mu_j` : Mean value of the Quantity of Interest for a given mean value
                   of random variabes

        """
        systemsize = x.size
        ref_collocation_pts = self.q
        ref_collocation_w = self.w
        idx = 0
        colloc_xi_arr = np.zeros(systemsize)
        colloc_w_arr = np.zeros(systemsize)
        mu_j = np.zeros(self.QoI_dimensions)
        idx = self.doNormalMean(x, sigma, mu_j, ref_collocation_pts,
                                ref_collocation_w, QoI_func, colloc_xi_arr,
                                colloc_w_arr, idx)
        assert idx == -1
        mu_j[:] = mu_j[:]/(np.sqrt(np.pi)**systemsize)

        if len(mu_j) == 1:
            return mu_j[0]
        else:
            return mu_j

    def variance(self, QoI_func, jdist, mu_j):
        """
        Public member that is used to perform stochastic collocation only along
        certain directions to compute the variance of a QoI.

        **Inputs**

        * `QoI` : Quantity of Interest Object
        * `jdist` : joint distribution object of the random variables
        * `mu_j` : Mean value of the quantity of interest

        **Outputs**

        * `variance_j` : variance of the Quantity of Interest for a given mean
                         value of random variabes
        """

        # np.testing.assert_equal(len(mu_j), self.QoI_dimensions) # sanity check
        x = cp.E(jdist)
        rv_covariance = cp.Cov(jdist)
        systemsize = x.size
        ref_collocation_pts = self.q
        ref_collocation_w = self.w
        idx = 0
        colloc_xi_arr = np.zeros(systemsize)
        colloc_w_arr = np.zeros(systemsize)
        variance_j = np.zeros(self.QoI_dimensions)

        idx = self.doVariance(x, rv_covariance, mu_j, variance_j, ref_collocation_pts,
                              ref_collocation_w, QoI_func, colloc_xi_arr,
                              colloc_w_arr, idx)

        assert idx == -1
        variance_j[:] = variance_j[:]/(np.sqrt(np.pi)**systemsize)

        if len(variance_j) == 0:
            return variance_j[0]
        else:
            return variance_j


    def reduced_mean(self, QoI_func, jdist, dominant_space):
        """
        Public member that is used to perform stochastic collocation only along
        certain directions to compute the mean value of a QoI.

        **Inputs**

        * `QoI` : Quantity of Interest Object
        * `jdist` : joint distribution object of the random variables
        * `dominant_space` : Array of vectors along which the stochastic
                             collocation is performed.

        **Outputs**

        * `mu_j` : Mean value of the Quantity of Interest for a given mean value
                   of random variabes
        """
        x = cp.E(jdist)
        covariance = cp.Cov(jdist)
        n_quadrature_loops = len(dominant_space.dominant_indices)
        dominant_dir = dominant_space.iso_eigenvecs[:, dominant_space.dominant_indices]
        ref_collocation_pts = self.q
        ref_collocation_w = self.w
        idx = 0
        colloc_xi_arr = np.zeros(n_quadrature_loops)
        colloc_w_arr = np.zeros(n_quadrature_loops)
        mu_j = np.zeros(self.QoI_dimensions)

        idx = self.doReducedNormalMean(x, covariance, dominant_dir, mu_j,
                                       ref_collocation_pts, ref_collocation_w,
                                       QoI_func, colloc_xi_arr,
                                       colloc_w_arr, idx)

        assert idx == -1
        mu_j[:] = mu_j[:]/(np.sqrt(np.pi)**n_quadrature_loops)

        if len(mu_j) == 1:
            return mu_j[0]
        else:
            return mu_j

    def reduced_variance(self, QoI_func, jdist, dominant_space, mu_j):
        x = cp.E(jdist)
        rv_covariance = cp.Cov(jdist)
        n_quadrature_loops = len(dominant_space.dominant_indices)
        dominant_dir = dominant_space.iso_eigenvecs[:, dominant_space.dominant_indices]
        ref_collocation_pts = self.q
        ref_collocation_w = self.w
        idx = 0
        colloc_xi_arr = np.zeros(n_quadrature_loops)
        colloc_w_arr = np.zeros(n_quadrature_loops)
        variance_j = np.zeros(self.QoI_dimensions)

        idx = self.doReducedNormalVariance(x, rv_covariance, dominant_dir, mu_j, variance_j,
                                       ref_collocation_pts, ref_collocation_w,
                                       QoI_func, colloc_xi_arr,
                                       colloc_w_arr, idx)

        assert idx == -1
        variance_j[:] = variance_j[:]/(np.sqrt(np.pi)**n_quadrature_loops)

        if len(variance_j) == 1:
            return variance_j[0]
        else:
            return variance_j

    def doNormalMean(self, x, sigma, mu_j, xi, w, QoI_func, colloc_xi_arr,
                            colloc_w_arr, idx):
        """
        Inner private function that actually computes the mean value. This is a
        recursive function and should not be used by the end user. This function
        is called by `StochasticCollocation.normal.mean`.

        **Inputs**

        * `x` : Mean value of the random variable
        * `sigma` : Standard deviation of the random variables
        * `mu_j` : 1 element array into which the mean value of the QoI is assimilated.
        * `xi` : 1D reference collocation points along a direction obtained from numpy
        * `w` : 1D reference collocation weights along a direction obtained from numpy
        * `QoI` : QuantityOfInterest object
        * `colloc_xi_arr` : Actual multidimensional perturbation where the QoI
                            is to be evaluated
        * `colloc_w_arr` : corresponding weight array to `colloc_xi_arr`
        * `idx` : index variable that is necessay for book-keeping and
                  traversing the tensor product grid.

        **Outputs**

        * `idx-1` : the previous index

        """

        if idx == x.size-1:
            sqrt2 = np.sqrt(2)
            for i in xrange(0, xi.size):
                colloc_w_arr[idx] = w[i] # Get the array of all the weights needed
                colloc_xi_arr[idx] = xi[i] # Get the array of all the locations needed
                # fval = QoI.eval_QoI(x, sqrt2*sigma*colloc_xi_arr)
                fval = QoI_func(x, sqrt2*sigma*colloc_xi_arr)
                mu_j[:] += np.prod(colloc_w_arr)*fval
            return idx-1
        else:
            for i in xrange(0, xi.size):
                colloc_xi_arr[idx] = xi[i]
                colloc_w_arr[idx] = w[i]
                idx = self.doNormalMean(x, sigma, mu_j, xi, w, QoI_func, colloc_xi_arr,
                                    colloc_w_arr, idx+1)
            return idx-1

    def doVariance(self, x, rv_covariance, mu_j, variance_j, xi, w, QoI_func, colloc_xi_arr,
                   colloc_w_arr, idx):
        """
        Inner private function that actually computes the variance. This is a
        recursive function and should not be used by the end user. This function
        is called by `StochasticCollocation.normal.variance`.

        **Inputs**

        * `x` : Mean value of the random variable
        * `rv_covariance` : Standard deviation of the random variables
        * `mu_j` : Mean value of the QoI
        * `variance_j` : 1 element array into which the variance of the QoI is assimilated.
        * `xi` : 1D reference collocation points along a direction obtained from numpy
        * `w` : 1D reference collocation weights along a direction obtained from numpy
        * `QoI` : QuantityOfInterest object
        * `colloc_xi_arr` : Actual multidimensional perturbation where the QoI
                            is to be evaluated
        * `colloc_w_arr` : corresponding weight array to `colloc_xi_arr`
        * `idx` : index variable that is necessay for book-keeping and
                  traversing the tensor product grid.

        **Outputs**

        * `idx-1` : the previous index

        """

        if idx == x.size-1:
            sqrt2 = np.sqrt(2)
            sqrt_rv_covariance = np.sqrt(rv_covariance)
            for i in xrange(0, xi.size):
                colloc_w_arr[idx] = w[i] # Get the array of all the weights needed
                colloc_xi_arr[idx] = xi[i] # Get the array of all the locations needed
                fval = QoI_func(x, sqrt2*sqrt_rv_covariance.dot(colloc_xi_arr)) - mu_j
                variance_j[:] += np.prod(colloc_w_arr)*fval*fval
            return idx-1
        else:
            for i in xrange(0, xi.size):
                colloc_xi_arr[idx] = xi[i]
                colloc_w_arr[idx] = w[i]
                idx = self.doVariance(x, rv_covariance, mu_j, variance_j, xi, w, QoI_func, colloc_xi_arr,
                                    colloc_w_arr, idx+1)
            return idx-1

    def doReducedNormalMean(self, x, covariance, dominant_dir, mu_j, xi, w,
                            QoI_func, colloc_xi_arr, colloc_w_arr, idx):
        """
        Inner private function that actually performs the meat of the collocation
        computaion. `StochasticCollocation.normal.reduced_mean` calls this function.
        The user should not use this function directly.

        **Inputs**

        * `x` : Mean value of the random variable
        * `covariance` : Variance - Covariance matrix of the joint distribution
        * `dominant_dir` : Array of vectors along which the stochastic
                           collocation is performed.
        * `mu_j` : 1 element array into which the mean value of the QoI is assimilated.
        * `xi` : 1D reference collocation points along a direction obtained from numpy
        * `w` : 1D reference collocation weights along a direction obtained from numpy
        * `QoI` : QuantityOfInterest object
        * `colloc_xi_arr` : Actual multidimensional perturbation where the QoI
                            is to be evaluated
        * `colloc_w_arr` : corresponding weight array to `colloc_xi_arr`
        * `idx` : index variable that is necessay for book-keeping and
                  traversing the tensor product grid.

        **Outputs**

        * `idx-1` : the previous index

        """

        if idx == np.size(dominant_dir,1)-1:  # x.size-1: # TODO: Change to nquadrature loops
            sqrt2 = np.sqrt(2)
            sqrt_Sigma = np.sqrt(covariance)
            for i in xrange(0, xi.size):
                colloc_w_arr[idx] = w[i] # Get the array of all the weights needed
                colloc_xi_arr[idx] = xi[i] # Get the array of all the locations needed
                q = sqrt2* np.matmul(sqrt_Sigma, dominant_dir.dot(colloc_xi_arr))
                fval = QoI_func(x, q)
                mu_j[:] += np.prod(colloc_w_arr)*fval
            return idx-1
        else:
            for i in xrange(0, xi.size):
                colloc_xi_arr[idx] = xi[i]
                colloc_w_arr[idx] = w[i]
                idx = self.doReducedNormalMean(x, covariance,
                      dominant_dir, mu_j, xi, w, QoI_func, colloc_xi_arr,
                      colloc_w_arr, idx+1)
            return idx-1

    def doReducedNormalVariance(self, x, rv_covariance, dominant_dir, mu_j, variance_j,
                                xi, w, QoI_func, colloc_xi_arr, colloc_w_arr, idx):

        """
        Variance computtation using only the dominant directions.
        """
        if idx == np.size(dominant_dir,1)-1:
            sqrt2 = np.sqrt(2)
            sqrt_Sigma = np.sqrt(rv_covariance)
            for i in xrange(0, xi.size):
                colloc_w_arr[idx] = w[i] # Get the array of all the weights needed
                colloc_xi_arr[idx] = xi[i] # Get the array of all the locations needed
                q = sqrt2* np.matmul(sqrt_Sigma, dominant_dir.dot(colloc_xi_arr))
                fval = (QoI_func(x, q) - mu_j)**2
                variance_j[:] += np.prod(colloc_w_arr)*fval
            return idx-1
        else:
            for i in xrange(0, xi.size):
                colloc_xi_arr[idx] = xi[i]
                colloc_w_arr[idx] = w[i]
                idx = self.doReducedNormalVariance(x, rv_covariance,
                      dominant_dir, mu_j, variance_j, xi, w, QoI_func, colloc_xi_arr,
                      colloc_w_arr, idx+1)
            return idx-1

# ------------------------------------------------------------------------------

class UniformDistribution(StochasticCollocation):

    def __init__(self, degree, QoI_dimensions):
        self.q, self.w = np.polynomial.legendre.leggauss(degree)
        self.QoI_dimensions = QoI_dimensions

    def mean(self, QoI_func, jdist):
        x = cp.E(jdist)
        covariance = cp.Cov(jdist)

        systemsize = x.size
        ref_collocation_pts = self.q
        ref_collocation_w = self.w
        idx = 0
        colloc_xi_arr = np.zeros(systemsize)
        colloc_w_arr = np.zeros(systemsize)
        mu_j = np.zeros(self.QoI_dimensions)
        idx = self.doUniformMean(x, covariance, mu_j, ref_collocation_pts,
                                       ref_collocation_w, QoI_func, colloc_xi_arr,
                                       colloc_w_arr, idx)
        assert idx == -1
        mu_j[:] = mu_j[:]*(0.5**systemsize)

        if len(mu_j) == 1:
            return mu_j[0]
        else:
            return mu_j

    def reduced_mean(self, QoI_func, jdist, dominant_space):

        x = cp.E(jdist)
        covariance = cp.Cov(jdist)
        n_quadrature_loops = len(dominant_space.dominant_indices)
        dominant_dir = dominant_space.iso_eigen_vectors[:, dominant_space.dominant_indices]
        ref_collocation_pts = self.q
        ref_collocation_w = self.w
        idx = 0
        colloc_xi_arr = np.zeros(n_quadrature_loops)
        colloc_w_arr = np.zeros(n_quadrature_loops)
        mu_j = np.zeros(self.QoI_dimensions)

        idx = self.doReducedUniformMean(x, covariance, dominant_dir, mu_j,
                                        ref_collocation_pts,
                                        ref_collocation_w, QoI_func, colloc_xi_arr,
                                        colloc_w_arr, idx)

        assert idx == -1
        # TODO: The following scaling doesnt look right, INVESTIGATE
        mu_j[:] = mu_j[:]*(0.5**n_quadrature_loops)

        if len(mu_j) == 1:
            return mu_j[0]
        else:
            return mu_j

    def variance(self, QoI_func, jdist, mu_j):
        x = cp.E(jdist)
        rv_covariance = cp.Cov(jdist)
        systemsize = x.size
        ref_collocation_pts = self.q
        ref_collocation_w = self.w
        idx = 0
        colloc_xi_arr = np.zeros(systemsize)
        colloc_w_arr = np.zeros(systemsize)
        variance_j = np.zeros(1)

        idx = self.doVariance(x, rv_covariance, mu_j, variance_j, ref_collocation_pts,
                              ref_collocation_w, QoI_func, colloc_xi_arr, colloc_w_arr,
                              idx)

        assert idx == -1
        variance_j[:] = variance_j[:]*(0.5**systemsize)

        if len(variance_j) == 1:
            return variance_j[0]
        else:
            return variance_j


    def doUniformMean(self, x, covariance, mu_j, xi, w,
                      QoI_func, colloc_xi_arr, colloc_w_arr, idx):

        if idx == x.size-1:
            sqrt3 = np.sqrt(3)
            sqrt_Sigma = np.sqrt(covariance)
            for i in xrange(0, xi.size):
                colloc_w_arr[idx] = w[i] # Get the array of all the weights needed
                colloc_xi_arr[idx] = xi[i] # Get the array of all the locations needed
                # fval = QoI.eval_QoI(x, sqrt3 * np.dot(sqrt_Sigma, colloc_xi_arr))
                fval = QoI_func(x, sqrt3 * np.dot(sqrt_Sigma, colloc_xi_arr))
                mu_j[:] += np.prod(colloc_w_arr)*fval
            return idx-1
        else:
            for i in xrange(0, xi.size):
                colloc_xi_arr[idx] = xi[i]
                colloc_w_arr[idx] = w[i]
                idx = self.doUniformMean(x, covariance, mu_j, xi, w, QoI_func, colloc_xi_arr,
                                        colloc_w_arr, idx+1)
            return idx-1

    def doReducedUniformMean(self, x, covariance, dominant_dir, mu_j, xi, w, QoI_func,
                                   colloc_xi_arr, colloc_w_arr, idx):

        if idx == np.size(dominant_dir,1)-1:  # x.size-1: # TODO: Change to nquadrature loops
            sqrt3 = np.sqrt(3)
            sqrt_Sigma = np.sqrt(covariance)
            for i in xrange(0, xi.size):
                colloc_w_arr[idx] = w[i] # Get the array of all the weights needed
                colloc_xi_arr[idx] = xi[i] # Get the array of all the locations needed
                q = sqrt3* np.matmul(sqrt_Sigma, dominant_dir.dot(colloc_xi_arr))
                # fval = QoI.eval_QoI(x, q)
                fval = QoI_func(x, q)
                mu_j[:] += np.prod(colloc_w_arr)*fval
            return idx-1
        else:
            for i in xrange(0, xi.size):
                colloc_xi_arr[idx] = xi[i]
                colloc_w_arr[idx] = w[i]
                idx = self.doReducedNormalMean(x, covariance,
                      dominant_dir, mu_j, xi, w, QoI_func, colloc_xi_arr,
                      colloc_w_arr, idx+1)
            return idx-1

    def doVariance(self, x, rv_covariance, mu_j, variance_j, xi, w, QoI_func, colloc_xi_arr,
                   colloc_w_arr, idx):
        if idx == x.size-1:
            sqrt3 = np.sqrt(3)
            sqrt_rv_covariance = np.sqrt(rv_covariance)
            for i in xrange(0, xi.size):
                colloc_w_arr[idx] = w[i] # Get the array of all the weights needed
                colloc_xi_arr[idx] = xi[i] # Get the array of all the locations needed
                # fval = QoI.eval_QoI(x, sqrt3*sqrt_rv_covariance.dot(colloc_xi_arr)) - mu_j
                fval = QoI_func(x, sqrt3*sqrt_rv_covariance.dot(colloc_xi_arr)) - mu_j
                variance_j[:] += np.prod(colloc_w_arr)*fval*fval
            return idx-1
        else:
            for i in xrange(0, xi.size):
                colloc_xi_arr[idx] = xi[i]
                colloc_w_arr[idx] = w[i]
                idx = self.doVariance(x, rv_covariance, mu_j, variance_j, xi, w, QoI_func, colloc_xi_arr,
                                    colloc_w_arr, idx+1)
            return idx-1
