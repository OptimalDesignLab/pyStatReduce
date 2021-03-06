# Test stochastic collocation module
import unittest
import numpy as np
import chaospy as cp

from pystatreduce.stochastic_collocation import StochasticCollocation
from pystatreduce.quantity_of_interest import QuantityOfInterest
from pystatreduce.dimension_reduction import DimensionReduction
import pystatreduce.examples as examples

class StochasticCollocationTest(unittest.TestCase):

    def test_normalStochasticCollocation2D(self):
        systemsize = 2
        x = np.random.rand(systemsize) # np.zeros(systemsize)
        theta = 0
        sigma = np.array([0.2, 0.1])
        tuple = (theta,)
        jdist = cp.MvNormal(x, np.diag(sigma))

        # Create a Stochastic collocation object
        collocation = StochasticCollocation(3, "Normal")

        # Create a QoI object using ellpsoid
        QoI = examples.Paraboloid2D(systemsize, tuple)

        # Compute the expected value
        QoI_func = QoI.eval_QoI
        mu_j = collocation.normal.mean(x, sigma, QoI_func)

        # Test against nested loops
        mu_j_hat = 0.0
        sqrt2 = np.sqrt(2)
        for i in range(0, collocation.normal.q.size):
            for j in range(0, collocation.normal.q.size):
                f_val = QoI.eval_QoI(x, sqrt2*sigma*[collocation.normal.q[i],
                        collocation.normal.q[j]])
                mu_j_hat += collocation.normal.w[i]*collocation.normal.w[j]*f_val

        mu_j_hat = mu_j_hat/(np.sqrt(np.pi)**systemsize)

        diff = abs(mu_j - mu_j_hat)
        self.assertTrue(diff < 1.e-15)

        # Test Check for the variance
        # - Analytical Value
        cov_mat = cp.Cov(jdist)
        mat1 = np.matmul(QoI.quadratic_matrix, cov_mat)
        mat2 = np.matmul(QoI.quadratic_matrix, np.matmul(cov_mat,
                         QoI.quadratic_matrix.dot(x)))
        variance_analytical = 2*np.trace(np.matmul(mat1, mat1)) + 4*x.dot(mat2)

        # - Numerical Value
        sigma_j = collocation.normal.variance(QoI_func, jdist, mu_j)
        diff = abs(variance_analytical - sigma_j)
        self.assertTrue(diff < 1.e-12)


    def test_normalStochasticCollocation3D(self):
        systemsize = 3
        x = np.zeros(systemsize)
        sigma = 0.2*np.ones(systemsize)

        # Create a Stochastic collocation object
        collocation = StochasticCollocation(3, "Normal")

        # Create a QoI object using ellpsoid
        QoI = examples.Paraboloid3D(systemsize)
        QoI_func = QoI.eval_QoI

        # Compute the expected value
        mu_j = collocation.normal.mean(x, sigma, QoI_func)

        # Test against nested loops
        mu_j_hat = 0.0
        sqrt2 = np.sqrt(2)
        for i in range(0, collocation.normal.q.size):
            for j in range(0, collocation.normal.q.size):
                for k in range(0, collocation.normal.q.size):
                    f_val = QoI.eval_QoI(x, sqrt2*sigma*[collocation.normal.q[i],
                            collocation.normal.q[j], collocation.normal.q[k]])
                    mu_j_hat += collocation.normal.w[i]*collocation.normal.w[j]* \
                                collocation.normal.w[k]*f_val

        mu_j_hat = mu_j_hat/(np.sqrt(np.pi)**systemsize)

        diff = abs(mu_j - mu_j_hat)
        self.assertTrue(diff < 1.e-15)

    def test_normalStochasticCollocation3D_grad(self):
        systemsize = 3
        x = np.random.rand(systemsize)
        sigma = np.random.rand(systemsize)

        # Create a Stochastic collocation object
        collocation = StochasticCollocation(4, "Normal", systemsize)

        # Create a QoI object using ellpsoid
        QoI = examples.Paraboloid3D(systemsize)
        QoI_func = QoI.eval_QoIGradient

        # Compute the expected value
        mu_j = collocation.normal.mean(x, sigma, QoI_func)

        # Test against nested loops
        mu_j_hat = np.zeros(systemsize) # 0.0
        sqrt2 = np.sqrt(2)
        for i in range(0, collocation.normal.q.size):
            for j in range(0, collocation.normal.q.size):
                for k in range(0, collocation.normal.q.size):
                    f_val = QoI.eval_QoIGradient(x, sqrt2*sigma*[collocation.normal.q[i],
                            collocation.normal.q[j], collocation.normal.q[k]])
                    mu_j_hat[:] += collocation.normal.w[i]*collocation.normal.w[j]* \
                                collocation.normal.w[k]*f_val

        mu_j_hat[:] = mu_j_hat/(np.sqrt(np.pi)**systemsize)

        diff = abs(mu_j - mu_j_hat)
        self.assertTrue((diff < 1.e-15).all())

        # Test against analytical value
        mu_j_analytical = QoI.eval_QoIGradient(x, np.zeros(systemsize))
        diff_analytical = abs(mu_j - mu_j_analytical)
        self.assertTrue((diff_analytical < 1.e-13).all())

    def test_normalStochasticCollocation5D(self):
        systemsize = 5
        x = np.random.rand(systemsize)
        sigma = 0.2*np.ones(systemsize)

        # Create a Stochastic collocation object
        collocation = StochasticCollocation(3, "Normal")

        # Create a QoI object using ellpsoid
        QoI = examples.Paraboloid5D(systemsize)
        QoI_func = QoI.eval_QoI

        # Compute the expected value
        mu_j = collocation.normal.mean(x, sigma, QoI_func)

        # Test against nested loops
        mu_j_hat = 0.0
        sqrt2 = np.sqrt(2)
        q = collocation.normal.q
        w = collocation.normal.w
        for i in range(0, q.size):
            for j in range(0, q.size):
                for k in range(0, q.size):
                    for l in range(0, q.size):
                        for m in range(0, q.size):
                            fval = QoI.eval_QoI(x, sqrt2*sigma*[q[i], q[j], q[k],
                                   q[l], q[m]])
                            mu_j_hat += w[i]*w[j]*w[k]*w[l]*w[m]*fval

        mu_j_hat = mu_j_hat/(np.sqrt(np.pi)**systemsize)

        diff = abs(mu_j - mu_j_hat)
        self.assertTrue(diff < 1.e-15)

    def test_gradient_wrt_parameters(self):
        """
        Test second and higher order gradients of statistical moments w.r.t
        independent parameters (i.e. not random variables)
        """
        systemsize = 2
        x = np.random.rand(systemsize)
        sigma = 0.2*np.ones(systemsize)
        jdist = cp.MvNormal(x, np.diag(sigma))

        # Create QoI object
        QoI = examples.PolyRVDV()
        dv = np.random.rand(QoI.n_parameters)
        QoI.set_dv(dv)

        # Create a Stochastic collocation object
        collocation = StochasticCollocation(3, "Normal")
        collocation_grad = StochasticCollocation(3, "Normal", QoI_dimensions=QoI.n_parameters)

        QoI_func = QoI.eval_QoI
        dQoI_func = QoI.eval_QoIGradient_dv

        # Check full integration
        #  - Compute the mean of the QoI and QoI gradient w.r.t dv
        mu_j = collocation.normal.mean(x, sigma, QoI_func)
        dmu_j = collocation_grad.normal.mean(x, sigma, dQoI_func)
        var_j = collocation.normal.variance(QoI_func, jdist, mu_j)
        std_dev_j = np.sqrt(var_j)

        # - Compute the corresponding variance and standard deviation
        dvariance_j = collocation_grad.normal.dVariance(QoI_func, jdist, mu_j, dQoI_func, dmu_j)
        dstd_dev_j = collocation_grad.normal.dStdDev(QoI_func, jdist, mu_j, var_j, dQoI_func, dmu_j)

        # - Check against finite difference
        pert = 1.e-7
        dvariance_j_fd = np.zeros(QoI.n_parameters)
        dstd_dev_j_fd = np.zeros(QoI.n_parameters)
        for i in range(0, QoI.n_parameters):
            dv[i] += pert
            QoI.set_dv(dv)
            mu_j_i = collocation.normal.mean(x, sigma, QoI_func)
            var_j_i = collocation.normal.variance(QoI_func, jdist, mu_j_i)
            std_dev_j_i = np.sqrt(var_j_i)
            dvariance_j_fd[i] = (var_j_i - var_j) / pert
            dstd_dev_j_fd[i] = (std_dev_j_i - std_dev_j) / pert
            dv[i] -= pert

        err = abs(dvariance_j - dvariance_j_fd)
        self.assertTrue((err < 1.e-4).all())
        err = abs(dstd_dev_j_fd - dstd_dev_j)
        self.assertTrue((err < 1.e-4).all())

        # Check reduced integration
        #  - Create dimension reduction object
        threshold_factor = 0.9
        dominant_space = DimensionReduction(threshold_factor=threshold_factor,
                                            exact_Hessian=True)
        dominant_space.getDominantDirections(QoI, jdist)
        #  - Compute the mean of the QoI and QoI gradient w.r.t dv
        mu_j = collocation.normal.reduced_mean(QoI_func, jdist, dominant_space)
        dmu_j = collocation_grad.normal.reduced_mean(QoI_func, jdist, dominant_space)
        var_j = collocation.normal.reduced_variance(QoI_func, jdist, dominant_space, mu_j)
        std_dev_j = np.sqrt(var_j)
        # - Compute the corresponding variance and standard deviation
        dvariance_j = collocation_grad.normal.reduced_dVariance(QoI_func, jdist,
                                        dominant_space, mu_j, dQoI_func, dmu_j)
        dstd_dev_j = collocation_grad.normal.dReducedStdDev(QoI_func, jdist,
                                  dominant_space, mu_j, var_j, dQoI_func, dmu_j)
        # - Check against finite difference
        dvariance_j_fd.fill(0.)
        dstd_dev_j_fd.fill(0.)
        for i in range(0, QoI.n_parameters):
            dv[i] += pert
            QoI.set_dv(dv)
            mu_j_i = collocation.normal.reduced_mean(QoI_func, jdist, dominant_space)
            var_j_i = collocation.normal.reduced_variance(QoI_func, jdist, dominant_space, mu_j_i)
            std_dev_j_i = np.sqrt(var_j_i)
            dvariance_j_fd[i] = (var_j_i - var_j) / pert
            dstd_dev_j_fd[i] = (std_dev_j_i - std_dev_j) / pert
            dv[i] -= pert

        err = abs(dvariance_j - dvariance_j_fd)
        self.assertTrue((err < 1.e-4).all())
        err = abs(dstd_dev_j_fd - dstd_dev_j)
        self.assertTrue((err < 1.e-4).all())

    ############################################################################
    # Uniform Distibution
    ############################################################################

    def test_uniformStochasticCollocationConstt(self):
        systemsize = 2
        mu = np.random.rand(systemsize)
        sigma = np.random.rand(systemsize)

        # Create a Stochastic collocation object
        collocation = StochasticCollocation(1, "Uniform")

        # Create a probability distribution object
        xlo = mu - np.sqrt(3)*sigma
        xhi = mu + np.sqrt(3)*sigma
        udist1 = cp.Uniform(xlo[0], xhi[0])
        udist2 = cp.Uniform(xlo[1], xhi[1])
        jdist = cp.J(udist1, udist2)

        # Create a QoI object using ellpsoid
        QoI = examples.ConstantFunction(systemsize)
        QoI_func = QoI.eval_QoI

        # Compute the expected value
        mu_j = collocation.uniform.mean(QoI_func, udist1)

        diff = abs(mu_j - QoI.eval_QoI(mu, np.zeros(systemsize)))
        self.assertTrue(diff < 1.e-15)

        # Test Check for the variance
        variance_j = collocation.uniform.variance(QoI_func, jdist, mu_j)
        self.assertAlmostEqual(variance_j, 0, places=15)


    def test_uniformStochasticCollocation5D(self):
        systemsize = 5
        mu = np.random.rand(systemsize)
        sigma = np.random.rand(systemsize)

        # Create a Stochastic collocation object
        collocation = StochasticCollocation(3, "Uniform")

        # Create a probability distribution object
        xlo = mu - np.sqrt(3)*sigma
        xhi = mu + np.sqrt(3)*sigma
        udist1 = cp.Uniform(xlo[0], xhi[0])
        udist2 = cp.Uniform(xlo[1], xhi[1])
        udist3 = cp.Uniform(xlo[2], xhi[2])
        udist4 = cp.Uniform(xlo[3], xhi[3])
        udist5 = cp.Uniform(xlo[4], xhi[4])
        jdist = cp.J(udist1, udist2, udist3, udist4, udist5)

        # Create a QoI object using ellpsoid
        QoI = examples.Paraboloid5D(systemsize)
        QoI_func = QoI.eval_QoI

        # Compute the expected value
        mu_j = collocation.uniform.mean(QoI_func, jdist)

        # Compute the analytical expected value
        mu_j_analytical = np.trace(np.matmul(QoI.quadratic_matrix, cp.Cov(jdist))) + \
                            QoI.eval_QoI(mu, np.zeros(systemsize))

        diff = abs(mu_j - mu_j_analytical)
        self.assertTrue(diff < 1.e-12)

if __name__ == "__main__":
    unittest.main()
