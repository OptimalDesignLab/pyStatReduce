# test_arnoldi_sampling.py
import unittest
import numpy as np
import chaospy as cp

from pystatreduce.stochastic_collocation import StochasticCollocation
from pystatreduce.quantity_of_interest import QuantityOfInterest
from pystatreduce.stochastic_arnoldi.arnoldi_sample import ArnoldiSampling
import pystatreduce.examples as examples

class ArnoldiSamplingTest(unittest.TestCase):

    def test_modified_GramSchmidt_fullRank(self):

        # Generate a random set of vectors and use modGramSchmidt to
        # orthogonalize them

        systemsize = 10

        # Initialize ArnoldiSampling object
        alpha = 1.0
        num_sample = 4
        arnoldi = ArnoldiSampling(alpha, num_sample)

        # Create arrays for modified_GramSchmidt
        Z = np.random.rand(systemsize, num_sample)
        H = np.zeros([num_sample, num_sample-1])

        # Populate Z
        for i in range(-1, num_sample-1):
            arnoldi.modified_GramSchmidt(i, H, Z)
            # Check that the vectors are unit normal
            self.assertAlmostEqual(np.linalg.norm(Z[:,i+1]), 1, places=14)

        # Check that the vectors are orthogonal
        for i in range(0, num_sample):
            for j in range(i+1, num_sample):
                self.assertAlmostEqual(np.dot(Z[:,i], Z[:,j]), 0, places=14)

    def test_modified_GramSchmidt_RankDeficient(self):

        # Generate a random set of vectors, make one of them a linear combination
        # of the others, and use modGramSchmidt to orthogonalize them

        systemsize = 10

        # Initialize ArnoldiSampling object
        alpha = 1.0
        num_sample = 4
        arnoldi = ArnoldiSampling(alpha, num_sample)

        # Create arrays for modified_GramSchmidt
        Z = np.random.rand(systemsize, num_sample)
        Z[:,num_sample-1] = Z[:,0:num_sample-1].dot(np.random.rand(num_sample-1))
        H = np.zeros([num_sample, num_sample-1])

        for i in range(-1, num_sample-2):
            arnoldi.modified_GramSchmidt(i, H, Z)
            # Check that the vectors are unit normal
            self.assertAlmostEqual(np.linalg.norm(Z[:,i+1]), 1, places=14)

        # calling now should produce lin_depend flag
        lin_depend = arnoldi.modified_GramSchmidt(num_sample-2, H, Z)
        self.assertTrue(lin_depend)

    def test_arnoldiSample_complete(self):

        # Compute all of the eigenmodes of an isoprobabilistic Hadamard
        # quadratic system using Arnoldi sampling and verify against the exact
        # computation

        systemsize = 16
        eigen_decayrate = 2.0

        # Create Hadmard Quadratic object
        QoI = examples.HadamardQuadratic(systemsize, eigen_decayrate)

        # Initialize chaospy distribution
        std_dev = np.random.rand(QoI.systemsize)
        sqrt_Sigma = np.diag(std_dev)
        mu = np.random.rand(QoI.systemsize)
        jdist = cp.MvNormal(mu, sqrt_Sigma)

        # Estimate the eigenmodes of the Hessenberg matrix using Arnoldi
        perturbation_size = 1.e-6
        num_sample = QoI.systemsize+1
        arnoldi = ArnoldiSampling(perturbation_size, num_sample)
        eigenvals = np.zeros(num_sample-1)
        eigenvecs = np.zeros([QoI.systemsize, num_sample-1])
        mu_iso = np.zeros(QoI.systemsize)
        mu_iso[:] = 0.0
        gdata0 = np.zeros([QoI.systemsize, num_sample])
        gdata0[:,0] = np.dot(QoI.eval_QoIGradient(mu, np.zeros(QoI.systemsize)),
                            sqrt_Sigma)
        dim, error_estimate = arnoldi.arnoldiSample(QoI, jdist, mu_iso, gdata0,
                                                    eigenvals, eigenvecs)

        # Compute the exact eigenmodes
        Hessian = QoI.eval_QoIHessian(mu, np.zeros(QoI.systemsize))
        Hessian_Product = np.matmul(sqrt_Sigma, np.matmul(Hessian, sqrt_Sigma))
        exact_eigenvals, exact_eigenvecs = np.linalg.eig(Hessian_Product)

        # Sort in descending order
        sort_ind1 = exact_eigenvals.argsort()[::-1]
        sort_ind2 = eigenvals.argsort()[::-1]

        # Compare the eigenvalues
        err_eigenvals = abs(exact_eigenvals[sort_ind1] -
                            eigenvals[sort_ind2])
        self.assertTrue((err_eigenvals < 1.e-7).all())

        # Compare the eigenvectors
        V_exact = exact_eigenvecs[:, sort_ind1]
        V_arnoldi = eigenvecs[:, sort_ind2]
        product = np.zeros(QoI.systemsize)
        for i in range(0, systemsize):
            product[i] = abs(np.dot(V_exact[:,i], V_arnoldi[:,i]))
            self.assertAlmostEqual(product[i], 1.0, places=7)


    def test_dimensionReduction_arnoldi_partial(self):

        # Compute 21 major the eigenmodes of an isoprobabilistic Hadamard
        # quadratic system using Arnoldi sampling and verify against the exact
        # computation

        systemsize = 64
        eigen_decayrate = 2.0
        num_sample = 51

        # Create Hadmard Quadratic object
        QoI = examples.HadamardQuadratic(systemsize, eigen_decayrate)

        # Initialize chaospy distribution
        std_dev = np.random.rand(QoI.systemsize)
        sqrt_Sigma = np.diag(std_dev)
        mu = np.random.rand(QoI.systemsize)
        jdist = cp.MvNormal(mu, sqrt_Sigma)

        # Estimate the eigenmodes of the Hessenberg matrix using Arnoldi
        perturbation_size = 1.e-6
        arnoldi = ArnoldiSampling(perturbation_size, num_sample)
        eigenvals = np.zeros(num_sample-1)
        eigenvecs = np.zeros([QoI.systemsize, num_sample-1])
        mu_iso = np.zeros(QoI.systemsize)
        mu_iso[:] = 0.0
        gdata0 = np.zeros([QoI.systemsize, num_sample])
        gdata0[:,0] = np.dot(QoI.eval_QoIGradient(mu, np.zeros(QoI.systemsize)),
                            sqrt_Sigma)
        dim, error_estimate = arnoldi.arnoldiSample(QoI, jdist, mu_iso, gdata0,
                                                    eigenvals, eigenvecs)

        # Compute the exact eigenmodes
        Hessian = QoI.eval_QoIHessian(mu, np.zeros(QoI.systemsize))
        Hessian_Product = np.matmul(sqrt_Sigma, np.matmul(Hessian, sqrt_Sigma))
        exact_eigenvals, exact_eigenvecs = np.linalg.eig(Hessian_Product)

        # Sort in descending order
        sort_ind1 = exact_eigenvals.argsort()[::-1]
        sort_ind2 = eigenvals.argsort()[::-1]
        lambda_exact = exact_eigenvals[sort_ind1]
        lambda_arnoldi = eigenvals[sort_ind2]

        # Compare the eigenvalues
        for i in range(0, num_sample-1):
            if i < 10:
                self.assertAlmostEqual(lambda_arnoldi[i], lambda_exact[i], places=6)
            else:
                # print "lambda_exact[i] = ", lambda_exact[i], "lambda_arnoldi[i] = ", lambda_arnoldi[i]
                self.assertAlmostEqual(lambda_arnoldi[i], lambda_exact[i], places=1)

        # Compare the eigenvectors
        V_exact = exact_eigenvecs[:, sort_ind1]
        V_arnoldi = eigenvecs[:, sort_ind2]
        for i in range(0, num_sample-1):
            product = abs(np.dot(V_exact[:,i], V_arnoldi[:,i]))
            if i < 10:
                self.assertAlmostEqual(product, 1.0, places=6)
            # else:
            #     self.assertAlmostEqual(product, 1.0, places=0)


    def test_arnoldiSample_og_positiveDefinite(self):

        # Use a synthetic quadratic function, and check that arnoldiSample recovers
        # its eigenvalues and eigenvectors

        systemsize = 10

        # Generate QuantityOfInterest
        QoI = examples.RandomQuadratic(systemsize, positive_definite=True)

        # generate data at initial point (1,1,1,...,1)^T
        xdata = np.zeros([systemsize, systemsize+1])
        fdata = np.zeros(systemsize+1)
        gdata = np.zeros([systemsize, systemsize+1])
        xdata[:,0] = np.ones(systemsize)
        fdata[0] = QoI.eval_QoI(xdata[:,0], np.zeros(systemsize))
        gdata[:,0] = QoI.eval_QoIGradient(xdata[:,0], np.zeros(systemsize))

        # Initialize ArnoldiSampling object
        alpha = 1.0
        num_sample = systemsize+1
        arnoldi = ArnoldiSampling(alpha, num_sample)

        # Generate sample
        eigenvals = np.zeros(systemsize)
        eigenvecs = np.zeros([systemsize, systemsize])
        grad_red = np.zeros(systemsize)
        arnoldi.arnoldiSample_og(QoI, xdata, fdata, gdata, eigenvals, eigenvecs,
                              grad_red)

        # Check that eigenvalues and eigenvectors agree
        for i in range(0, systemsize):
            self.assertAlmostEqual(eigenvals[i], QoI.E[i], places=7)
            self.assertAlmostEqual(abs(np.dot(eigenvecs[:,i], QoI.V[:,i])), 1.0, places=7)

    def test_arnoldiSample_og_positiveSemiDefinite(self):

        # Use a synthetic quadratic function, and check that arnoldiSample recovers
        # its eigenvalues and eigenvectors

        systemsize = 10

        # Generate QuantityOfInterest
        QoI = examples.RandomQuadratic(systemsize, positive_definite=False)

        # generate data at initial point (1,1,1,...,1)^T
        xdata = np.zeros([systemsize, systemsize+1])
        fdata = np.zeros(systemsize+1)
        gdata = np.zeros([systemsize, systemsize+1])
        xdata[:,0] = np.ones(systemsize)
        fdata[0] = QoI.eval_QoI(xdata[:,0], np.zeros(systemsize))
        gdata[:,0] = QoI.eval_QoIGradient(xdata[:,0], np.zeros(systemsize))

        # Initialize ArnoldiSampling object
        alpha = 1.0
        num_sample = systemsize+1
        arnoldi = ArnoldiSampling(alpha, num_sample)

        # Generate sample
        eigenvals = np.zeros(systemsize)
        eigenvecs = np.zeros([systemsize, systemsize])
        grad_red = np.zeros(systemsize)
        dim, error_estimate = arnoldi.arnoldiSample_og(QoI, xdata, fdata, gdata,
                                                    eigenvals, eigenvecs,
                                                    grad_red)

        self.assertAlmostEqual(error_estimate, 0.0, places=6)
        self.assertEqual(dim, systemsize-1)

        # Check that eigenvalues and eigenvectors agree
        for i in range(0, dim):
            self.assertAlmostEqual(eigenvals[i], QoI.E[i+1], places=7)
            self.assertAlmostEqual(abs(np.dot(eigenvecs[:,i], QoI.V[:,i+1])), 1.0, places=7)

    def test_arnoldiSample_og_reducedGradient(self):

        # use a nonlinear function and a small perturbation to test the reduced
        # gradient produced by arnoldiSample

        systemsize = 10

        # Generate QuantityOfInterest
        QoI = examples.ExponentialFunction(systemsize)

        # generate data at initial point (1,1,1,...,1)^T
        xdata = np.zeros([systemsize, systemsize+1])
        fdata = np.zeros(systemsize+1)
        gdata = np.zeros([systemsize, systemsize+1])
        xdata[:,0] = np.ones(systemsize)
        fdata[0] = QoI.eval_QoI(xdata[:,0], np.zeros(systemsize))
        gdata[:,0] = QoI.eval_QoIGradient(xdata[:,0], np.zeros(systemsize))

        # Initialize ArnoldiSampling object
        alpha = np.sqrt(np.finfo(float).eps)
        num_sample = systemsize+1
        arnoldi = ArnoldiSampling(alpha, num_sample)

        # Generate sample
        eigenvals = np.zeros(systemsize)
        eigenvecs = np.zeros([systemsize, systemsize])
        grad_red = np.zeros(systemsize)
        dim, error_estimate = arnoldi.arnoldiSample_og(QoI, xdata, fdata, gdata,
                                                    eigenvals, eigenvecs,
                                                    grad_red)

        # check reduced gradient;
        # rotate back out of eigenvector coordinates
        g = eigenvecs.dot(grad_red)
        err_g = abs(g - gdata[:,0])
        self.assertTrue((err_g < 1.e-6).all())


if __name__ == "__main__":
    unittest.main()
