# Arnoldi_sample
import numpy as np
import chaospy as cp

np.set_printoptions(linewidth=150, suppress=True)

class ArnoldiSampling(object):

    def __init__(self, alpha, num_sample):
        assert alpha > 0
        self.alpha = alpha # Perturbation size
        self.num_sample = num_sample

    def arnoldiSample(self, QoI, jdist, xdata0, gdata0, eigenvals, eigenvecs):
        n = QoI.systemsize
        m = self.num_sample - 1

        # Assertions
        assert gdata0.shape[0] == eigenvecs.shape[0] == n
        assert eigenvals.shape[0] == m
        # assert gdata.shape[1] == self.num_sample

        # Initialize the basis-vector array and Hessenberg matrix
        Z = np.zeros([n, m+1])
        H = np.zeros([m+1, m])
        Z[:,0] = -gdata0[:,0]/np.linalg.norm(gdata0[:,0])
        linear_dependence = False

        rv_mean = cp.E(jdist)
        covariance = cp.Cov(jdist)
        xdata_i = np.zeros(QoI.systemsize)
        # gdata_i = np.zeros(QoI.systemsize)

        # Check if variance covariance matrix is diagonal
        if np.count_nonzero(covariance - np.diag(np.diagonal(covariance))) == 0:
            sqrt_Sigma = np.sqrt(covariance)
        else:
            raise NotImplementedError

        for i in xrange(0, m):
            # Find new sample point and data; Compute function and gradient values
            xdata_i[:] = xdata0 + self.alpha * Z[:,i]

            # Convert the new sample point into the original space
            x_val = np.dot(sqrt_Sigma, xdata_i) + rv_mean
            # gdata_i = np.dot(QoI.eval_QoIGradient(rv_mean, x_val - rv_mean),
            #                 sqrt_Sigma)
            gdata0[:,i+1] = np.dot(QoI.eval_QoIGradient(rv_mean, x_val - rv_mean),
                            sqrt_Sigma)

            # Find the new basis vector and orthogonalize it against the old ones
            Z[:,i+1] = (gdata0[:,i+1] - gdata0[:,0])/self.alpha
            linear_dependence = self.modified_GramSchmidt(i, H, Z)
            if linear_dependence == True:
                # new basis vector is linealy dependent, so terminate early
                break

        if (i != n-1 and linear_dependence == True):
            i -= 1
        elif (i == n-1 and linear_dependence == True):
            pass

        # Symmetrize the Hessenberg matrix, and find its eigendecomposition
        Hsym = 0.5*(H[0:i+1, 0:i+1] + H[0:i+1,0:i+1].transpose())
        # print "H = ", '\n', H
        # print "Hsym = ", '\n', Hsym
        eigenvals_red, eigenvecs_red = np.linalg.eig(Hsym)

        # Sort the reduced eigenvalues and eigenvectors reduced in ascending order
        idx = np.argsort(eigenvals_red)
        eigenvecs_red = eigenvecs_red[:,idx]
        eigenvals_red = eigenvals_red[idx]

        # Populate the system eigenvalue and eigenvectors
        eigenvals[:] = 0.0
        eigenvals[0:i+1] = eigenvals_red
        # error_estimate = np.linalg.norm(0.5*(H[0:i+1,0:i+1] - H[0:i+1,0:i+1].transpose()))

        # Generate the full-space eigenvector approximations and get the error
        # estimate for the Rayleigh-Ritz eigen pairs
        e_m = np.zeros(i+1)
        e_m[-1] = 1.0 # Last value of the basis = 1
        error_estimate = np.zeros(i+1)
        for k in xrange(0, i+1):
            eigenvecs[:,k] = Z[:,0:i+1].dot(eigenvecs_red[0:i+1, k])
            error_estimate[k] = H[i+1,i]*abs(np.dot(e_m,eigenvecs_red[:,k]))

        # Finally, sort the system eigen pairs in descending order
        idx = np.argsort(np.abs(eigenvals))[::-1]
        eigenvecs[:,:] = eigenvecs[:,idx]
        eigenvals[:] = eigenvals[idx]
        error_estimate = error_estimate[idx]

        return i+1, error_estimate

    def modified_GramSchmidt(self, i, Hsbg, w):
        assert Hsbg.shape[1] >= i+1
        assert Hsbg.shape[0] >= i+2
        assert w.shape[1] >= i+2

        err_msg = "modified_GramSchmidt failed: "
        reorth = 0.98 # CONSTANT!

        # get the norm of the vector being orthogonalized, and find the
        # threshold for re-orthogonalization
        nrm = np.linalg.norm(w[:,i+1]) # w[:,i+1].dot(w[:,i+1])
        thr = nrm*reorth
        if abs(nrm) <= 10*np.finfo(Hsbg[0,0]).eps:
            # the norm of w[i+1] is effectively zero; it is linearly dependent
            return True
        elif nrm < -np.finfo(Hsbg[0,0]).eps:
            # the norm of w[i+1] < 0.0
            raise Exception(err_msg + "InnerProd(w[i+1], w[i+1]) = " + str(nrm) + " < 0.0")
            return False

        if i < 0:
            # Only one vector, so just normalize and exit
            w[:,i+1] = w[:,i+1]/nrm
            return False

        # Begin main Gram-Schmidt loop
        for k in xrange(0,i+1):
            prod = np.dot(w[:,i+1], w[:,k])
            Hsbg[k,i] = prod
            w[:,i+1] -= prod*w[:,k]
            # check if reorthogonalization is necessary
            if prod*prod > thr:
                prod = np.dot(w[:,i+1], w[:,k])
                Hsbg[k,i] += prod
                w[:,i+1] -= prod*(w[:,k])

            # update the norm and check its size
            nrm -= Hsbg[k,i]*Hsbg[k,i]
            if nrm < 0:
                nrm = 0.0
            thr = nrm*reorth

        # test the resulting vector
        nrm = np.linalg.norm(w[:,i+1])
        Hsbg[i+1,i] = nrm
        if nrm <= 10*np.finfo(Hsbg[0,0]).eps:
            return True
        else:
            # Scale the vector
            w[:,i+1] = w[:,i+1]/nrm
            return False

    #---------------------------------------------------------------------------

    def arnoldiSample_og(self, QoI, xdata, fdata, gdata, eigenvals, eigenvecs, grad_red):
        """
        Generates a sample of points, function values, and gradients using Arnoldi
        sampling.  Also returns a quadratic model based on the sampled gradients.

        **Inputs**

        * `QoI`: Quantity of interest

        **In/Outs**

        * `xdata`: array of sampled locations; on entry, xdata[:,1] is initial sample
        * `fdata`: array of sampled function values; fdata[:,1] is the initial function
        * `gdata`: array of sampled gradient values; gdata[:,1] is the initial gradient
        * `returnV` (optional): returns the Arnoldi vectors

        **Outputs**

        * `eigenvals`: eigenvalues of the quadratic model
        * `eigenvecs`: eigenvectors of the quadratic model
        * `grad_red`: the reduced gradient approximated using directional derivatives

        **Returns**

        * the size of the subspace; this may not equal num_sample-1
        * an error estimate for the Hessian-vector product

        """

        n = QoI.systemsize
        m = self.num_sample - 1

        # Assertions
        assert gdata.shape[0] == eigenvecs.shape[0] == n
        assert xdata.shape[1] == fdata.shape[0] == gdata.shape[1] == self.num_sample

        # Initialize the basis-vector array and Hessenberg matrix
        Z = np.zeros([n, m+1])
        H = np.zeros([m+1, m])
        Z[:,0] = gdata[:,0]/np.linalg.norm(gdata[:,0])
        linear_dependence = False

        for i in xrange(0, m):
            # Find new sample point and data; Compute function and gradient values
            xdata[:,i+1] = xdata[:,0] + self.alpha * Z[:,i]
            fdata[i+1] = QoI.eval_QoI(xdata[:,0], self.alpha*Z[:,i]) # TODO: Figure out a consistrnt API for function and gradient evaluation
            gdata[:,i+1] = QoI.eval_QoIGradient(xdata[:,0], self.alpha*Z[:,i])

            # Find the new basis vector and orthogonalize it against the old ones
            Z[:,i+1] = (gdata[:,i+1] - gdata[:,0])/self.alpha
            linear_dependence = self.modified_GramSchmidt(i, H, Z)
            if linear_dependence == True:
                # new basis vector is linealy dependent, so terminate early
                break

        if linear_dependence == True:
            i -= 1

        # Symmetrize the Hessenberg matrix, and find its eigendecomposition
        Hsym = 0.5*(H[0:i+1, 0:i+1] + H[0:i+1,0:i+1].transpose())
        eigenvals_red, eigenvecs_red = np.linalg.eig(Hsym)

        # Sort the reduced eigenvalues and eigenvectors reduced in ascending order
        idx = np.argsort(eigenvals_red)
        eigenvecs_red = eigenvecs_red[:,idx]
        eigenvals_red = eigenvals_red[idx]

        # Populate the system eigenvalue and eigenvectors
        eigenvals[:] = 0.0
        eigenvals[0:i+1] = eigenvals_red
        error_estimate = np.linalg.norm(0.5*(H[0:i+1,0:i+1] - H[0:i+1,0:i+1].transpose()))

        # Generate the full-space eigenvector approximations
        for k in xrange(0, i+1):
            eigenvecs[:,k] = Z[:,0:i+1].dot(eigenvecs_red[0:i+1, k])

        # Finally, sort the system eigenvalues and eigenvectors
        idx = np.argsort(eigenvals)
        eigenvecs = eigenvecs[:,idx]
        eigenvals = eigenvals[idx]

        # Generate the directional-derivative approximation to the reduced gradient
        tmp = (fdata[1:i+2] - np.ones(i+1)*fdata[0])/self.alpha
        grad_red[0:i+1] = eigenvecs_red[0:i+1, 0:i+1].transpose().dot(tmp)

        return i+1, error_estimate
