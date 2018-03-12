# Arnoldi_sample
import numpy as np

class ArnoldiSampling(object):

    def __init__(self, alpha, num_sample):
        assert alpha > 0
        self.perturbation_size = alpha
        self.num_sample = num_sample

    def arnoldiSample(self, QoI, xdata, fdata, gdata, eigenvals, eigenvecs, grad_red):
        n = xdata.shape[0]
        m = self.num_sample - 1

        # Assertions
        assert gdata.shape[0] == eigenvecs.shape[0] == n
        assert xdata.shape[1] == fdata.shape[1] == gdata.shape[1] == self.num_sample

        # Initialize system-wide eigen value and eigen_vectors
        eigenvals = np.zeros(QoI.systemsize)
        eigenvecs = np.zeros([QoI.systemsize, QoI.systemsize])

        # Initialize the basis-vector array and Hessenberg matrix
        Z = np.zeros([n, m+1])
        H = np.zeros([m+1, m])
        Z[:,0] = gdata[:,0]/np.linalg.norm(gdata[:,0])

        for i in xrange(0, m):

            # Find new sample point and data; Compute function and gradient values
            xdata[:,i+1] = xdata[:,0] + alpha * Z[:,i]
            fdata[:,i+1] = QoI.eval_QoI(xdata[:,i+1]) # TODO: Figure out a consistrnt API for function and gradient evaluation
            gdata[:,i+1] = QoI.eval_QoI(xdata[i+1])

            # Find the new basis vector and orthogonalize it against the old ones
            Z[:,i+1] = (gdata[:,i+1] - gdata[:,0])/alpha
            linear_dependence = self.modified_GramSchmidt(i, H, Z)
            if linear_dependence == True:
                # new basis vector is linealy dependent, so terminate early
                break

        if linear_dependence == True:
            i -= 1

        # Symmetrize the Hessenberg matrix, and find its eigendecomposition
        Hsym = 0.5*(H[0:i+1, 0:i+1] + H[0:i+1,0:i+1].transpose())
        eigenvals[0,i+1], eigenvecs_red = np.linalg.eig(Hsym)
        error_estimate = np.linalg.norm(0.5*(H[0:i+1,0:i+1] - H[0:i+1,0:i+1].transpose()))

        # Generate the full-space eigenvector approximations
        for k in xrange(0, i+1):
            eigenvecs[:,k] = Z[:,0:i+1]*eigenvecs_red[0:i+1, k]

        # Generate the directional-derivative approximation to the reduced gradient
        tmp = (fdata[0,1:i+2].transpose() - ones(i,0)*fdata[0,0])/alpha
        grad_red[0:i+1] = eigenvecs_red[0:i+1, 0:i+1].transpose()*tmp

        return i, err_est, eigenvals, eigenvecs

    def modified_GramSchmidt(self, i, Hsbg, w):
        assert Hsbg.shape[1] >= i+1
        # print "i = ", i
        # print "Hsbg.shape = ", Hsbg.shape
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
        for k in xrange(0,i):
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
