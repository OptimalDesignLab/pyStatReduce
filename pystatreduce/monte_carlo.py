# File that contains the class for Monte-Carlo methods
import numpy as np
import chaospy as cp

class MonteCarlo(object):
    """
    Base class for uncertainty propagation using monte carlo method`. The
    user must create an object of this class in order to propagate uncertainty
    using monte carlo.
    """
    def __init__(self, nsamples, QoI_dict, data_type=np.float):
        assert nsamples > 0, "Number of MonteCarlo samples must be greater than 0"
        self.num_samples = nsamples
        self.data_type = data_type # To enable complex variables
        self.QoI_dict = QoI_dict
        # Create storage for all the QoI value of sampling points
        for i in QoI_dict:
            self.QoI_dict[i]['fvals'] = np.zeros([self.num_samples,
                                        self.QoI_dict[i]['output_dimensions']],
                                        dtype=self.data_type)


    def getSamples(self, jdist):
        self.samples = jdist.sample(self.num_samples)
        n_rv = cp.E(jdist).shape
        pert = np.zeros(n_rv)
        # Get the all the function values for the given set of samples
        for i in range(0,self.num_samples):
            for j in self.QoI_dict:
                QoI_func = self.QoI_dict[j]['QoI_func']
                # print "self.samples = ", self.samples[:,i]
                self.QoI_dict[j]['fvals'][i,:] = QoI_func(self.samples[:,i], pert)

    def mean(self, jdist, of=None, new_samples=False):
        mean_val = {}
        for i in of:
            if i in self.QoI_dict:
                mean_val[i] = np.sum(self.QoI_dict[i]['fvals'], axis=0) / self.num_samples
        return mean_val

    def variance(self, jdist, of=None, new_samples=False):
        variance_val = {}
        for i in of:
            if i in self.QoI_dict:
                mu = np.sum(self.QoI_dict[i]['fvals'], axis=0) / self.num_samples
                val = np.zeros((self.QoI_dict[i]['output_dimensions'],
                                self.QoI_dict[i]['output_dimensions']))
                for j in range(0,self.num_samples):
                    val += np.outer((self.QoI_dict[i]['fvals'][j,:] - mu),
                                       (self.QoI_dict[i]['fvals'][j,:] - mu))
                variance_val[i] = val / (self.num_samples - 1)

        return variance_val

    """
    def mean(self, QoI_func, jdist, mu_j, new_samples=False):
        if new_samples == True:
            self.get_samples(jdist)
        n_rv = cp.E(jdist).shape
        pert = np.zeros(n_rv) # Perturbation for feeding into QoI_func
        for i in xrange(0, self.num_samples):
            mu_j[:] += QoI_func(rv[:,i], pert)
        mu_j[:] = mu_j[:]/self.num_samples
        # return mu_j

    def variance(self, QoI_func, jdist, mu_j, var_j, new_samples=False):
        if new_samples == True:
            self.get_samples(jdist)
        n_rv = cp.E(jdist).shape
        pert = np.zeros(n_rv) # Perturbation for feeding into QoI_func
        for i in xrange(0, self.num_samples):
            fval = QoI_func(rv[:,i], pert)
            var_j[:] += np.outer(fval-mu_j, fval-mu_j) # (fval - mu_j)**2
        var_j[:] = var_j[:]/ (self.num_samples - 1)
        # return sigma_j
    """
    # def dvariance
