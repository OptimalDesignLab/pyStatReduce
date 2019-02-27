# File that contains the class for Monte-Carlo methods
import numpy as np
import chaospy as cp

class MonteCarlo(object):
    """
    Base class for uncertainty propagation using monte carlo method`. The
    user must create an object of this class in order to propagate uncertainty
    using monte carlo.
    """
    def __init__(self, nsamples, jdist, QoI_dict, include_derivs=False, data_type=np.float):
        assert nsamples > 0, "Number of MonteCarlo samples must be greater than 0"
        self.num_samples = nsamples
        self.data_type = data_type # To enable complex variables
        self.QoI_dict = QoI_dict
        self.samples = jdist.sample(self.num_samples)
        for i in self.QoI_dict:
            self.QoI_dict[i]['fvals'] = np.zeros([self.num_samples,
                                        self.QoI_dict[i]['output_dimensions']],
                                        dtype=self.data_type)
        if include_derivs == True:
            for i in self.QoI_dict:
                for j in self.QoI_dict[i]['deriv_dict']:
                    self.QoI_dict[i]['deriv_dict'][j]['fvals'] =  np.zeros([self.num_samples,
                                                self.QoI_dict[i]['output_dimensions'],
                                                self.QoI_dict[i]['deriv_dict'][j]['output_dimensions']],
                                                dtype=self.data_type)

    def getSamples(self, jdist, include_derivs=False):
        n_rv = cp.E(jdist).shape
        pert = np.zeros(n_rv, dtype=self.data_type)
        # Get the all the function values for the given set of samples

        for i in range(0,self.num_samples):
            for j in self.QoI_dict:
                QoI_func = self.QoI_dict[j]['QoI_func']
                self.QoI_dict[j]['fvals'][i,:] = QoI_func(self.samples[:,i], pert)
                if include_derivs == True:
                    for k in self.QoI_dict[j]['deriv_dict']:
                        dQoI_func = self.QoI_dict[j]['deriv_dict'][k]['dQoI_func']
                        # print('\n j = ', j)
                        # print('self.num_samples = ', self.num_samples)
                        # print('dQoI = ', dQoI_func(self.samples[:,i], pert))
                        self.QoI_dict[j]['deriv_dict'][k]['fvals'][i,:] = dQoI_func(self.samples[:,i], pert)
        # for i in self.QoI_dict:
        #     QoI_func = self.QoI_dict[i]['QoI_func']
        #     self.QoI_dict[i]['fvals'] = [QoI_func(sample, pert) for sample in self.samples.T]

    def mean(self, jdist, of=None):
        mean_val = {}
        for i in of:
            if i in self.QoI_dict:
                mean_val[i] = np.mean(self.QoI_dict[i]['fvals'], 0)
        return mean_val

    def variance(self, jdist, of=None):
        variance_val = {}
        for i in of:
            if i in self.QoI_dict:
                variance_val[i] = np.var(self.QoI_dict[i]['fvals'], 0)

        return variance_val

    def dmean(self, jdist, of=None, wrt=None):
        dmean_val = {}
        for i in of:
            if i in self.QoI_dict:
                dmean_val[i] = {}
                for j in wrt:
                    if j in self.QoI_dict[i]['deriv_dict']:
                        dmean_val[i][j] = np.mean(self.QoI_dict[i]['deriv_dict'][j]['fvals'], 0)

        return dmean_val

    def dvariance(self, jdist, of=None, wrt=None):
        dvariance_val = {}
        for i in of:
            if i in self.QoI_dict:
                dvariance_val[i] = {}
                mu_j = np.mean(self.QoI_dict[i]['fvals'], 0)
                for j in wrt:
                    if j in self.QoI_dict[i]['deriv_dict']:
                        dmu_j = np.mean(self.QoI_dict[i]['deriv_dict'][j]['fvals'], 0)
                        # Finally, do the loop for the product
                        dval_j = np.zeros([self.QoI_dict[i]['output_dimensions'],
                                           self.QoI_dict[i]['deriv_dict'][j]['output_dimensions']],
                                           dtype=self.data_type)
                        for k in range(0, self.num_samples):
                            dval_j[:] += self.QoI_dict[i]['fvals'][k,:] *\
                                         self.QoI_dict[i]['deriv_dict'][j]['fvals'][k,:]
                        dvariance_val[i][j] = (dval_j - self.num_samples*mu_j*dmu_j) * 2 / (self.num_samples-1)

        return dvariance_val

    def dStdDev(self, jdist, of=None, wrt=None):
        dstd_dev_val = {}
        var = self.variance(jdist, of=of)
        dvar = self.dvariance(jdist, of=of, wrt=wrt)
        for i in of:
            if i in self.QoI_dict:
                dstd_dev_val[i] = {}
                for j in wrt:
                    if j in self.QoI_dict[i]['deriv_dict']:
                        dstd_dev_val[i][j] = 0.5 * dvar[i][j] / np.sqrt(var[i])
        return dstd_dev_val
