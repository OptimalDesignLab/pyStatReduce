import numpy as np
import chaospy as cp
import openturns as ot
import warnings

class Distribution(object):
    def __init__(self, package="chaospy", distribution_type="Normal", mean_val=0,
                 standard_deviation=1, correleation_matrix=None, **kwargs):
        """
        This is wrapper to standardize operations between multiple UQ analysis
        tools. Currently the aim is to support ChaosPy, OpenTurns, and DAKOTA
        here.
        """

        self.uq_package_name=package
        self.n_rv = 0 # Number of random variables (initialized to 0)

        # If the User wants to use ChaosPy
        if package is "chaospy":
            if distribution_type is "Normal":
                if np.isscalar(mean_val) and np.isscalar(standard_deviation):
                    # Create a univariate distribution
                    self.distribution = cp.Normal(mu=mean_val, sigma=standard_deviation)
                    self.n_rv = 1
                elif mean_val.shape == standard_deviation.shape:
                    # Create a Multivariate Distribution
                    if correleation_matrix is None:
                        covariance = np.diag(standard_deviation**2)
                    else:
                        # Construct the covariance matrix
                        D = np.diag(standard_deviation)
                        covariance = np.dot(D, np.dot(correleation_matrix, D))
                    self.distribution = cp.MvNormal(mean_val, covariance)
                    self.n_rv = mean_val.size
                else:
                    raise ValueError('Incorrect data structures for the mean and standard deviation. 1D numpy array expected.')
                self.mean = mean_val
                self.standard_deviation = standard_deviation
            elif distribution_type is "Uniform":
                raise NotImplementedError
            else:
                raise NotImplementedError

        # If the User wants to use OpenTurns
        elif package is "openturns":
            if distribution_type is "Normal":
                if np.isscalar(mean_val) and np.isscalar(standard_deviation):
                    self.distribution = ot.Normal(mean_val, standard_deviation)
                    self.n_rv = 1
                elif mean_val.shape == standard_deviation.shape:
                    self.n_rv = mean_val.size
                    if correleation_matrix is None:
                        print("OpenTurns expects a correlation matrix, since none is provided the random variables are assumed to uncorrelated.")
                        self.distribution = ot.Normal(mean_val, standard_deviation, ot.CorrelationMatrix(self.n_rv))
                    else:
                        self.distribution = ot.Normal(mean_val, standard_deviation, correleation_matrix)
                else:
                    raise ValueError('Incorrect data structures for the mean and standard deviation. 1D numpy array expected.')
            elif distribution_type is "Uniform":
                raise NotImplementedError
            else:
                raise NotImplementedError

        elif package is "dakota":
            raise NotImplementedError

        else:
            raise ValueError('Only chaospy, openturns, and dakota are supported.')

    def E(self):
        """
        Returns the mean value from the distribution object
        """
        # self.check_distribution_validity(distribution)
        if self.uq_package_name is "chaospy":
            return cp.E(self.distribution)
        elif self.uq_package_name is "openturns":
            val = np.array(self.distribution.getMean())
            if self.n_rv == 1:
                return np.float(val)
            else:
                return val
        else:
            # DAKOTA
            raise NotImplementedError

    def Cov(self):
        """
        Returns the covariance matrix from the distribution object
        """
        # self.check_distribution_validity(distribution)
        if self.uq_package_name is "chaospy":
            if self.n_rv == 1:
                return np.float(cp.Cov(self.distribution))
            else:
                return cp.Cov(self.distribution)
        elif self.uq_package_name is "openturns":
            val = np.array(self.distribution.getCovariance())
            if self.n_rv == 1:
                return np.float(val)
            else:
                return val
        else:
            # DAKOTA
            raise NotImplementedError

    def Std(self):
        """
        Returns the standard deviation for a given distribution.
        """
        # self.check_distribution_validity(distribution)
        if self.uq_package_name is "chaospy":
            return cp.Std(self.distribution)
        elif self.uq_package_name is "openturns":
            val = np.array(self.distribution.getStandardDeviation())
            if self.n_rv == 1:
                return np.float(val)
            else:
                return val
        else:
            # DAKOTA
            raise NotImplementedError

    def check_distribution_validity(self, distribution):
        if distribution is not self.distribution:
            warnings.warn('The distribution supplied is external to the one constructed. Value returned from the internal distribution.')


if __name__ == '__main__':
    pass
