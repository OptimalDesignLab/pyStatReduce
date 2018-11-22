# QuantityOfInterest.py
import numpy as np

class QuantityOfInterest(object):
    """
    Base class for pyStatReduce quantity of interest (QoI) functions. It is
    designed to be a template for any user defined QoI. All user defined QoI
    objects should be a subclass of this class.

    **Class Members**

    * `systemsize` : Total number of random variables needed to evaluate a QoI

    **Constructor Arguments**

    * `nvariables` : total number of random variables

    """

    def __init__(self, nvariables, data_type=np.float):
        assert nvariables > 0, "Quantity of Interest cannot have zero inputs"
        self.systemsize = nvariables
        self.data_type = data_type

    def eval_QoI(self, mu, xi):
        """
        Evaluate the Quantity of interest at a given set of random variables.
        This method must be implemented for every QoI class, but should have the
        same function signature.

        **Arguments**

        * `mu` : Mean value of the random variables
        * `xi` : perturbation from the mean value that denotes the random
                 variable.

        A given random variable, q, must be deconstructed as

            q = mu + xi

        where, mu is the mean value and xi is the perturbation from the mean
        value

        """
        raise NotImplementedError

    def eval_QoIGradient(self, mu, xi):
        raise NotImplementedError

    def eval_QoIHessian(self, mu, xi):
        """
        Evaluate the Hessian of the quantity of interest with respect to the
        random variables. This method must be implemented for every QoI subclass,
        but should have the same function signature.

        **Arguments**

        * `mu` : Mean value of the random variables
        * `xi` : perturbation from the mean value that denotes the random
                 variable.

        A given random variable, q, must be deconstructed as

            q = mu + xi

        where, mu is the mean value and xi is the perturbation from the mean
        value
        """
        raise NotImplementedError
