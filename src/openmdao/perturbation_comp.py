import numpy as np
import chaospy as cp

# from openmdao.api import Problem, IndepVarComp, Group
from openmdao.api import Problem, pyOptSparseDriver, SqliteRecorder, IndepVarComp, ExplicitComponent, Group
try:
    from openmdao.parallel_api import PETScVector
    vector = PETScVector
except:
    from openmdao.vectors.default_vector import DefaultVector
    vector = DefaultVector


class PerturbationComp(ExplicitComponent):

    def initialize(self):
        self.options.declare('degree', types=int)
        self.options.declare('n_random', types=int)
        self.options.declare('distribution_matrix', types=np.ndarray)
        self.options.declare('locations', types=np.ndarray)

    def setup(self):

        sc_degree = self.options['degree']
        uq_systemsize = self.options['n_random']
        n_sample = sc_degree**uq_systemsize
        # Inputs
        self.covariance = self.options['distribution_matrix']
        self.add_input('mu', shape=self.options['n_random'])
        self.xi = np.zeros([n_sample, uq_systemsize])
        idx = 0
        ctr = 0
        colloc_xi_arr = np.zeros(self.options['n_random'])
        ref_collocation_pts, _ = np.polynomial.hermite.hermgauss(self.options['degree'])
        sqrt_Sigma = np.sqrt(self.covariance)
        idx, ctr = self.compute_perturbation(sqrt_Sigma, ref_collocation_pts, colloc_xi_arr, self.xi, idx, ctr)
        assert idx == -1

        # Outputs
        self.add_output('xi', shape=(n_sample, uq_systemsize))

    def compute(self, inputs, outputs):

        # outputs['xi'] = (inputs['mu'] + self.xi.T).T
        outputs['xi'] = inputs['mu'] + self.xi
        # print(inputs['mu'] + self.xi)

    def compute_perturbation(self, sqrt_Sigma, ref_collocation_pts, colloc_xi_arr,
                             actual_location, idx, ctr):
        """
        # TODO: generalize this to multiple dimensions
        sqrt2 = np.sqrt(2)
        sqrt_Sigma = np.sqrt(self.covariance)
        for i in xrange(0, DEGREE):
            for j in xrange(0, DEGREE):
                idx = DEGREE*i + j
                self.xi[:,idx] = sqrt2* \
                                 np.matmul(sqrt_Sigma, np.array([SCLOC_Q[i], SCLOC_Q[j]]))
        """
        if idx == colloc_xi_arr.size-1:
            sqrt2 = np.sqrt(2)
            for i in xrange(0, ref_collocation_pts.size):
                # colloc_w_arr[idx] = ref_collocation_w[i] # Get the array of all the weights needed
                colloc_xi_arr[idx] = ref_collocation_pts[i] # Get the array of all the locations needed
                actual_location[ctr,:] = sqrt2*np.matmul(sqrt_Sigma, colloc_xi_arr)
                ctr += 1
            return idx-1, ctr
        else:
            for i in xrange(0, ref_collocation_pts.size):
                colloc_xi_arr[idx] = ref_collocation_pts[i]
                # colloc_w_arr[idx] = ref_collocation_w[i]
                idx, ctr = self.compute_perturbation(sqrt_Sigma, ref_collocation_pts,
                                        colloc_xi_arr,
                                        actual_location,
                                        idx+1, ctr)
            return idx-1, ctr
