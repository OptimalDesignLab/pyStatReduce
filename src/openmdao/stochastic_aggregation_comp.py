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
    

class StochasticCollocation(ExplicitComponent):
    """
    Class that aggregates the function values at the random variable quadrature
    locations.
    """
    def initialize(self):
        self.options.declare('degree', types=int)
        self.options.declare('n_random', types=int)
        self.options.declare('weights', types=np.ndarray)

    def setup(self):
        print('Within StochasticCollocation setup')
        systemsize = self.options['n_random']
        degree = self.options['degree']

        # self.add_input('fval', val=np.zeros(degree**systemsize))

        # Since OpenMDAO does not have anything like target_indices, you need to
        # loop over all the components from which you need to get the output
        # function values.
        for i in xrange(0, degree**systemsize):
            self.add_input('fval{}'.format(i), val=0.0)

        ref_collocation_w = self.options['weights']
        idx = 0
        ctr = 0
        colloc_w_arr = np.zeros(systemsize)
        self.quadrature_weights = np.zeros(degree**systemsize)
        self.compute_weights(ref_collocation_w, colloc_w_arr,
                             self.quadrature_weights, idx, ctr)


        # Outputs
        self.add_output('mu_j', val=0.0)

    def compute(self, inputs, outputs):
        systemsize = self.options['n_random']
        degree = self.options['degree']
        J_determ = np.zeros(degree**systemsize)

        for i in xrange(0, degree**systemsize):
            J_determ[i] = inputs['fval{}'.format(i)]

        mu_j = np.sum(J_determ * self.quadrature_weights) / (np.sqrt(np.pi)**systemsize)
        outputs['mu_j'] = mu_j
        print('mu_j = ', mu_j)

    def compute_weights(self, ref_collocation_w, colloc_w_arr, quadrature_weights, idx, ctr):
        """
        ref_collocation_w = self.options['weights']
        for i in xrange(0, DEGREE):
            for j in xrange(0, DEGREE):
                idx = DEGREE*i + j
                self.quadrature_weights[idx] = ref_collocation_w[i] \
                                               * ref_collocation_w[j]
        """
        if idx == colloc_w_arr.size-1:
            for i in xrange(0, ref_collocation_w.size):
                colloc_w_arr[idx] = ref_collocation_w[i] # Get the array of all the weights needed
                quadrature_weights[ctr] = np.prod(colloc_w_arr)
                ctr += 1
            return idx-1, ctr
        else:
            for i in xrange(0, ref_collocation_w.size):
                colloc_w_arr[idx] = ref_collocation_w[i]
                idx, ctr = self.compute_weights(ref_collocation_w, colloc_w_arr,
                                                quadrature_weights,
                                                idx+1, ctr)
            return idx-1, ctr
