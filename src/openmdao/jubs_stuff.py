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


p = Problem()
dvs = p.model.add_subsystem('des_vars', IndepVarComp(), promotes_outputs=['*'])

N_RANDOM = 2
DEGREE = 3
N_SAMPLES = DEGREE**N_RANDOM
SCLOC_Q, SCLOC_W = np.polynomial.hermite.hermgauss(DEGREE)

x_init = np.zeros(N_RANDOM)
dvs.add_output('mu', val=x_init)

J_dist = cp.MvNormal(x_init, np.eye(N_RANDOM))
co_mat = cp.Cov(J_dist)

class PerturbationComp(ExplicitComponent):

    def initialize(self):
        self.options.declare('degree', types=int)
        self.options.declare('n_random', types=int)
        self.options.declare('distribution_matrix', types=np.ndarray)
        self.options.declare('locations', types=np.ndarray)

    def setup(self):
        print("Within PerturbationComp Setup")

        # Inputs
        self.covariance = self.options['distribution_matrix']
        self.add_input('mu', shape=self.options['n_random'])
        self.xi = np.zeros([N_SAMPLES, N_RANDOM])
        idx = 0
        ctr = 0
        colloc_xi_arr = np.zeros(self.options['n_random'])
        ref_collocation_pts, _ = np.polynomial.hermite.hermgauss(self.options['degree'])
        sqrt_Sigma = np.sqrt(self.covariance)
        idx, ctr = self.compute_perturbation(sqrt_Sigma, ref_collocation_pts, colloc_xi_arr, self.xi, idx, ctr)
        assert idx == -1

        # Outputs
        self.add_output('xi', shape=(N_SAMPLES, N_RANDOM))

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



class Rosenbrock_OMDAO(ExplicitComponent):
    """
    Base class that computes the deterministic Rosenbrock function.
    """

    def initialize(self):
        self.options.declare('size', types=int)

    def setup(self):
        print('Within Rosenbrock_OMDAO setup')
        self.add_input('rv', val = np.zeros(N_RANDOM)) # rv = random variable
        self.add_output('fval', val=0.0)

    def compute(self, inputs, outputs):
        rv = inputs['rv']
        outputs['fval'] = sum(100.0*(rv[1:]-rv[:-1]**2.0)**2.0 + (1-rv[:-1])**2.0)


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


p.model.add_subsystem('perturb', PerturbationComp(degree=DEGREE,
                      n_random=N_RANDOM, distribution_matrix=co_mat,
                      locations=SCLOC_Q),
                      promotes_inputs=['mu'])
                      # xi will be of size N_SAMPLES


p.model.add_subsystem('multi_point', Group())

for i in range(N_SAMPLES):
    name = 'R{}'.format(i)
    p.model.multi_point.add_subsystem(name, Rosenbrock_OMDAO(size=N_RANDOM))
    p.model.connect('perturb.xi', 'multi_point.{}.rv'.format(name), src_indices=[(i,0), (i,1)])
    # p.model.connect('{}.fval'.format(name), 'stochastic_cloc.fval{}'.format(i))

p.model.add_subsystem('stochastic_colloc', StochasticCollocation(degree=DEGREE,
                                         n_random=N_RANDOM,
                                         weights=SCLOC_W),
                                         promotes_outputs=['mu_j'])

for i in range(N_SAMPLES):
    name = 'R{}'.format(i)
    p.model.connect('multi_point.{}.fval'.format(name), 'stochastic_colloc.fval{}'.format(i))
    # p.model.connect('stochastic_colloc.fval', 'multi_point.{}.fval'.format(name), src_indices=[i])

p.setup()
p.run_model()
