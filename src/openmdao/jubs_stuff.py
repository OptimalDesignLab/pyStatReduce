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
        self.covariance = self.options['distribution_matrix']
        self.add_input('mu', shape=N_RANDOM)
        self.xi = np.zeros([N_RANDOM, N_SAMPLES])
        self.compute_perturbation()
        self.add_output('xi', shape=(N_RANDOM,N_SAMPLES))

    def compute(self, inputs, outputs):
        inputs['xi'] = inputs['mu'] + self.xi

    def compute_perturbation():
        # TODO: generalize this to multiple dimensions
        sqrt2 = np.sqrt(2)
        sqrt_Sigma = np.sqrt(self.covariance)
        for i in xrange(0, DEGREE):
            for j in xrange(0, DEGREE):
                idx = DEGREE*i + j
                self.xi[:,idx] = sqrt2* \
                                 np.matmul(sqrt_Sigma, np.array([SCLOC_Q[i], SCLOC_Q[j]]))


class Rosenbrock_OMDAO(ExplicitComponent):
    """
    Base class that computes the deterministic Rosenbrock function.
    """

    def initialize(self):
        self.options.declare('size', types=int)

    def setup(self):
        self.add_input('rv', val = np.zeros(N_RANDOM)) # rv = random variable
        self.add_output('fval', val=0.0)

    def compute(self, inputs, outputs):
        rv = inputs['rv']
        outputs['fval'] = sum(100.0*(rv[1:]-rv[:-1]**2.0)**2.0 + (1-rv[:-1])**2.0)


class StochasticCollocation(ExplicitComponent)
    """
    Class that aggregates the function values at the random variable quadrature
    locations.
    """
    def initialize(self):
        self.options.declare('degree', types=int)
        self.options.declare('n_random', types=int)
        self.options.declare('weights', types=np.ndarray)

    def setup(self):
        self.add_input('fval', val=0.0)
        self.add_output('mu_j', val=0.0)

    def compute(self, inputs, outputs):
        # Boo!

    def compute_weights





p.model.add_subsystem('perturb', PerturbationComp(degree=DEGREE,
                      n_random=N_RANDOM, distribution_matrix=co_mat,
                      locations=SCLOC_Q),
                      promotes_inputs=['mu'])
                      # xi will be of size N_SAMPLES


p.model.add_subsystem('multi_point', Group())

for i in range(N_SAMPLES):
    name = 'R{}'.format(i)
    p.model.multi_point.add_subsystem(name, Rosenbrock_OMDAO(size=N_RANDOM))
    p.model.connect('perturb.xi', '{}.rv'.format(name), src_indices=[i])
    p.model.connect('{}.fval'.format(name), 'stochastic_cloc.fval{}'.format(i))

p.model.add_subsystem('stochastic_cloc', StochasticCollocation(degree=DEGREE,
                                         n_random=N_RANDOM,
                                         weights=SCLOC_W),
                    promotes_outputs=['mu_j'])
