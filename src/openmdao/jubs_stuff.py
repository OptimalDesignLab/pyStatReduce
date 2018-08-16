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

# Import stuff from this  directory
from perturbation_comp import PerturbationComp
from stochastic_aggregation_comp import StochasticCollocation
from oas_aerodynamic_group import OASAerodynamic


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

class Rosenbrock_OMDAO(ExplicitComponent):
    """
    Base class that computes the deterministic Rosenbrock function.
    """

    def initialize(self):
        self.options.declare('size', types=int)

    def setup(self):
        self.add_input('rv', val = np.zeros(N_RANDOM)) # rv = random variable
        self.add_output('fval', val=0.0)
        self.declare_partials('fval', 'rv')

    def compute(self, inputs, outputs):
        rv = inputs['rv']
        outputs['fval'] = sum(100.0*(rv[1:]-rv[:-1]**2.0)**2.0 + (1-rv[:-1])**2.0)

    def compute_partials(self, inputs, J):
        rv = inputs['rv']
        xm = rv[1:-1]
        xm_m1 = rv[:-2]
        xm_p1 = rv[2:]
        der = np.zeros_like(rv)
        der[1:-1] = 200*(xm-xm_m1**2) - 400*(xm_p1 - xm**2)*xm - 2*(1-xm)
        der[0] = -400*rv[0]*(rv[1]-rv[0]**2) - 2*(1-rv[0])
        der[-1] = 200*(rv[-1]-rv[-2]**2)
        J['fval', 'rv'] = der


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
p.check_totals(of=['mu_j'], wrt=['mu'])
#Compute the total derivative of the partial
val = p.compute_totals(of=['mu_j'], wrt=['mu'])
print('partial val = ', val)
