# Run the uq_oas_problem
import numpy as np
import chaospy as cp

# Import the OpenMDAo shenanigans
from openmdao.api import IndepVarComp, Problem, Group, NewtonSolver, \
    ScipyIterativeSolver, LinearBlockGS, NonlinearBlockGS, \
    DirectSolver, LinearBlockGS, PetscKSP, SqliteRecorder

from openaerostruct.geometry.utils import generate_mesh
from openaerostruct.geometry.geometry_group import Geometry
from openaerostruct.aerodynamics.aero_groups import AeroPoint

# Import stuff from this  directory
from perturbation_comp import PerturbationComp
from stochastic_aggregation_comp import StochasticCollocation
from oas_aerodynamic_group import OASAerodynamic

# Declare the random variables, Assume that the random variables have a
# multivariate gaussian distribution eventhough it makes absolutely no physical
# sense
uq_systemsize = 2
mean_v = 248.136 # Mean value of input random variable
mean_alpha = 5   #
mean_Ma = 0.84
mean_re = 1.e6
mean_rho = 0.38
mean_cg = np.zeros((3))
# mu = np.array([mean_alpha, mean_Ma])
mu = np.array([mean_re, mean_rho])
# mu = np.array([mean_v, mean_cg])

covariance_matrix = np.eye(uq_systemsize) # corresponding covariance matrix
covariance_matrix[0,0] = 1.e2
covariance_matrix[1,1] = 0.02 # np.ones((3))
jdist = cp.MvNormal(mu, covariance_matrix)

# Stochastic collocation specific variables
sc_degree = 3 # Creates a 3^uq_systemsize grid
n_colloc_samples = sc_degree**uq_systemsize # Total number of collocation samples

p = Problem() # Create problem object

rvs = p.model.add_subsystem('random_variables', IndepVarComp(), promotes_outputs=['*'])
# rvs.add_output('mean_v', val=mean_v)
# rvs.add_output('mean_alpha', val=mean_alpha)
rvs.add_output('mu', val=np.array([mean_re, mean_rho]))

# Get the 1D Gauss-Hermite quadrature locations and weights
hermite_q, hermite_w = np.polynomial.hermite.hermgauss(sc_degree)

# Compute the locations where the OpenAeroStruct model needs to be evaluated.
p.model.add_subsystem('perturb', PerturbationComp(degree=sc_degree,
                                                  n_random=uq_systemsize,
                                                  distribution_matrix=covariance_matrix,
                                                  locations=hermite_q),
                      promotes_inputs=['mu'])

p.model.add_subsystem('multi_point', Group())

for i in range(n_colloc_samples):
    name = 'OAS{}'.format(i)
    p.model.multi_point.add_subsystem(name, OASAerodynamic())
    p.model.connect('perturb.xi', 'multi_point.{}.re'.format(name), src_indices=[(i,0)])
    p.model.connect('perturb.xi', 'multi_point.{}.rho'.format(name), src_indices=[(i,1)])

p.model.add_subsystem('stochastic_colloc', StochasticCollocation(degree=sc_degree,
                                                                 n_random=uq_systemsize,
                                                                 weights=hermite_w),
                      promotes_outputs=['mu_j'])

for i in range(n_colloc_samples):
    name = 'OAS{}'.format(i)
    p.model.connect('multi_point.{}.aero_point_0.CD'.format(name), 'stochastic_colloc.fval{}'.format(i))

p.setup()
p.run_model()
deriv = p.compute_totals(of=['mu_j'], wrt=['mu'])
print('deriv = ', deriv)
# data = p.check_partials()
