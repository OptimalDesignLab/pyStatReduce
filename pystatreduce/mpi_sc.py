from mpi4py import MPI
import numpy as np
import cmath
import chaospy as cp

from pystatreduce.new_stochastic_collocation import StochasticCollocation2
from pystatreduce.stochastic_collocation import StochasticCollocation
from pystatreduce.quantity_of_interest import QuantityOfInterest
from pystatreduce.dimension_reduction import DimensionReduction
import pystatreduce.examples as examples

# MPI.Init()

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

systemsize = 3
mu = np.random.randn(systemsize)
std_dev = np.diag(np.random.rand(systemsize))
jdist = cp.MvNormal(mu, std_dev)
# Create QoI Object
QoI = examples.Paraboloid3D(systemsize)

# Create the Stochastic Collocation object
deriv_dict = {'xi' : {'dQoI_func' : QoI.eval_QoIGradient,
                      'output_dimensions' : systemsize}
             }
QoI_dict = {'paraboloid' : {'QoI_func' : QoI.eval_QoI,
                            'output_dimensions' : 1,
                            'deriv_dict' : deriv_dict
                            }
            }
sc_obj = StochasticCollocation2(jdist, 2, 'MvNormal', QoI_dict, mpi_comm=comm)
sc_obj.evaluateQoIs(jdist)
mu_js = sc_obj.mean(of=['paraboloid'])
var_js = sc_obj.variance(of=['paraboloid'])

# Analytical mean
mu_j_analytical = QoI.eval_QoI_analyticalmean(mu, cp.Cov(jdist))
err = abs((mu_js['paraboloid'][0] - mu_j_analytical)/ mu_j_analytical)
if rank == 0:
    print('err = ', err)
# self.assertTrue(err < 1.e-15)

# sc_obj2 = StochasticCollocation2(jdist, 2, 'MvNormal', QoI_dict, mpi_comm=None)
# sc_obj2.evaluateQoIs(jdist)

print()

MPI.Finalize()
