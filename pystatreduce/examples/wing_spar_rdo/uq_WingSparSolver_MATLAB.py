# WingsparUQ
import numpy as np
import chaospy as cp
# import pystatreduce.examples.wing_spar_rdo.source.WingSpar as ws
from pystatreduce.examples.wing_spar_rdo.source.WingSparSolver_MATLAB import SparSolver
from pystatreduce.quantity_of_interest import QuantityOfInterest
import pyoptsparse

# Plotting imports
from mpl_toolkits import mplot3d
import matplotlib
import matplotlib.pyplot as plt

np.set_printoptions(linewidth=150)

class UQSparSolver(QuantityOfInterest):

    def __init__(self, nelem, length=7.5, rho=1600.0, Young=70e9, Weight=0.5*500*9.8,
                 yield_stress=600e6, lb=0.01, up=0.05, minthick=0.0025, n_rv=4):

        self.det_spar_solver_obj = SparSolver(nelem, length=length, rho=rho,
                                             Young=Young, Weight=Weight,
                                             yield_stress=yield_stress, lb=lb,
                                             up=up, minthick=minthick, n_rv=n_rv)

        # Bookkeeping
        self.num_design = self.det_spar_solver_obj.num_design
        self.num_state = self.det_spar_solver_obj.num_state
        self.num_eq = self.det_spar_solver_obj.num_eq
        self.num_nonlin_ineq = self.det_spar_solver_obj.num_nonlin_ineq # 4*(nelem+1)
        self.num_lin_ineq = self.det_spar_solver_obj.num_lin_ineq
        self.nelem = self.det_spar_solver_obj.nelem

        # f_root = 2*(2.5*Weight) / length
        # self.force = (2*(2.5*Weight)/(length**2))*np.linspace(length,0.0,nelem+1)
        self.force = self.det_spar_solver_obj.force

        # Random variables
        self.num_rv = n_rv # Number of random variables in the problem
        # self.rv_arr = self.det_spar_solver_obj.xi # Create a shortcut
        self.num_nonlin_ineq = self.det_spar_solver_obj.num_nonlin_ineq

        # Design variables
        self.dv = np.zeros(self.det_spar_solver_obj.num_design)

        # Exporting functions from subclass
        # self.init_design = self.det_spar_solver_obj.init_design

        # Generate joint distribution for uncertainty propagation
        std_dev_matrix = self.generate_standard_deviation() # get the standard deviation
        self.jdist = cp.MvNormal(np.zeros(self.num_rv), std_dev_matrix)

    def update_design_variables(self, dv):
        # This function needs to be called before computing the stress constraints
        self.dv[:] = dv

    def update_rvs(self, rv_arr):
        self.det_spar_solver_obj.xi[:] = rv_arr[:]

    def eval_obj(self, at_design, at_state=np.zeros([])):
        self.update_design_variables(at_design)
        obj_val = self.det_spar_solver_obj.eval_obj(at_design, at_state)
        return obj_val

    def eval_stress_con_qoi(self, mean_val, perturbation):
        rv_arr = mean_val + perturbation
        self.update_rvs(rv_arr)
        con_val =  self.det_spar_solver_obj.eval_stress_constraint(self.dv)
        return con_val

    def generate_standard_deviation(self):
        # std_dev_arr = np.zeros(self.num_rv)
        std_dev_arr = 0.1 * self.force[0] / (np.arange(0, self.num_rv) + 1)
        return np.diag(std_dev_arr)

    def init_design(self):
        x0 = np.zeros(self.num_design)
        x0[0:self.nelem+1] = 0.0415*np.ones(self.nelem+1)
        x0[self.nelem+1:] = 0.05*np.ones(self.nelem+1)
        return x0

if __name__ == '__main__':

      # Initialize UQSparSolver
    nelem = 40
    uq_spar_solver_obj = UQSparSolver(nelem)
    design_vars = uq_spar_solver_obj.init_design()
    # print('design_vars = \n', design_vars)

    obj_val  = uq_spar_solver_obj.eval_obj(design_vars)
    print('obj_val = ', obj_val)

    # Compute the pert force
    temp_xi = np.array([-163.333333333333,
                        -81.6666666666667,
                        -54.4444444444445,
                        -40.8333333333333])
    con_val = uq_spar_solver_obj.eval_stress_con_qoi(temp_xi, np.zeros(4))
    print('con_val = ', con_val)

    # Lets compute the standard deviation of the robust constraint at the initial
    # design
    import chaospy as cp
    from pystatreduce.new_stochastic_collocation import StochasticCollocation2

    QoI_dict = {'stress_constraints': {'QoI_func' : uq_spar_solver_obj.eval_stress_con_qoi,
                                       'output_dimensions' : uq_spar_solver_obj.num_nonlin_ineq,
                                       'deriv_dict' : {},
                                       }
                }
    sc_obj = StochasticCollocation2(uq_spar_solver_obj.jdist, 2, 'MvNormal', QoI_dict,
                                        include_derivs=False, reduced_collocation=False)
    sc_obj.evaluateQoIs(uq_spar_solver_obj.jdist, include_derivs=False)
    # sc_obj.evaluateQoIs(uq_spar_solver_obj.jdist)

    mu_j = sc_obj.mean(of=['stress_constraints'])
    var_j = sc_obj.variance(of=['stress_constraints'])
    """
    mu_con_matlab = np.array([0.494755932174840,
                              0.458569174073115,
                              0.424191367348404,
                              0.391576128632065,
                              0.360677074555459,
                              0.331447821749942,
                              0.303841986846874,
                              0.277813186477613,
                              0.253315037273518,
                              0.230301155865948,
                              0.208725158886261,
                              0.188540662965815,
                              0.169701284735970,
                              0.152160640828084,
                              0.135872347873516,
                              0.120790022503623,
                              0.106867281349765,
                              0.094057741043301,
                              0.082315018215589,
                              0.071592729497987,
                              0.061844491521855,
                              0.053023920918550,
                              0.045084634319432,
                              0.037980248355859,
                              0.031664379659190,
                              0.026090644860783,
                              0.021212660591996,
                              0.016984043484189,
                              0.013358410168721,
                              0.010289377276949,
                              0.007730561440232,
                              0.005635579289929,
                              0.003958047457399,
                              0.002651582574000,
                              0.001669801271090,
                              0.000966320180029,
                              0.000494755932175,
                              0.000208725158886,
                              0.000061844491522,
                              0.000007730561440,
                              0.000000000000000])
    err = abs(mu_j['stress_constraints'] - mu_con_matlab)
    assert (err < 1.e-10).all()
    """

    print('mu_j = \n', mu_j['stress_constraints'])
    print('var_j = \n', var_j['stress_constraints'])
    robust_con_val = mu_j['stress_constraints'] + 6.0 * np.sqrt(var_j['stress_constraints'])  - np.ones(nelem + 1)
    print("nonlcon = \n", robust_con_val)
