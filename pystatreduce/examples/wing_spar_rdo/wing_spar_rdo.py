import numpy as np
import chaospy as cp
from pystatreduce.examples.wing_spar_rdo.uq_WingSparSolver_MATLAB import UQSparSolver
from pystatreduce.new_stochastic_collocation import StochasticCollocation2
from pystatreduce.quantity_of_interest import QuantityOfInterest
from pystatreduce.dimension_reduction import DimensionReduction
import pyoptsparse

# Plotting imports
from mpl_toolkits import mplot3d
import matplotlib
import matplotlib.pyplot as plt

ctr = 0
def objfunc(xdict):
    dv = xdict['radii']

    # Compute the objective function
    obj_val = uq_spar_solver_obj.eval_obj(dv)

    # Compute the statistical metrics
    uq_spar_solver_obj.update_design_variables(dv) # Update design variables
    sc_obj.evaluateQoIs(uq_spar_solver_obj.jdist)
    mu_j = sc_obj.mean(of=['stress_constraints'])
    var_j = sc_obj.variance(of=['stress_constraints'])
    std_dev_stress_con = np.sqrt(np.diagonal(var_j['stress_constraints']))
   

    # Put the values in dictionary
    funcs = {}
    funcs['obj'] = obj_val
    funcs['thickness_con'] = dv[(uq_spar_solver_obj.nelem+1):] - dv[0:(uq_spar_solver_obj.nelem+1)]

    # Inequalty constraints
    funcs['uq_stress_con'] = mu_j['stress_constraints'] + 2*std_dev_stress_con

    # Outer_radius_con
    outer_rad_val = dv[(uq_spar_solver_obj.nelem+1):] # dv[0:(uq_spar_solver_obj.nelem+1)] + dv[(uq_spar_solver_obj.nelem+1):]
    funcs['outer_radius_con'] = outer_rad_val

    # Inner radius constraint
    inner_rad_val = dv[0:(uq_spar_solver_obj.nelem+1)]
    funcs['inner_rad_con'] = inner_rad_val

    global ctr
    if ctr < 1:
        print('uq_stress_con = \n', repr(funcs['uq_stress_con']))
        print('mu_j = \n', mu_j['stress_constraints'])
        print('var_j = \n', np.diagonal(var_j['stress_constraints']))
        ctr += 1

    fail = False
    return funcs, fail

if __name__ == '__main__':

    # Initialize UQ WingSpar Solver
    nelem = 15
    uq_spar_solver_obj = UQSparSolver(nelem)

    # Get the initial design variables
    baseline_design = uq_spar_solver_obj.init_design()
    print('baseline_design = \n', baseline_design)

    # # Initialize the stochastic collocation object
    # rv = {'load_perturbations' : {'mean' : np.array([]),
    #                               'variance' : np.array([]),
    #                               }
    #       }

    QoI_dict = {'stress_constraints': {'QoI_func' : uq_spar_solver_obj.eval_stress_con_qoi,
                                       'output_dimensions' : uq_spar_solver_obj.num_nonlin_ineq,
                                       'deriv_dict' : {},
                                       }
                }
    full_colloc = True
    if full_colloc == True:
        sc_obj = StochasticCollocation2(uq_spar_solver_obj.jdist, 2, 'MvNormal', QoI_dict,
                                        include_derivs=False, reduced_collocation=False)
        sc_obj.evaluateQoIs(uq_spar_solver_obj.jdist, include_derivs=False)
    else:
        raise NotImplementedError

    # Initialize Optimizer object
    optProb = pyoptsparse.Optimization('wing_spar_opt', objfunc)
    optProb.addVarGroup('radii', uq_spar_solver_obj.num_design, 'c',
                        value=baseline_design,
                        # value = matlab_dv,
                        lower=-1.e10, upper=1.e10)

    optProb.addConGroup('uq_stress_con', uq_spar_solver_obj.num_nonlin_ineq, lower=0., scale=1.)
    optProb.addConGroup('thickness_con', (uq_spar_solver_obj.nelem+1), lower=0.0025)
    optProb.addConGroup('outer_radius_con', (uq_spar_solver_obj.nelem+1), upper=0.05)
    optProb.addConGroup('inner_rad_con', (uq_spar_solver_obj.nelem+1), lower=0.01)
    optProb.addObj('obj')
    opt = pyoptsparse.SNOPT(options = {'Major feasibility tolerance' : 1e-10,
                                       'Major optimality tolerance' : 1.e-8,
                                       'Verify level' : 0})
    sol = opt(optProb)# , sens='FD')

    # print(sol)
    # print(repr(sol.__dict__.keys()))
    print('optimal value = ', sol.fStar)
    print('\n', repr(sol.xStar))

    # Get the optimal design variables
    optimal_dv = sol.xStar['radii']
    r_inner = optimal_dv[0:(nelem+1)]
    r_outer = optimal_dv[(nelem+1):]
    length_discretization = np.linspace(0, uq_spar_solver_obj.det_spar_solver_obj.length, nelem+1)

    # Plot
    plotfigure = False
    if plotfigure:
        fname = "spar_radii.pdf"
        plt.rc('text', usetex=True)
        matplotlib.rcParams['mathtext.fontset'] = 'cm'
        fig = plt.figure("radius_distribution", figsize=(6,6))
        ax = plt.axes()
        ax.plot(length_discretization, r_inner)
        ax.plot(length_discretization, r_inner, 'o')
        ax.plot(length_discretization, r_outer)
        ax.plot(length_discretization, r_outer, 'o')
        ax.set_xlabel('half wingspan')
        ax.set_ylabel('radii')
        plt.tight_layout()
        plt.show()
