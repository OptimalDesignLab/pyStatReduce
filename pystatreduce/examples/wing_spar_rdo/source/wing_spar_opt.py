import numpy as np
import WingSpar as ws
from WingSparSolver import SparSolver
import pyoptsparse

# Plotting imports
from mpl_toolkits import mplot3d
import matplotlib
import matplotlib.pyplot as plt

ctr = 0
def objfunc(xdict):
    dv = xdict['radii']
    global ctr
    if ctr == 0:
        print('\ndv = ', dv)
        ctr += 1
    obj_val = spar_solver_obj.eval_obj(dv)
    ineq_constr_val = spar_solver_obj.eval_ineq_cnstr(dv)

    funcs = {}
    funcs['obj'] = obj_val
    funcs['ineq_constr_val'] = ineq_constr_val
    funcs['thickness_con'] = dv[(spar_solver_obj.nelem+1):]

    fail = False
    return funcs, fail

def sens(xdict, funcs):
    dv = xdict['radii']
    dobj_dx = spar_solver_obj.eval_dFdX(dv)
    dineqcon_dx = spar_solver_obj.eval_dCindX(dv)
    dthickness_con = np.zeros([(spar_solver_obj.nelem+1), dv.size])
    dthickness_con[:,(spar_solver_obj.nelem+1):] = np.eye((spar_solver_obj.nelem+1))
    # print('dthickness_con = \n', dthickness_con)

    funcsSens = {}
    funcsSens['obj', 'radii'] = dobj_dx
    funcsSens['ineq_constr_val', 'radii'] = dineqcon_dx
    funcsSens['thickness_con', 'radii'] = dthickness_con

    fail = False
    return funcsSens, fail

if __name__ == '__main__':

    # Initialize the wingspar object
    nElem = 20
    spar_solver_obj = SparSolver(nElem)

    # Get the initial design variables
    baseline_design = spar_solver_obj.init_design()
    print('baseline_design = ', baseline_design)

    # Initialize Optimizer object
    optProb = pyoptsparse.Optimization('wing_spar_opt', objfunc)
    optProb.addVarGroup('radii', spar_solver_obj.num_design, 'c',
                        value=baseline_design,
                        lower=spar_solver_obj.lb, upper=spar_solver_obj.up)
    ineq_con_ub = np.array([])
    ineq_con_lb = np.array([])

    optProb.addConGroup('ineq_constr_val', spar_solver_obj.num_nonlin_ineq, lower=0.)
    optProb.addConGroup('thickness_con', (spar_solver_obj.nelem+1), lower=0.0025)
    optProb.addObj('obj')
    opt = pyoptsparse.SNOPT(options = {'Major feasibility tolerance' : 1e-9,
                                       'Verify level' : 0})
    sol = opt(optProb, sens=sens)

    # print(sol)
    # print(repr(sol.__dict__.keys()))
    # print(repr(sol.xStar))

    # Get the optimal design variables
    optimal_dv = sol.xStar['radii']
    r_inner = optimal_dv[0:(nElem+1)]
    r_outer = r_inner + optimal_dv[(nElem+1):]
    length_discretization = np.linspace(0, spar_solver_obj.length, nElem+1)
    # print('spar_solver_obj.length = ', spar_solver_obj.length)
    # print('length_discretization = \n', length_discretization)
    # Plot
    fname = "spar_radii.pdf"
    plt.rc('text', usetex=True)
    matplotlib.rcParams['mathtext.fontset'] = 'cm'
    fig = plt.figure("radius_distribution", figsize=(6,6))
    ax = plt.axes()
    ax.plot(length_discretization, r_inner)
    ax.plot(length_discretization, r_outer)
    ax.set_xlabel('half wingspan')
    ax.set_ylabel('radii')
    plt.tight_layout()
    plt.show()
