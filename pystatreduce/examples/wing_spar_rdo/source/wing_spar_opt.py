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
    # global ctr
    # if ctr == 0:
    #     print('\ndv = ', dv)
    #     ctr += 1
    obj_val = spar_solver_obj.eval_obj(dv)
    ineq_constr_val = spar_solver_obj.eval_ineq_cnstr(dv)

    funcs = {}
    funcs['obj'] = obj_val
    funcs['ineq_constr_val'] = ineq_constr_val
    funcs['thickness_con'] = dv[(spar_solver_obj.nelem+1):]

    # Outer_radius_con
    outer_rad_val = dv[0:(spar_solver_obj.nelem+1)] + dv[(spar_solver_obj.nelem+1):]
    funcs['outer_radius_con'] = outer_rad_val

    fail = False
    return funcs, fail

"""
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
"""

if __name__ == '__main__':

    # Initialize the wingspar object
    nElem = 20
    spar_solver_obj = SparSolver(nElem)

    # Get the initial design variables
    baseline_design = spar_solver_obj.init_design()
    # MATLAB optimal design
    mat_r_in = np.array([0.0463800846549072,
                         0.0469501235894234,
                         0.0474534139190887,
                         0.0438990886775874,
                         0.0400102875021209,
                         0.0362403463368939,
                         0.0325930574074570,
                         0.0290725894960047,
                         0.0256835545249442,
                         0.0224311116491194,
                         0.0193212660309266,
                         0.0163624611569126,
                         0.0135756058993421,
                         0.0110582955427300,
                         0.0100000000000000,
                         0.0100000000000000,
                         0.0100000000000000,
                         0.0100000000000000,
                         0.0100000000000000,
                         0.0100000000000000,
                         0.0100000000000000])
    mat_r_out = np.array([0.0500000000000000,
                          0.0500000000000000,
                          0.0500000000000000,
                          0.0463990886775874,
                          0.0425102875021209,
                          0.0387403463368939,
                          0.0350930574074570,
                          0.0315725894960047,
                          0.0281835545249442,
                          0.0249311116491194,
                          0.0218212660309266,
                          0.0188624611569126,
                          0.0160756058993421,
                          0.0135582955427300,
                          0.0125000000000000,
                          0.0125000000000000,
                          0.0125000000000000,
                          0.0125000000000000,
                          0.0125000000000000,
                          0.0125000000000000,
                          0.0125000000000000])
    mat_thickness = mat_r_out - mat_r_in
    matlab_dv = np.concatenate((mat_r_in, mat_thickness), axis=0)

    # Initialize Optimizer object
    optProb = pyoptsparse.Optimization('wing_spar_opt', objfunc)
    optProb.addVarGroup('radii', spar_solver_obj.num_design, 'c',
                        value=baseline_design,
                        # value = matlab_dv,
                        lower=spar_solver_obj.lb, upper=spar_solver_obj.up)

    optProb.addConGroup('ineq_constr_val', spar_solver_obj.num_nonlin_ineq, lower=0., scale=100.)
    optProb.addConGroup('thickness_con', (spar_solver_obj.nelem+1), lower=0.0025)
    optProb.addConGroup('outer_radius_con', (spar_solver_obj.nelem+1), upper=0.05)
    optProb.addObj('obj')
    opt = pyoptsparse.SNOPT(options = {'Major feasibility tolerance' : 1e-10,
                                       'Verify level' : 0})
    sol = opt(optProb)# , sens='FD')

    print(sol)
    # print(repr(sol.__dict__.keys()))
    # print('optinal value = ', sol.fStar)
    # print('\n', repr(sol.xStar))

    # Get the optimal design variables
    optimal_dv = sol.xStar['radii']
    r_inner = optimal_dv[0:(nElem+1)]
    r_outer = r_inner + optimal_dv[(nElem+1):]
    length_discretization = np.linspace(0, spar_solver_obj.length, nElem+1)

    # Plot
    plotfigure = True
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

    fStar_matlab = spar_solver_obj.eval_obj(matlab_dv)
    # print('fStar_matlab = ', fStar_matlab)
    # print('fStar_snopt = ', sol.fStar)

    # print('r_inner = \n', repr(r_inner))
    # print('r_outer = \n', repr(r_outer))
