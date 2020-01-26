import numpy as np
import WingSpar as ws
from WingSparSolver_MATLAB import SparSolver
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
    ineq_constr_val = spar_solver_obj.eval_stress_constraint(dv)

    funcs = {}
    funcs['obj'] = obj_val
    funcs['ineq_constr_val'] = ineq_constr_val
    funcs['thickness_con'] = dv[spar_solver_obj.num_nonlin_ineq:] - dv[0:spar_solver_obj.num_nonlin_ineq] # dv[(spar_solver_obj.nelem+1):]

    # Outer_radius_con
    outer_rad_val = dv[(spar_solver_obj.nelem+1):]
    funcs['outer_radius_con'] = outer_rad_val

    # Inner radius constraint
    inner_rad_val = dv[0:(spar_solver_obj.nelem+1)]
    funcs['inner_rad_con'] = inner_rad_val

    fail = False
    return funcs, fail

if __name__ == '__main__':

    # Initialize the wingspar object
    nElem = 20
    spar_solver_obj = SparSolver(nElem)

    # Get the initial design variables
    baseline_design = spar_solver_obj.init_design()

    # Initialize Optimizer object
    optProb = pyoptsparse.Optimization('wing_spar_opt', objfunc)
    optProb.addVarGroup('radii', spar_solver_obj.num_design, 'c',
                        value=baseline_design,
                        # value = matlab_dv,
                        # lower=spar_solver_obj.lb, upper=spar_solver_obj.up)
                        lower=-1.e10, upper=1.e10)

    optProb.addConGroup('ineq_constr_val', spar_solver_obj.num_nonlin_ineq, upper=0., scale=100.)
    optProb.addConGroup('thickness_con', (spar_solver_obj.nelem+1), lower=0.0025, scale=1.)
    optProb.addConGroup('outer_radius_con', (spar_solver_obj.nelem+1), upper=0.05)
    optProb.addConGroup('inner_rad_con', (spar_solver_obj.nelem+1), lower=0.01)
    optProb.addObj('obj',)
    opt = pyoptsparse.SNOPT(options = {'Major feasibility tolerance' : 1e-10,
                                       'Major optimality tolerance' : 1.e-8,
                                       'Verify level' : 0})
    sol = opt(optProb)# , sens='FD')

    print(sol)
    # Get the optimal design variables
    optimal_dv = sol.xStar['radii']
    r_inner = optimal_dv[0:(nElem+1)]
    r_outer = optimal_dv[(nElem+1):]
    length_discretization = np.linspace(0, spar_solver_obj.length, nElem+1)

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

    compare_optimal_vals = True
    if compare_optimal_vals:
        xStar_MATLAB = np.array([0.0463750957308978,
                                 0.0469453312433772,
                                 0.0474413985574117,
                                 0.0440168594749753,
                                 0.0401310661248841,
                                 0.0363643555230994,
                                 0.0327205444825895,
                                 0.0292038282797150,
                                 0.0258188435599749,
                                 0.0225707451289231,
                                 0.0194652996493684,
                                 0.0165089988854966,
                                 0.0137091924701694,
                                 0.0110742305667352,
                                 0.0100000000000000,
                                 0.0100000000000000,
                                 0.0100000000000000,
                                 0.0100000000000000,
                                 0.0100000000000000,
                                 0.0100000000000000,
                                 0.0100000000000000,
                                 0.0500000000000000,
                                 0.0500000000000000,
                                 0.0500000000000000,
                                 0.0465168594749753,
                                 0.0426310661248841,
                                 0.0388643555230994,
                                 0.0352205444825895,
                                 0.0317038282797150,
                                 0.0283188435599749,
                                 0.0250707451289231,
                                 0.0219652996493684,
                                 0.0190089988854966,
                                 0.0162091924701694,
                                 0.0135742305667352,
                                 0.0125000000000000,
                                 0.0125000000000000,
                                 0.0125000000000000,
                                 0.0125000000000000,
                                 0.0125000000000000,
                                 0.0125000000000000,
                                 0.0125000000000000])

        xStar_SNOPT = sol.xStar['radii']
        err = abs(xStar_SNOPT - xStar_MATLAB)
        print('\nerr = \n', err)
