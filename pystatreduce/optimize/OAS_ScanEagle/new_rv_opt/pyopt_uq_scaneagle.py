################################################################################
# pyopt_uq_scaneagle_8rv.py
# This file performs a robust design optimizarion on a Boeing ScanEagle aircraft
# with 7 random variables. The random variables are Ma, TSFC, W0, E, G, mrho,
# load_factor, range.
# This file can be run as
#               `python pyopt_uq_scaneagle.py`
################################################################################

import sys
import time

# pyStatReduce specific imports
import numpy as np
import chaospy as cp
import copy
from pystatreduce.new_stochastic_collocation import StochasticCollocation2
from pystatreduce.quantity_of_interest import QuantityOfInterest
from pystatreduce.dimension_reduction import DimensionReduction
from pystatreduce.stochastic_arnoldi.arnoldi_sample import ArnoldiSampling
import pystatreduce.examples as examples
import pystatreduce.optimize.OAS_ScanEagle.oas_scaneagle_opt as scaneagle_opt

#pyoptsparse sepecific imports
from scipy import sparse
import argparse
import pyoptsparse # from pyoptsparse import Optimization, OPT, SNOPT

# Import the OpenMDAo shenanigans
from openmdao.api import IndepVarComp, Problem, Group, NewtonSolver, \
    ScipyIterativeSolver, LinearBlockGS, NonlinearBlockGS, \
    DirectSolver, LinearBlockGS, PetscKSP, SqliteRecorder, ScipyOptimizeDriver

from openaerostruct.geometry.utils import generate_mesh
from openaerostruct.geometry.geometry_group import Geometry
from openaerostruct.aerodynamics.aero_groups import AeroPoint

import pystatreduce.optimize.OAS_ScanEagle.check_scripts.optimal_vals_dict as optimal_vals_dict
from pystatreduce.optimize.OAS_ScanEagle.mean_values import *

def objfunc_uq(xdict):
    #
    # Objective funtion supplied to pyOptSparse for RDO.
    #
    rdo_factor = UQObj.rdo_factor
    UQObj.QoI.p['oas_scaneagle.wing.twist_cp'] = xdict['twist_cp']
    UQObj.QoI.p['oas_scaneagle.wing.thickness_cp'] = xdict['thickness_cp']
    UQObj.QoI.p['oas_scaneagle.wing.sweep'] = xdict['sweep']
    UQObj.QoI.p['oas_scaneagle.alpha'] = xdict['alpha']
    funcs = {}

    # Compute statistical metrics
    sc_obj.evaluateQoIs(UQObj.jdist)
    mu_j = sc_obj.mean(of=['fuelburn', 'constraints'])
    var_j = sc_obj.variance(of=['fuelburn', 'con_failure'])

    # The RDO Objective function is
    funcs['obj'] = mu_j['fuelburn'] + rdo_factor * np.sqrt(var_j['fuelburn'])

    # The RDO Constraint function is
    n_thickness_intersects = UQObj.QoI.p['oas_scaneagle.AS_point_0.wing_perf.thickness_intersects'].size
    n_CM = UQObj.QoI.p['oas_scaneagle.AS_point_0.CM'].size
    funcs['con_failure'] = mu_j['constraints'][0] + rdo_factor * np.sqrt(var_j['con_failure'][0,0])
    funcs['con_thickness_intersects'] = mu_j['constraints'][1:n_thickness_intersects+1]
    funcs['con_L_equals_W'] = mu_j['constraints'][n_thickness_intersects+1]
    funcs['con_CM'] = mu_j['constraints'][n_thickness_intersects+2:n_thickness_intersects+2+n_CM]
    funcs['con_twist_cp'] = mu_j['constraints'][n_thickness_intersects+2+n_CM:]

    fail = False
    return funcs, fail

def sens_uq(xdict, funcs):
    #
    # Sensitivity function provided to pyOptSparse for RDO.
    #
    rdo_factor = UQObj.rdo_factor
    UQObj.QoI.p['oas_scaneagle.wing.twist_cp'] = xdict['twist_cp']
    UQObj.QoI.p['oas_scaneagle.wing.thickness_cp'] = xdict['thickness_cp']
    UQObj.QoI.p['oas_scaneagle.wing.sweep'] = xdict['sweep']
    UQObj.QoI.p['oas_scaneagle.alpha'] = xdict['alpha']

    # Compute the statistical metrics
    sc_obj.evaluateQoIs(UQObj.jdist, include_derivs=True)
    dmu_js = sc_obj.dmean(of=['fuelburn', 'constraints'], wrt=['dv'])
    dstd_dev_js = sc_obj.dStdDev(of=['fuelburn', 'con_failure'], wrt=['dv'])

    # Get some of the intermediate variables
    n_twist_cp = UQObj.QoI.input_dict['n_twist_cp']
    n_cp = n_twist_cp + UQObj.QoI.input_dict['n_thickness_cp']
    n_CM = UQObj.QoI.input_dict['n_CM']
    n_thickness_intersects = UQObj.QoI.p['oas_scaneagle.AS_point_0.wing_perf.thickness_intersects'].size

    # Populate the dictionary
    funcsSens = {}
    dmu_j = dmu_js['fuelburn']['dv']
    dstd_dev_j = dstd_dev_js['fuelburn']['dv']
    funcsSens['obj', 'twist_cp'] = dmu_j[0,0:n_twist_cp] + rdo_factor * dstd_dev_j[0,0:n_twist_cp]
    funcsSens['obj', 'thickness_cp'] = dmu_j[0,n_twist_cp:n_cp] + rdo_factor * dstd_dev_j[0,n_twist_cp:n_cp]
    funcsSens['obj', 'sweep'] = dmu_j[0,n_cp:n_cp+1] + rdo_factor * dstd_dev_j[0,n_cp:n_cp+1]
    funcsSens['obj', 'alpha'] = dmu_j[0,n_cp+1:n_cp+2] + rdo_factor * dstd_dev_j[0,n_cp+1:n_cp+2]

    dmu_con = dmu_js['constraints']['dv']
    dstd_dev_con = dstd_dev_js['con_failure']['dv']
    funcsSens['con_failure', 'twist_cp'] = dmu_con[0,0:n_twist_cp] + rdo_factor * dstd_dev_con[0,0:n_twist_cp]
    funcsSens['con_failure', 'thickness_cp'] = dmu_con[0,n_twist_cp:n_cp] + rdo_factor * dstd_dev_con[0,n_twist_cp:n_cp]
    funcsSens['con_failure', 'sweep'] = dmu_con[0,n_cp] + rdo_factor * dstd_dev_con[0,n_cp]
    funcsSens['con_failure', 'alpha'] = dmu_con[0,n_cp+1] + rdo_factor * dstd_dev_con[0,n_cp+1]

    funcsSens['con_thickness_intersects', 'twist_cp'] = dmu_con[1:n_thickness_intersects+1,0:n_twist_cp]
    funcsSens['con_thickness_intersects', 'thickness_cp'] = dmu_con[1:n_thickness_intersects+1,n_twist_cp:n_cp]
    funcsSens['con_thickness_intersects', 'sweep'] = dmu_con[1:n_thickness_intersects+1,n_cp:n_cp+1]
    funcsSens['con_thickness_intersects', 'alpha'] = dmu_con[1:n_thickness_intersects+1,n_cp+1:]

    funcsSens['con_L_equals_W', 'twist_cp'] = dmu_con[n_thickness_intersects+1,0:n_twist_cp]
    funcsSens['con_L_equals_W', 'thickness_cp'] = dmu_con[n_thickness_intersects+1,n_twist_cp:n_cp]
    funcsSens['con_L_equals_W', 'sweep'] = dmu_con[n_thickness_intersects+1,n_cp]
    funcsSens['con_L_equals_W', 'alpha'] = dmu_con[n_thickness_intersects+1,n_cp+1]

    idx = n_thickness_intersects + 2
    funcsSens['con_CM', 'twist_cp'] = dmu_con[idx:idx+n_CM,0:n_twist_cp]
    funcsSens['con_CM', 'thickness_cp'] = dmu_con[idx:idx+n_CM,n_twist_cp:n_cp]
    funcsSens['con_CM', 'sweep'] = dmu_con[idx:idx+n_CM,n_cp:n_cp+1]
    funcsSens['con_CM', 'alpha'] = dmu_con[idx:idx+n_CM,n_cp+1:]

    idx = n_thickness_intersects + 2 + n_CM
    funcsSens['con_twist_cp', 'twist_cp'] = dmu_con[idx:,0:n_twist_cp]
    # funcsSens['con_twist_cp', 'thickness_cp'] = mu_con[idx:,n_twist_cp:n_cp]
    # funcsSens['con_twist_cp', 'sweep'] = mu_con[idx:,n_cp:n_cp+1]
    # funcsSens['con_twist_cp', 'alpha'] = mu_con[idx:,n_cp+1:]

    fail = False
    return funcsSens, fail

if __name__ == "__main__":

    sc_sol_dict = optimal_vals_dict.sc_sol_dict

    full_colloc = False # Bool for using full collocation or not

    # Declare the dictionary
    rv_dict = { 'Mach_number' : {'mean' : mean_Ma,
                                 'std_dev' : std_dev_Ma},
                'CT' : {'mean' : mean_TSFC,
                        'std_dev' : std_dev_TSFC},
                'W0' : {'mean' : mean_W0,
                        'std_dev' : std_dev_W0},
                'R' : {'mean' : mean_R,
                       'std_dev' : std_dev_R},
                'load_factor' : {'mean' : mean_load_factor,
                                 'std_dev' : std_dev_load_factor},
                'mrho' : {'mean' : mean_mrho,
                         'std_dev' : std_dev_mrho},
                'altitude' : {'mean' : mean_altitude,
                              'std_dev' : std_dev_altitude},
               }

    start_time = time.time()
    dict_val = 'sc_init' # 'act_init_7rv_2_2_lf1'

    # Print the sys.argv
    print('sys.argv[1] = ', sys.argv[1])
    print('sys.argv[2] = ', sys.argv[2])
    print('sys.argv[3] = ', sys.argv[3])

    if full_colloc == True:
        print('Full collocation = ', full_colloc)
        print('input seed = ', dict_val)
        UQObj = scaneagle_opt.UQScanEagleOpt(rv_dict, design_point=sc_sol_dict[dict_val],
                                             rdo_factor=float(sys.argv[1]))
    else:
        print('Full collocation = ', full_colloc)
        print('input seed = ', dict_val)
        UQObj = scaneagle_opt.UQScanEagleOpt(rv_dict, design_point=sc_sol_dict[dict_val],
                                             rdo_factor=float(sys.argv[1]),
                                             active_subspace=False,
                                             krylov_pert=float(sys.argv[2]),
                                             max_eigenmodes=int(sys.argv[3]))

    print()
    print("eigenvals = ", UQObj.dominant_space.iso_eigenvals)
    print('eigenvecs = \n', UQObj.dominant_space.iso_eigenvecs)
    print('dominant_dir = \n', UQObj.dominant_space.dominant_dir)
    print('#-------------------------------------------------------------#')

    # Get some information on the total number of constraints
    ndv = 3 + 3 + 1 + 1 # number of design variabels
    n_thickness_intersects = UQObj.QoI.p['oas_scaneagle.AS_point_0.wing_perf.thickness_intersects'].size
    n_CM = 3
    n_constraints = 1 + n_thickness_intersects  + 1 + n_CM + 3

    # Create the stochastic collocation object
    if full_colloc == True:
        sc_obj = StochasticCollocation2(UQObj.jdist, 3, 'MvNormal', UQObj.QoI_dict,
                                        include_derivs=True , reduced_collocation=False)
        sc_obj.evaluateQoIs(UQObj.jdist, include_derivs=True)
    else:
        sc_obj = StochasticCollocation2(UQObj.jdist, 3, 'MvNormal', UQObj.QoI_dict,
                                        include_derivs=True , reduced_collocation=True,
                                        dominant_dir=UQObj.dominant_space.dominant_dir)
        sc_obj.evaluateQoIs(UQObj.jdist, include_derivs=True)

    # Create the optimization problem for pyOptSparse
    optProb = pyoptsparse.Optimization('UQ_OASScanEagle', objfunc_uq)
    n_twist_cp = UQObj.QoI.input_dict['n_twist_cp']
    n_thickness_cp = UQObj.QoI.input_dict['n_thickness_cp']
    optProb.addVarGroup('twist_cp', n_twist_cp, 'c', lower=-5., upper=10,
                        value=sc_sol_dict[dict_val]['twist_cp'])
    optProb.addVarGroup('thickness_cp', n_thickness_cp, 'c', lower=0.001,
                        upper=0.01, scale=1.e3, value=sc_sol_dict[dict_val]['thickness_cp'])
    optProb.addVar('sweep', lower=10., upper=30., value=sc_sol_dict[dict_val]['sweep'])
    optProb.addVar('alpha', lower=-10., upper=10., value=sc_sol_dict[dict_val]['alpha'])

    # Constraints
    optProb.addConGroup('con_failure', 1, upper=0.)
    optProb.addConGroup('con_L_equals_W', 1, lower=0., upper=0.)
    optProb.addConGroup('con_thickness_intersects', n_thickness_intersects,
                        upper=0., wrt=['thickness_cp'])
    optProb.addConGroup('con_CM', n_CM, lower=-0.001, upper=0.001)
    optProb.addConGroup('con_twist_cp', 3, lower=np.array([-1e20, -1e20, 5.]),
                        upper=np.array([1e20, 1e20, 5.]), wrt=['twist_cp'])

    # Objective
    optProb.addObj('obj', scale=1.e-1)
    opt = pyoptsparse.SNOPT(options = {'Major feasibility tolerance' : 1e-9,
                                       'Verify level' : -1})
    sol = opt(optProb, sens=sens_uq)

    end_time = time.time()
    elapsed_time = end_time - start_time

    print(sol)
    print(sol.fStar)
    print()
    print("twist = ", UQObj.QoI.p['oas_scaneagle.wing.geometry.twist'])
    print("thickness =", UQObj.QoI.p['oas_scaneagle.wing.thickness'])
    print('\nthickness_cp = ', UQObj.QoI.p['oas_scaneagle.wing.thickness_cp'])
    print('twist_cp = ', UQObj.QoI.p['oas_scaneagle.wing.twist_cp'])
    print("sweep = ", UQObj.QoI.p['oas_scaneagle.wing.sweep'])
    print("aoa = ", UQObj.QoI.p['oas_scaneagle.alpha'])
    print()
    print('time elapsed = ', elapsed_time)

    # Compute the statistical moments
    mu_j = sc_obj.mean(of=['fuelburn'])
    var_j = sc_obj.variance(of=['fuelburn'])
    print('mu fuelburn = ', mu_j['fuelburn'])
    print('var fuelburn = ', var_j['fuelburn'])
