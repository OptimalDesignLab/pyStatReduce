import sys
import time

# pyStatReduce specific imports
import unittest
import numpy as np
import chaospy as cp
import copy
from pystatreduce.new_stochastic_collocation import StochasticCollocation2
from pystatreduce.stochastic_collocation import StochasticCollocation
from pystatreduce.monte_carlo import MonteCarlo
from pystatreduce.quantity_of_interest import QuantityOfInterest
from pystatreduce.dimension_reduction import DimensionReduction
from pystatreduce.stochastic_arnoldi.arnoldi_sample import ArnoldiSampling
import pystatreduce.examples as examples
import pystatreduce.utils as utils

np.set_printoptions(precision=12)
np.set_printoptions(linewidth=150, suppress=True)

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

# Plotting modules
from mpl_toolkits import mplot3d
import matplotlib
import matplotlib.pyplot as plt

# Default mean values
mean_Ma = 0.071
mean_TSFC = 9.80665 * 8.6e-6
mean_W0 = 10.0
mean_E = 85.e9
mean_G = 25.e9
mean_mrho = 1600
mean_R = 1800e3
mean_load_factor = 1.0
mean_altitude = 4.57e3
# Default standard values
std_dev_Ma = 0.005
std_dev_TSFC = 0.00607/3600
std_dev_W0 = 0.2
std_dev_mrho = 50
std_dev_R = 500.e3
std_dev_load_factor = 0.1
std_dev_E = 5.e9
std_dev_G = 1.e9
std_dev_altitude = 0.5e3

if sys.argv[1] == 'generate_data':
    # Gather all the system arguments
    rv_key = sys.argv[2] # key for deciding which random variable to generate data for
    dv_dict_key = sys.argv[3] # key for picking the design variables from the dictionary
                              # optimal_dv_dict below

    # Total number of nodes to use in the spanwise (num_y) and
    # chordwise (num_x) directions. Vary these to change the level of fidelity.
    num_y = 21
    num_x = 3
    mesh_dict = {'num_y' : num_y,
                 'num_x' : num_x,
                 'wing_type' : 'rect',
                 'symmetry' : True,
                 'span_cos_spacing' : 0.5,
                 'span' : 3.11,
                 'root_chord' : 0.3,
                }
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
                # 'E' : {'mean' : mean_E,
                #        'std_dev' : std_dev_E},
                # 'G' : {'mean' : mean_G,
                #        'std_dev' : std_dev_G},
                'mrho' : {'mean' : mean_mrho,
                         'std_dev' : std_dev_mrho},
                'altitude' : {'mean' : mean_altitude,
                              'std_dev' : std_dev_altitude},
               }

    # String keys should be specified as numberof random variables followed by
    # underscore followed by arnoldi sample radius followed by underscore,
    # followed number of dominant directions, followed by the rdo_factor.
    optimal_dv_dict = { # 6 RV, sample radius = 1e-1, n_dominant_dir = 1, rdo factor = 2
                        '6rv_1e_1_1_2' : {'twist_cp' : np.array([2.596540, 10, 5]),
                                          'thickness_cp' : 1.e-3 * np.ones(3),
                                          'sweep' : 18.87097254,
                                          'alpha' : 2.09881337 },
                        # 6 RV, sample radius = 1e-1, n_dominant_dir = 2, rdo factor = 2
                        '6rv_1e_1_2_2' : {'twist_cp' : np.array([2.603630, 10., 5.0]),
                                          'thickness_cp' : 1.e-3 * np.ones(3),
                                          'sweep' : 18.85350968,
                                          'alpha' : 2.0316547 },
                        # 6 RV, sample radius = 1e-1, n_dominant_dir = 3, rdo factor = 2
                        '6rv_1e_1_3_2' : {'twist_cp' : np.array([2.646244, 10., 5.0]),
                                          'thickness_cp' : 1.e-3 * np.ones(3),
                                          'sweep' : 18.81694237,
                                          'alpha' : 1.91196833 },
                        # 6 RV, sample radius = 1e-1, n_dominant_dir = 4, rdo factor = 2
                        '6rv_1e_1_4_2' : {'twist_cp' : np.array([2.621730, 10., 5.0]),
                                          'thickness_cp' : 1.e-3 * np.ones(3),
                                          'sweep' : 18.82486945,
                                          'alpha' : 1.93058918 },
                        # 6 RV, sample radius = 1e-1, n_dominant_dir = 5, rdo factor = 2
                        '6rv_1e_1_5_2' : {'twist_cp' : np.array([2.622514, 10., 5.0]),
                                          'thickness_cp' : 1.e-3 * np.ones(3),
                                          'sweep' : 18.82404409,
                                          'alpha' : 1.92780292 },
                        # 6 RV, sample radius = 1e-1, n_dominant_dir = 6, rdo factor = 2
                        '6rv_mc_full_2' : {'twist_cp' : np.array([2.640780, 10., 5.0]),
                                          'thickness_cp' : 1.e-3 * np.ones(3),
                                          'sweep' : 18.81181131,
                                          'alpha' : 1.8905709 },

                        'deterministic' : {'twist_cp' : np.array([2.61227376, 10, 5]) ,
                                           'thickness_cp' : 1.e-3 * np.ones(3) ,
                                           'sweep' : 18.88466275 ,
                                           'alpha' : 2.1682715 },

                        'initial' : {'twist_cp' : np.array([2.5, 2.5, 5.0]),
                                     'thickness_cp' : np.array([0.008, 0.008, 0.008]),
                                     'sweep' : 20.0,
                                     'alpha' : 5.},
                      }

    uq_systemsize = len(rv_dict)
    mu, std_dev = utils.get_scaneagle_input_rv_statistics(rv_dict)
    jdist = cp.MvNormal(mu, std_dev)

    input_dict = {'n_twist_cp' : 3,
               'n_thickness_cp' : 3,
               'n_CM' : 3,
               'n_thickness_intersects' : 10,
               'n_constraints' : 1 + 10 + 1 + 3 + 3,
               'ndv' : 3 + 3 + 2,
               'mesh_dict' : mesh_dict,
               'rv_dict' : rv_dict
                }

    QoI = examples.oas_scaneagle2.OASScanEagleWrapper2(uq_systemsize, input_dict)

    # Step 1: Instantiate all objects needed for test
    QoI.p['oas_scaneagle.wing.thickness_cp'] = optimal_dv_dict[dv_dict_key]['thickness_cp']
    QoI.p['oas_scaneagle.wing.twist_cp'] = optimal_dv_dict[dv_dict_key]['twist_cp']
    QoI.p['oas_scaneagle.wing.sweep'] = optimal_dv_dict[dv_dict_key]['sweep']
    QoI.p['oas_scaneagle.alpha'] = optimal_dv_dict[dv_dict_key]['alpha']
    QoI.p.final_setup()



    rdo_factor = float(sys.argv[4])
    npts = 101 # Number of data points for plotting
    rv_lb = rv_dict[rv_key]['mean'] - 0.5 * rdo_factor * rv_dict[rv_key]['std_dev']
    rv_ub = rv_dict[rv_key]['mean'] + 0.5 * rdo_factor * rv_dict[rv_key]['std_dev']
    rv_arr = np.linspace(rv_lb, rv_ub, num=npts)
    fval_arr = np.zeros(npts)
    for i in range(0, npts):
        QoI.p[rv_key] = rv_arr[i]
        QoI.p.run_model()
        fval_arr[i] = QoI.p['oas_scaneagle.AS_point_0.fuelburn']

    # print('rv_arr = ', rv_arr)
    # print('fval_arr = ', fval_arr)
    combined_arr = np.array([rv_arr,fval_arr])
    # Write to file
    fname = './' + rv_key + '/fburn_vs_' + rv_key + '_' + dv_dict_key + '.txt'
    np.savetxt(fname, combined_arr)

elif sys.argv[1] == 'plot':

    rv_name = sys.argv[2]
    # filepath = './' + rv_name + '/' + sys.argv[3]
    # combined_arr = np.loadtxt(filepath)

    # Load the deterministic file
    det_fpath = './' + rv_name + '/' + '/fburn_vs_' + rv_name + '_deterministic.txt'
    det_combined_arr = np.loadtxt(det_fpath)

    # Load UQ files
    fpath1 = './' + rv_name + '/' + '/fburn_vs_' + rv_name + '_6rv_1e_1_1_2.txt'
    fpath2 = './' + rv_name + '/' + '/fburn_vs_' + rv_name + '_6rv_1e_1_2_2.txt'
    fpath3 = './' + rv_name + '/' + '/fburn_vs_' + rv_name + '_6rv_1e_1_3_2.txt'
    fpath4 = './' + rv_name + '/' + '/fburn_vs_' + rv_name + '_6rv_1e_1_4_2.txt'
    fpath5 = './' + rv_name + '/' + '/fburn_vs_' + rv_name + '_6rv_1e_1_5_2.txt'
    fpath_mc = './' + rv_name + '/' + '/fburn_vs_' + rv_name + '_6rv_mc_full_2.txt'

    comb_arr1 = np.loadtxt(fpath1)
    comb_arr2 = np.loadtxt(fpath2)
    comb_arr3 = np.loadtxt(fpath3)
    comb_arr4 = np.loadtxt(fpath4)
    comb_arr5 = np.loadtxt(fpath5)
    comb_arr_mc = np.loadtxt(fpath_mc)

    # Split the combined array
    rv_arr1 = comb_arr1[0,:]
    fburn_arr1 = comb_arr1[1,:]
    rv_arr2 = comb_arr2[0,:]
    fburn_arr2 = comb_arr2[1,:]
    rv_arr3 = comb_arr3[0,:]
    fburn_arr3 = comb_arr3[1,:]
    rv_arr4 = comb_arr4[0,:]
    fburn_arr4 = comb_arr4[1,:]
    rv_arr5 = comb_arr5[0,:]
    fburn_arr5 = comb_arr5[1,:]
    rv_arr_mc_full = comb_arr_mc[0,:]
    fburn_arr_mc_full = comb_arr_mc[1,:]

    # Make some assertions
    np.testing.assert_array_equal(rv_arr1, det_combined_arr[0,:])
    np.testing.assert_array_equal(rv_arr2, det_combined_arr[0,:])
    np.testing.assert_array_equal(rv_arr3, det_combined_arr[0,:])
    np.testing.assert_array_equal(rv_arr4, det_combined_arr[0,:])
    np.testing.assert_array_equal(rv_arr5, det_combined_arr[0,:])
    np.testing.assert_array_equal(rv_arr_mc_full, det_combined_arr[0,:])
    det_fburn_arr = det_combined_arr[1,:]

    # Actually plot
    fname = './' + rv_name + '/fburn_vs_' + rv_name +'.pdf'
    # plt.rc('text', usetex=True)
    matplotlib.rcParams['mathtext.fontset'] = 'cm'
    matplotlib.rcParams.update({'font.size': 14})
    fig = plt.figure(figsize=(5,5))
    ax = plt.axes()
    ax.plot(rv_arr1, det_fburn_arr, label='deterministic solution')
    ax.plot(rv_arr1, fburn_arr1, label='ndim = 1')
    ax.plot(rv_arr2, fburn_arr2, label='ndim = 2')
    ax.plot(rv_arr3, fburn_arr3, label='ndim = 3')
    ax.plot(rv_arr3, fburn_arr4, label='ndim = 4')
    ax.plot(rv_arr3, fburn_arr5, label='ndim = 5')
    ax.plot(rv_arr3, fburn_arr_mc_full, label='mc full')
    ax.set_ylabel('Fuelburn, Wf')
    ax.set_xlabel(rv_name)
    ax.legend()
    plt.tight_layout()
    plt.savefig(fname, format='pdf')
