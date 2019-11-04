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

# # Default mean values 6rv case
# mean_Ma = 0.071
# mean_TSFC = 9.80665 * 8.6e-6
# mean_W0 = 10.0
# mean_E = 85.e9
# mean_G = 25.e9
# mean_mrho = 1600
# mean_R = 1800
# mean_load_factor = 1.0
# mean_altitude = 4.57
# # Default standard values
# std_dev_Ma = 0.005
# std_dev_TSFC = 0.00607/3600
# std_dev_W0 = 0.2
# std_dev_mrho = 50
# std_dev_R = 500
# std_dev_load_factor = 0.1
# std_dev_E = 5.e9
# std_dev_G = 1.e9
# std_dev_altitude = 0.5

# Default mean values
mean_Ma = 0.08 # 0.071
mean_TSFC = 9.80665 * 8.6e-6 * 3600
mean_W0 = 16 # 10.0
mean_E = 85.e9
mean_G = 25.e9
mean_mrho = 1600
mean_R = 1800
mean_load_factor = 1.0
mean_altitude = 4.57
# Default standard values
std_dev_Ma = 0.005 # 0.015 # 0.005
std_dev_TSFC = 0.00607# /3600
std_dev_W0 = 0.2
std_dev_mrho = 50
std_dev_R = 300 # 500
std_dev_load_factor = 0.2
std_dev_E = 5.e9
std_dev_G = 1.e9
std_dev_altitude = 0.1 # 0.45

# Try plotting against the dominant directions
# # 6 rv
# iso_eigenvals =  np.array([8.81835838, -8.44152071, -0.12884449, 0.0314637, -0.00256401, 0.00041618])
# iso_eigenvecs = np.array([[ 0.15480831,  0.13542573,  0.29761902, -0.92469618, -0.00719036, -0.11833192],
#                       [-0.04497716, -0.03972589, -0.10979421, -0.00835489,  0.92009768, -0.37107222],
#                       [-0.00298347,  0.0019684,  -0.05265072,  0.10340413, -0.37613963, -0.91926149],
#                       [-0.59581563, -0.58804357,  0.54610604, -0.00617504, -0.00173974, -0.03058643],
#                       [-0.35992377, -0.35200829, -0.7735168,  -0.36608374, -0.10900464,  0.04814032],
#                       [-0.69961501,  0.71440379,  0.00600013, -0.01092308, -0.00157855,  0.0028739 ]])

# 7 rv
iso_eigenvals = np.array([ 12.79883138, -12.25574346, 0.31276045, -0.0273638, -0.00559258,   0.00312678,  0.00124909])
iso_eigenvecs = np.array([[-0.25586111, -0.19581788, -0.93855217, -0.02076572, -0.05598124,  0.10830751,  0.00382753],
                          [ 0.02867164,  0.02320538,  0.0003773,  0.03274935,  0.82238835,  0.53800515,  0.17831052],
                          [ 0.00953314, -0.00599485,  0.12130477, -0.4446585, -0.49059669,  0.71775296,  0.17771529],
                          [ 0.07956942,  0.06463962,  0.00117079,  0.88882824, -0.27944181,  0.33797233,  0.08461909],
                          [ 0.66324669,  0.66791024, -0.31968693, -0.10353199, -0.01985812, -0.02585482, -0.00427982],
                          [ 0.69808369, -0.71467084, -0.04231968, -0.00586013,  0.00266756, -0.00938594, -0.0020593 ],
                          [ 0.00859305,  0.00664458,  0.02031713,  0.00249408,  0.03700269,  0.26204726, -0.96406692]])

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

                    # '7rv_1e_1_2_2' :  {'twist_cp' : np.array([2.794426, 10, 5]),
                    #                   'thickness_cp' : 1.e-3 * np.ones(3),
                    #                   'sweep' : 19.07767381,
                    #                   'alpha' : 3.71407371 },
                    '7rv_1e_1_2_2' :  {'twist_cp' : np.array([2.864292, 10, 5]),
                                       'thickness_cp' : 1.e-3 * np.ones(3),
                                       'sweep' : 19.03869011,
                                       'alpha' : 3.54465185},

                    'deterministic' : {'twist_cp' : np.array([2.61227376, 10, 5]) ,
                                       'thickness_cp' : 1.e-3 * np.ones(3) ,
                                       'sweep' : 18.88466275 ,
                                       'alpha' : 2.1682715 },

                    'deterministic_7rv' : {'twist_cp' : np.array([2.1023522, 10, 5]) ,
                                           'thickness_cp' : np.array([0.001, 0.001, 0.001504379]), # 1.e-3 * np.ones(3) ,
                                           'sweep' : 19.29470279,
                                           'alpha' : 4.25979533},

                    'initial' : {'twist_cp' : np.array([2.5, 2.5, 5.0]),
                                 'thickness_cp' : np.array([0.008, 0.008, 0.008]),
                                 'sweep' : 20.0,
                                 'alpha' : 5.},
                  }

# Declare some of the stuff needed
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
            'mrho' : {'mean' : mean_mrho,
                     'std_dev' : std_dev_mrho},
            'altitude' : {'mean' : mean_altitude,
                          'std_dev' : std_dev_altitude},
           }

input_dict = {'n_twist_cp' : 3,
              'n_thickness_cp' : 3,
              'n_CM' : 3,
              'n_thickness_intersects' : 10,
              'n_constraints' : 1 + 10 + 1 + 3 + 3,
              'ndv' : 3 + 3 + 2,
              'mesh_dict' : mesh_dict,
              'rv_dict' : rv_dict
              }


uq_systemsize = len(rv_dict)
QoI = examples.oas_scaneagle2.OASScanEagleWrapper2(uq_systemsize, input_dict)

if sys.argv[1] == 'generate_data_rvs':
    # Gather all the system arguments
    rv_key = sys.argv[2] # key for deciding which random variable to generate data for
    dv_dict_key = sys.argv[3] # key for picking the design variables from the dictionary
                              # optimal_dv_dict below

    uq_systemsize = len(rv_dict)
    mu, std_dev = utils.get_scaneagle_input_rv_statistics(rv_dict)
    jdist = cp.MvNormal(mu, std_dev)

    # Step 1: Instantiate all objects needed for test
    QoI.p['oas_scaneagle.wing.thickness_cp'] = optimal_dv_dict[dv_dict_key]['thickness_cp']
    QoI.p['oas_scaneagle.wing.twist_cp'] = optimal_dv_dict[dv_dict_key]['twist_cp']
    QoI.p['oas_scaneagle.wing.sweep'] = optimal_dv_dict[dv_dict_key]['sweep']
    QoI.p['oas_scaneagle.alpha'] = optimal_dv_dict[dv_dict_key]['alpha']
    QoI.p.final_setup()



    rdo_factor = float(sys.argv[4])
    npts = 21 # 101 # Number of data points for plotting
    rv_lb = rv_dict[rv_key]['mean'] - rdo_factor * rv_dict[rv_key]['std_dev']
    rv_ub = rv_dict[rv_key]['mean'] + rdo_factor * rv_dict[rv_key]['std_dev']
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

elif sys.argv[1] == 'generate_data_dom_dir':
    dv_dict_key = sys.argv[2] # key for picking the design variables from the dictionary
                              # optimal_dv_dict below

    mu, std_dev = utils.get_scaneagle_input_rv_statistics(rv_dict)
    jdist = cp.MvNormal(mu, std_dev)

    # Step 1: Instantiate all objects needed for test
    QoI.p['oas_scaneagle.wing.thickness_cp'] = optimal_dv_dict[dv_dict_key]['thickness_cp']
    QoI.p['oas_scaneagle.wing.twist_cp'] = optimal_dv_dict[dv_dict_key]['twist_cp']
    QoI.p['oas_scaneagle.wing.sweep'] = optimal_dv_dict[dv_dict_key]['sweep']
    QoI.p['oas_scaneagle.alpha'] = optimal_dv_dict[dv_dict_key]['alpha']
    QoI.p.final_setup()

    # Get the direction number
    dir_no = int(sys.argv[3])
    npts = 11
    rdo_range = 9.0 # 3.0
    sqrt_Sigma = np.sqrt(cp.Cov(jdist))
    v_samples = np.linspace(-rdo_range, rdo_range, num=npts)
    # Create vector for plucking out points in the isoprobabilistic space
    e_i = np.zeros(uq_systemsize)
    e_i[dir_no] = 1.0
    fval_arr = np.zeros(npts)
    for pts in range(0, npts):
        rvs = mu + np.dot(sqrt_Sigma, np.dot(iso_eigenvecs, v_samples[pts] * e_i))
        # Update the random variables
        QoI.update_rv(rvs)
        QoI.p.run_model()
        fval_arr[pts] = QoI.p['oas_scaneagle.AS_point_0.fuelburn']

    # Write to file
    combined_arr = np.array([v_samples, fval_arr])
    fname = './V' + str(dir_no) + '/fburn_vs_V' + str(dir_no) + '_' + dv_dict_key + '.txt'
    np.savetxt(fname, combined_arr)

elif sys.argv[1] == 'plot_rv':

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
    # ax.plot(rv_arr1, fburn_arr1, label='ndim = 1')
    ax.plot(rv_arr2, fburn_arr2, label='ndim = 2')
    ax.plot(rv_arr3, fburn_arr3, label='ndim = 3')
    # ax.plot(rv_arr3, fburn_arr4, label='ndim = 4')
    # ax.plot(rv_arr3, fburn_arr5, label='ndim = 5')
    # ax.plot(rv_arr3, fburn_arr_mc_full, label='mc full')
    ax.set_ylabel('Fuelburn, Wf')
    ax.set_xlabel(rv_name)
    ax.legend()
    plt.tight_layout()
    plt.savefig(fname, format='pdf')

elif sys.argv[1] == 'plot_dom_dir':
    # Get the direction number
    dir_no = sys.argv[2]
    # fname = './V' + dir_no + '/fburn_vs_V' + dir_no + '_' + dv_dict_key + '.txt'
    # Load the deterministic file
    det_fpath = './V' + dir_no + '/fburn_vs_V' + dir_no + '_deterministic_7rv.txt'
    det_combined_arr = np.loadtxt(det_fpath)

    # Load UQ files
    # fpath1 = './V' + dir_no + '/fburn_vs_V' + dir_no + '_6rv_1e_1_1_2.txt'
    # fpath2 = './V' + dir_no + '/fburn_vs_V' + dir_no + '_6rv_1e_1_2_2.txt'
    # fpath3 = './V' + dir_no + '/fburn_vs_V' + dir_no + '_6rv_1e_1_3_2.txt'
    # fpath4 = './V' + dir_no + '/fburn_vs_V' + dir_no + '_6rv_1e_1_4_2.txt'
    # fpath5 = './V' + dir_no + '/fburn_vs_V' + dir_no + '_6rv_1e_1_5_2.txt'
    # fpath_mc = './V' + dir_no + '/fburn_vs_V' + dir_no + '_6rv_mc_full_2.txt'
    fpath2 = './V' + dir_no + '/fburn_vs_V' + dir_no + '_7rv_1e_1_2_2.txt'

    # comb_arr1 = np.loadtxt(fpath1)
    comb_arr2 = np.loadtxt(fpath2)
    # comb_arr3 = np.loadtxt(fpath3)
    # comb_arr4 = np.loadtxt(fpath4)
    # comb_arr5 = np.loadtxt(fpath5)
    # comb_arr_mc = np.loadtxt(fpath_mc)

    # Split the combined array
    # rv_arr1 = comb_arr1[0,:]
    # fburn_arr1 = comb_arr1[1,:]
    rv_arr2 = comb_arr2[0,:]
    fburn_arr2 = comb_arr2[1,:]
    # rv_arr3 = comb_arr3[0,:]
    # fburn_arr3 = comb_arr3[1,:]
    # rv_arr4 = comb_arr4[0,:]
    # fburn_arr4 = comb_arr4[1,:]
    # rv_arr5 = comb_arr5[0,:]
    # fburn_arr5 = comb_arr5[1,:]
    # rv_arr_mc_full = comb_arr_mc[0,:]
    # fburn_arr_mc_full = comb_arr_mc[1,:]

    # Make some assertions
    # np.testing.assert_array_equal(rv_arr1, det_combined_arr[0,:])
    np.testing.assert_array_equal(rv_arr2, det_combined_arr[0,:])
    # np.testing.assert_array_equal(rv_arr3, det_combined_arr[0,:])
    # np.testing.assert_array_equal(rv_arr4, det_combined_arr[0,:])
    # np.testing.assert_array_equal(rv_arr5, det_combined_arr[0,:])
    # np.testing.assert_array_equal(rv_arr_mc_full, det_combined_arr[0,:])
    det_fburn_arr = det_combined_arr[1,:]

    # Actually plot
    fname = './V' + dir_no + '/fburn_vs_V' + dir_no + '.pdf'
    # plt.rc('text', usetex=True)
    matplotlib.rcParams['mathtext.fontset'] = 'cm'
    matplotlib.rcParams.update({'font.size': 14})
    fig = plt.figure(figsize=(5,5))
    ax = plt.axes()
    ax.plot(rv_arr2, det_fburn_arr, label='deterministic solution')
    # ax.plot(rv_arr1, fburn_arr1, label='ndim = 1')
    ax.plot(rv_arr2, fburn_arr2, label='ndim = 2')
    # ax.plot(rv_arr3, fburn_arr3, label='ndim = 3')
    # ax.plot(rv_arr3, fburn_arr4, label='ndim = 4')
    # ax.plot(rv_arr3, fburn_arr5, label='ndim = 5')
    # ax.plot(rv_arr3, fburn_arr_mc_full, label='mc full')
    ax.set_ylabel('Fuelburn, Wf')
    ax.set_xlabel('V' + dir_no)
    ax.legend()
    plt.tight_layout()
    plt.savefig(fname, format='pdf')
