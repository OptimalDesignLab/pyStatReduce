################################################################################
# plot_altitude_vs_density.py
#
# The following file plots 1 standard deviation of the density variation w.r.t the altitude
# the altitude. This is a vertical plot, i.e., the altitude is a y axis.
#
################################################################################
import sys
import time
import os

# pyStatReduce specific imports
import numpy as np
import chaospy as cp
import copy
from pystatreduce.new_stochastic_collocation import StochasticCollocation2
from pystatreduce.quantity_of_interest import QuantityOfInterest
from pystatreduce.dimension_reduction import DimensionReduction
from pystatreduce.active_subspace import ActiveSubspace
from pystatreduce.stochastic_arnoldi.arnoldi_sample import ArnoldiSampling
import pystatreduce.examples as examples
from pystatreduce.examples.supersonic_interceptor.interceptor_rdo2 import DymosInterceptorGlue # DymosInterceptorQoI
import pystatreduce.utils as utils
from pystatreduce.examples.supersonic_interceptor.atmosphere.density_variations import DensityVariations1976

#pyoptsparse sepecific imports
from scipy import sparse
import argparse
import pyoptsparse # from pyoptsparse import Optimization, OPT, SNOPT

from mpl_toolkits import mplot3d
import matplotlib
import matplotlib.pyplot as plt
plt.rc('text', usetex=True)

nominal_altitude = np.array([100.        ,   100.        ,   164.0557643 ,   218.11824534,   218.11824534,   594.87366893,  1884.81580942,  2500.06507343,
                             2500.06507343,  3873.1995597 ,  5540.24615145,  5960.08519216,  5960.08519216,  6887.76706542,  8020.52078944,  8337.99144103,
                             8337.99144103,  8804.73183899,  8791.21435824,  8607.8540786 ,  8607.8540786 ,  8063.55369594,  7333.01653081,  7194.34561926,
                             7194.34561926,  7036.90966352,  7100.51166151,  7160.9019215 ,  7160.9019215 ,  7337.32623784,  7621.40509094,  7710.9464599 ,
                             7710.9464599 ,  7914.46736459,  8185.63625057,  8268.88773658,  8268.88773658,  8456.13966968,  8699.65906157,  8771.14526332,
                             8771.14526332,  8913.56513682,  9047.59333244,  9071.30566043,  9071.30566043,  9091.13009621,  9181.87442526,  9267.26464136,
                             9267.26464136,  9758.6687448 , 11245.96312222, 11943.03478097, 11943.03478097, 13766.11714855, 16424.42017061, 17194.76290022,
                             17194.76290022, 18705.88879153, 19906.99495105, 20000.        ])
nominal_density = np.array([1.2134249 , 1.2134249 , 1.20596386, 1.19969442, 1.19969442, 1.1566927 , 1.01831581, 0.9569675 , 0.9569675 , 0.83024452, 0.69412043,
                            0.66270585, 0.66270585, 0.59715196, 0.52396751, 0.5047471 , 0.5047471 , 0.47747268, 0.47824631, 0.48883651, 0.48883651, 0.5213297 ,
                            0.56751045, 0.57661742, 0.57661742, 0.58709374, 0.58284416, 0.57883082, 0.57883082, 0.56722922, 0.54892303, 0.54324711, 0.54324711,
                            0.53051155, 0.51390202, 0.50888425, 0.50888425, 0.49773277, 0.48351195, 0.47939671, 0.47939671, 0.47127899, 0.46373496, 0.46240999,
                            0.46240999, 0.46130454, 0.45627079, 0.45157192, 0.45157192, 0.42524583, 0.35010238, 0.31367084, 0.31367084, 0.235301  , 0.15472919,
                            0.13703077, 0.13703077, 0.10797699, 0.08934021, 0.08802636])

plot_nominal_density = False
if plot_nominal_density:
    matplotlib.rcParams.update({'font.size': 14})
    fname = "nominal_density_vs_altitude.pdf"
    fig = plt.figure("deviations", figsize=(4,6))
    ax = plt.axes()
    # print(altitude_vec.size)
    # print(nominal_density.size)
    s1 = ax.plot(nominal_density, nominal_altitude)#, marker='o' , facecolors='none', edgecolors='b')
    ax.set_xlabel(r'nominal density $(kg/m^3)$')
    ax.set_ylabel('altitude (m)')
    plt.tight_layout()
    plt.show()
    # plt.savefig(fname, format='pdf')

plot_density_deviations = True
if plot_density_deviations:
    altitude_arr = np.array([[  100.        ],
                                 [  100.        ],
                                 [  158.71154193],
                                 [  208.26354802],
                                 [  581.54820633],
                                 [ 1885.33065083],
                                 [ 2508.96598669],
                                 [ 3886.01157759],
                                 [ 5531.48403351],
                                 [ 5938.86520482],
                                 [ 6854.36076941],
                                 [ 8019.53950214],
                                 [ 8362.38092809],
                                 [ 8902.14566076],
                                 [ 9016.16316281],
                                 [ 8875.61192646],
                                 [ 8401.84220327],
                                 [ 7648.20166125],
                                 [ 7461.23794421],
                                 [ 7181.56651798],
                                 [ 7112.68294527],
                                 [ 7157.94912386],
                                 [ 7319.9111466 ],
                                 [ 7610.00221154],
                                 [ 7699.66347111],
                                 [ 7905.03567485],
                                 [ 8180.61093254],
                                 [ 8266.38511845],
                                 [ 8458.44557179],
                                 [ 8704.9284588 ],
                                 [ 8775.95403863],
                                 [ 8916.74233772],
                                 [ 9048.63089773],
                                 [ 9072.17357963],
                                 [ 9091.93587063],
                                 [ 9183.14534263],
                                 [ 9268.69147944],
                                 [ 9760.70248235],
                                 [11248.9293303 ],
                                 [11946.20490623],
                                 [13769.19026677],
                                 [16426.53699354],
                                 [17196.47753757],
                                 [18706.73938156],
                                 [19907.06151334],
                                 [20000.        ]])
    altitude_vec = np.squeeze(altitude_arr, axis=1)
    # Get the standard deviations for the density
    density_variation_obj = DensityVariations1976()
    density_dev_val = density_variation_obj.get_density_deviations(altitude_vec)

    # Plot these two values
    matplotlib.rcParams.update({'font.size': 14})
    fname = "density_deviations_vs_altitude.pdf"
    fig = plt.figure("deviations", figsize=(4.5,6))
    ax = plt.axes()
    s1 = ax.scatter(density_dev_val, altitude_vec, marker='o', facecolors='none', edgecolors='k')
    ax.set_xlabel(r'density standard deviation $(kg/m^3)$')
    ax.set_ylabel('altitude $(m)$')
    plt.tight_layout()
    # plt.show()
    plt.savefig(fname, format='pdf')

plot_mean_plus_std_dev = False
if plot_mean_plus_std_dev:
    # Get the standard deviations for the density
    density_variation_obj = DensityVariations1976()
    density_dev_val = density_variation_obj.get_density_deviations(nominal_altitude)

    mean_plus2 = nominal_density + 2*density_dev_val
    mean_minus2 = nominal_density - 2*density_dev_val

    matplotlib.rcParams.update({'font.size': 14})
    fname = "density_stats_vs_altitude.pdf"
    fig = plt.figure("deviations", figsize=(5,6))
    ax = plt.axes()
    p1 = ax.plot(nominal_density, nominal_altitude, color='k', label='mean')
    p2 = ax.plot(mean_plus2, nominal_altitude, '--', color='k', label=r'$2\sigma$')
    p3 = ax.plot(mean_minus2, nominal_altitude, '--', color='k')
    ax.set_xlabel(r'density $(kg/m^3)$')
    ax.set_ylabel('altitude $(m)$')
    ax.legend()
    plt.tight_layout()
    # plt.show()
    plt.savefig(fname, format='pdf')
