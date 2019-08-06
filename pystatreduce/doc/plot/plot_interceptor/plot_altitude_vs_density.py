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

nominal_altitude = np.array([[  100.        ],
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
altitude_vec = np.squeeze(nominal_altitude, axis=1)
density_variation_obj = DensityVariations1976()
density_dev_val = density_variation_obj.get_density_deviations(altitude_vec)

# Plot these two values
fname = "density_deviations_vs_altitude.pdf"
fig = plt.figure("deviations", figsize=(4,6))
ax = plt.axes()
s1 = ax.scatter(density_dev_val, altitude_vec, marker='o', facecolors='none', edgecolors='b')
# s2 = ax.scatter(altitude_vec, density_dev_val)
ax.set_xlabel('density standard deviation (g/cm^3)')
ax.set_ylabel('altitude (m)')
plt.tight_layout()
# plt.show()
plt.savefig(fname, format='pdf')
