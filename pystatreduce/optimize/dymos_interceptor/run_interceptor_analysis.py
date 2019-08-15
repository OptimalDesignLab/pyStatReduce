################################################################################
# run_interceptor_analysis.py
#
# This file contains the script for evaluating the mean and standard deviation
# for time duration for the supersonic interceptor trajectory analysis using
# reduced stochastic collocation and active subspace method.
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

# Import the OpenMDAo shenanigans
from openmdao.api import IndepVarComp, Problem, Group, NewtonSolver, \
    ScipyIterativeSolver, LinearBlockGS, NonlinearBlockGS, \
    DirectSolver, LinearBlockGS, PetscKSP, SqliteRecorder, ScipyOptimizeDriver

input_dict = {'num_segments': 15,
              'transcription_order' : 3,
              'transcription_type': 'LGR',
              'solve_segments': False,
              'use_for_collocation' : False,
              'n_collocation_samples': 20,
              'use_polynomial_control': False}
systemsize = input_dict['num_segments'] * input_dict['transcription_order']

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


# Generate the joint distibution
mu = np.zeros(systemsize)
density_variation_obj = DensityVariations1976()
std_dev_density = density_variation_obj.get_density_deviations(np.squeeze(nominal_altitude, axis=1))
# print('std_dev_density = ', repr(std_dev_density))
jdist = cp.MvNormal(mu, np.diag(std_dev_density[:-1]))

dymos_obj = DymosInterceptorGlue(systemsize, input_dict)

use_dominant_spaces = True
use_active_subspace = False
if use_dominant_spaces == True:
    dominant_space = DimensionReduction(n_arnoldi_sample=int(sys.argv[1]),
                                        exact_Hessian=False,
                                        sample_radius=1.e-2)
    dominant_space.getDominantDirections(dymos_obj, jdist, max_eigenmodes=10)

    print('eigenvals = ', repr(dominant_space.iso_eigenvals))
    print('eigenvecs = \n', repr(dominant_space.iso_eigenvecs))

    # Save to file:
    fname = os.environ['HOME'] + '/UserApps/pyStatReduce/pystatreduce/optimize/dymos_interceptor/eigenmodes/eigenmodes_' + sys.argv[1] + '_samples_1e_2'
    np.savez(fname, eigenvals=dominant_space.iso_eigenvals, eigenvecs=dominant_space.iso_eigenvecs)
elif use_active_subspace == True:
    dominant_space = ActiveSubspace(dymos_obj,
                                    n_dominant_dimensions=20,
                                    n_monte_carlo_samples=1000,
                                    read_rv_samples=False,
                                    use_svd=True,
                                    use_iso_transformation=True)
    dominant_space.getDominantDirections(dymos_obj, jdist)
    fname = os.environ['HOME'] + '/UserApps/pyStatReduce/pystatreduce/optimize/dymos_interceptor/eigenmodes/eigenmodes_1000_samples_active_subspace'
    np.savez(fname, eigenvals=dominant_space.iso_eigenvals, eigenvecs=dominant_space.iso_eigenvecs)
