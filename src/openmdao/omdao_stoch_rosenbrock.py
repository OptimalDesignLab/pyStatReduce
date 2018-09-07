# OpenMDAO group for the rosenbrock function
from __future__ import division, print_function
import os
import sys
import errno
sys.path.insert(0, '../../src')

# pyStatReduce specific imports
import numpy as np
import chaospy as cp
from stochastic_collocation import StochasticCollocation
from quantity_of_interest import QuantityOfInterest
from dimension_reduction import DimensionReduction
from stochastic_arnoldi.arnoldi_sample import ArnoldiSampling
import examples

# openMDAO specific imports
from openmdao.api import Problem, pyOptSparseDriver, SqliteRecorder, IndepVarComp, ExplicitComponent, Group
try:
    from openmdao.parallel_api import PETScVector
    vector = PETScVector
except:
    from openmdao.vectors.default_vector import DefaultVector
    vector = DefaultVector

class Rosenbrock_OMDAO(ExplicitComponent):
    """
    Base class that computes the deterministic Rosenbrock function.
    """

    def setup(self):

        # TODO: This setup assumes a 2D Function, figure out to generalize it to
        #       multidimensional Rosenbrock
        self.add_input('mean_xi', val = np.zeros(2))
        self.add_input('xi_perturbation', val = np.zeros(2))
        self.add_output('fval', val=0.0)

    def compute(self, inputs, outputs):

        mean_xi = inputs['mean_xi']
        q = inputs['xi_perturbation']
        rv = mean + q

        outputs['fval'] = sum(100.0*(rv[1:]-rv[:-1]**2.0)**2.0 + (1-rv[:-1])**2.0)

class SCLocations_OMDAO(Group):
    """
    This class spawns the stochastic collocation locations and weights where the
    deterministic function needs to be computed.
    """

    def setup(self):

        # TODO: This setup assumes that you are using 3 collocation points in
        #       every direction, figure out how to generalize it.
        self.add_input('collocation_degree', val=3)
        self.add_input('QoI_dimension', val=2) # TODO: Fix this!!

        # Get the Gauss Hermite quadrature point and quadrature locations
        ref_collocation_pts, ref_collocation_w = np.polynomial.hermite.hermgauss(degree)
        self.add_input('ref_collocation_pts', val=ref_collocation_pts)
        self.add_input('ref_collocation_w', val=ref_collocation_w)

        # Check this input
        self.add_input('joint_distribution')

    def compute(self, inputs, outputs):
        # This implementation assumes a standard tensor product collocation
        # scheme with the same number of collocation points in every dimension.
        QoI_dimension = inputs['QoI_dimension']
        total_collocation_pts = inputs['collocation_degree']**QoI_dimension
        collocation_location = np.zeros([total_collocation_pts, QoI_dimension])

        # Populate the collcoation locations
        idx = 0
        


    def spawn_locations(self, sigma, ref_collocation_pts, ref_collocation_w,
                        colloc_xi_arr, colloc_w_arr, actual_location, idx)
        if idx == colloc_xi_arr.size-1:
            sqrt2 = np.sqrt(2)
            for i in xrange(0, ref_collocation_pts.size):
                colloc_w_arr[idx] = ref_collocation_w[i] # Get the array of all the weights needed
                colloc_xi_arr[idx] = ref_collocation_pts[i] # Get the array of all the locations needed
                actual_location = sqrt2*sigma*colloc_xi_arr
                # fval = QoI_func(x, sqrt2*sigma*colloc_xi_arr)
                # mu_j[:] += np.prod(colloc_w_arr)*fval
            return idx-1
        else:
            for i in xrange(0, ref_collocation_pts.size):
                colloc_xi_arr[idx] = ref_collocation_pts[i]
                colloc_w_arr[idx] = ref_collocation_w[i]
                idx = self.doNormalMean(sigma, mu_j, ref_collocation_pts,
                                        ref_collocation_w, QoI_func,
                                        colloc_xi_arr, colloc_w_arr,
                                        actual_location, idx+1)
            return idx-1
