# prototype object for handling multiple QoI for a given OpenMDAO problem
from __future__ import division, print_function
import os, sys, errno, copy

# pyStatReduce specific imports
import numpy as np
import chaospy as cp
from pystatreduce.stochastic_collocation import StochasticCollocation
from pystatreduce.quantity_of_interest import QuantityOfInterest
from pystatreduce.dimension_reduction import DimensionReduction
from pystatreduce.stochastic_arnoldi.arnoldi_sample import ArnoldiSampling
from pystatreduce.examples.oas_scaneagle_group import OASScanEagle
import pystatreduce.examples as examples

#pyoptsparse sepecific imports
from scipy import sparse
import argparse
from pyoptsparse import Optimization, OPT, SNOPT

# Import the OpenMDAo shenanigans
from openmdao.api import IndepVarComp, Problem, Group, NewtonSolver, \
    ScipyIterativeSolver, LinearBlockGS, NonlinearBlockGS, \
    DirectSolver, LinearBlockGS, PetscKSP, SqliteRecorder, ScipyOptimizeDriver

from openaerostruct.geometry.utils import generate_mesh
from openaerostruct.integration.aerostruct_groups import AerostructGeometry, AerostructPoint
from openmdao.api import IndepVarComp, Problem, SqliteRecorder

class OASScanEagleWrapper(object):
    def __init__(self, systemsize, input_dict, data_type=np.float):
        QuantityOfInterest.__init__(self, systemsize, data_type=data_type)
        self.input_dict = input_dict
        self.rv_dict = self.input_dict['rv_dict']

        self.p = Problem()
        self.rvs = self.p.model.add_subsystem('random_variables', IndepVarComp(), promotes_outputs=['*'])
        self.p.model.add_subsystem('oas_scaneagle',
                                   OASScanEagle(mesh_dict=self.input_dict['mesh_dict'],
                                                rv_dict=self.rv_dict))

        # Declare rvs units to ensure type stability
        if 'Mach_number' in self.rv_dict:
            self.rvs.add_output('Mach_number', val=self.rv_dict['Mach_number']['mean'])
            self.p.model.connect('Mach_number', 'oas_scaneagle.Mach_number')

        if 'CT' in self.rv_dict:
            # self.rvs.add_output('CT', val=self.rv_dict['CT']['mean'], units='1/s') # TSFC
            self.rvs.add_output('CT', val=self.rv_dict['CT']['mean'], units='1/h') # TSFC
            self.p.model.connect('CT', 'oas_scaneagle.CT')

        if 'W0' in self.rv_dict:
            self.rvs.add_output('W0', val=self.rv_dict['W0']['mean'],  units='kg')
            self.p.model.connect('W0', 'oas_scaneagle.W0')

        if 'R' in self.rv_dict:
            self.rvs.add_output('R', val=self.rv_dict['R']['mean'], units='km')
            self.p.model.connect('R', 'oas_scaneagle.R')

        if 'load_factor' in self.rv_dict:
            self.rvs.add_output('load_factor', val=self.rv_dict['load_factor']['mean'])
            self.p.model.connect('load_factor', 'oas_scaneagle.load_factor')
            self.p.model.connect('load_factor', 'oas_scaneagle.AS_point_0.coupled.wing.load_factor')

        if 'E' in self.rv_dict:
            self.rvs.add_output('E', val=self.rv_dict['E']['mean'], units='N/m**2')
            self.p.model.connect('E', 'oas_scaneagle.wing.struct_setup.assembly.E')
            self.p.model.connect('E', 'oas_scaneagle.AS_point_0.wing_perf.struct_funcs.vonmises.E')

        if 'G' in self.rv_dict:
            self.rvs.add_output('G', val=self.rv_dict['G']['mean'], units='N/m**2')
            self.p.model.connect('G', 'oas_scaneagle.wing.struct_setup.assembly.G')
            self.p.model.connect('G', 'oas_scaneagle.AS_point_0.wing_perf.struct_funcs.vonmises.G')

        if 'mrho' in self.rv_dict:
            self.rvs.add_output('mrho', val=self.rv_dict['mrho']['mean'], units='kg/m**3')
            self.p.model.connect('mrho', 'oas_scaneagle.wing.struct_setup.structural_weight.mrho')

        if 'altitude' in self.rv_dict:
            self.rvs.add_output('altitude', val=self.rv_dict['altitude']['mean'], units='km')
            self.p.model.connect('altitude', 'oas_scaneagle.altitude')

        self.p.setup(check=False)

        # Set up reusable arrays
        self.dJ_ddv = np.zeros(self.input_dict['ndv'], dtype=self.data_type) # Used in eval_ObjGradient_dv
        self.con_arr = np.zeros(self.input_dict['n_constraints'], dtype=self.data_type) # Used in eval_ConstraintQoI
        self.con_jac = np.zeros((self.input_dict['n_constraints'], self.input_dict['ndv']), dtype=self.data_type)

    def update_rv(self, rv):
        ctr = 0
        for rvs in self.rv_dict:
            self.p[rvs] = rv[ctr]
            ctr += 1

#-------------------------------------------------------------------------------

class Fuelburn(QuantityOfInterest):
    def __init__(self, systemsize, oas_object, data_type=np.float):
        QuantityOfInterest.__init__(self, systemsize, data_type=data_type)
        self.data_type = data_type
        self.p = oas_object.p
        self.rvs = oas_object.rvs
        self.update_rv = oas_object.update_rv
        self.input_dict = oas_object.input_dict
        self.rv_dict = self.input_dict['rv_dict']
        self.dJ_ddv =  np.zeros(oas_object.input_dict['ndv'], dtype=self.data_type)

    def eval_QoI(self, mu, xi):
        rv = mu + xi
        self.update_rv(rv)
        self.p.run_model()
        return self.p['oas_scaneagle.AS_point_0.fuelburn']

    def eval_QoIGradient(self, mu, xi):
        rv = mu + xi
        deriv_arr = np.zeros(self.systemsize, dtype=self.data_type)
        self.update_rv(rv)
        self.p.run_model()
        rv_name_list = list(self.rv_dict.keys())
        deriv = self.p.compute_totals(of=['oas_scaneagle.AS_point_0.fuelburn'],
                                      wrt=rv_name_list)
        # Populate deriv_arr
        ctr = 0
        for rvs in self.rv_dict:
            deriv_arr[ctr] = deriv['oas_scaneagle.AS_point_0.fuelburn', rvs]
            ctr += 1

        return deriv_arr

    def eval_QoIGradient_dv(self, mu, xi):
        """
        Computes the gradient of the QoI (in this case the objective function)
        w.r.t the design variables for a given set of random variable
        realizations. The design variables for this implementation are NOT the
        random variables.
        """
        rv = mu + xi
        self.update_rv(rv)
        self.p.run_model()
        deriv = self.p.compute_totals(of=['oas_scaneagle.AS_point_0.fuelburn'],
                                      wrt=['oas_scaneagle.wing.twist_cp',
                                           'oas_scaneagle.wing.thickness_cp',
                                           'oas_scaneagle.wing.sweep',
                                           'oas_scaneagle.alpha'])

        self.dJ_ddv.fill(0.0)
        # dJ_ddv = np.zeros(self.input_dict['ndv'])
        n_twist_cp = self.input_dict['n_twist_cp']
        n_thickness_cp = self.input_dict['n_thickness_cp']
        n_cp = n_twist_cp + n_thickness_cp
        self.dJ_ddv[0:n_twist_cp] = deriv['oas_scaneagle.AS_point_0.fuelburn', 'oas_scaneagle.wing.twist_cp']
        self.dJ_ddv[n_twist_cp:n_cp] = deriv['oas_scaneagle.AS_point_0.fuelburn', 'oas_scaneagle.wing.thickness_cp']
        self.dJ_ddv[n_cp:n_cp+1] = deriv['oas_scaneagle.AS_point_0.fuelburn', 'oas_scaneagle.wing.sweep']
        self.dJ_ddv[n_cp+1:n_cp+2] = deriv['oas_scaneagle.AS_point_0.fuelburn', 'oas_scaneagle.alpha']

        return dJ_ddv

#-------------------------------------------------------------------------------

class StressConstraint(QuantityOfInterest):
    def __init__(self, systemsize, oas_object, data_type=np.float):
        QuantityOfInterest.__init__(self, systemsize, data_type=data_type)
        self.data_type = data_type
        self.p = oas_object.p
        self.rvs = oas_object.rvs
        self.rv_dict = oas_object.input_dict['rv_dict']
        self.update_rv = oas_object.update_rv

    def eval_QoI(self, mu, xi):
        rv = mu + xi
        self.update_rv(rv)
        self.p.run_model()
        return self.p['oas_scaneagle.AS_point_0.wing_perf.failure']

    def eval_QoIGradient(self, mu, xi):
        rv = mu + xi
        deriv_arr = np.zeros(self.systemsize, dtype=self.data_type)
        self.update_rv(rv)
        self.p.run_model()
        rv_name_list = list(self.rv_dict.keys())
        deriv = self.p.compute_totals(of=['oas_scaneagle.AS_point_0.wing_perf.failure'],
                                      wrt=rv_name_list)
        # Populate deriv_arr
        ctr = 0
        for rvs in self.rv_dict:
            deriv_arr[ctr] = deriv['oas_scaneagle.AS_point_0.wing_perf.failure', rvs]
            ctr += 1


        # deriv = self.p.compute_totals(of=['oas_scaneagle.AS_point_0.wing_perf.failure'],
        #                     wrt=['Mach_number', 'CT', 'W0', 'E', 'G', 'mrho'])
        # deriv_arr[0] = deriv['oas_scaneagle.AS_point_0.wing_perf.failure', 'Mach_number']
        # deriv_arr[1] = deriv['oas_scaneagle.AS_point_0.wing_perf.failure', 'CT']
        # deriv_arr[2] = deriv['oas_scaneagle.AS_point_0.wing_perf.failure', 'W0']
        # deriv_arr[3] = deriv['oas_scaneagle.AS_point_0.wing_perf.failure', 'E']
        # deriv_arr[4] = deriv['oas_scaneagle.AS_point_0.wing_perf.failure', 'G']
        # deriv_arr[5] = deriv['oas_scaneagle.AS_point_0.wing_perf.failure', 'mrho']
        return deriv_arr

    def eval_QoIGradient_dv(self, mu, xi):
        rv = mu + xi
        self.update_rv(rv)
        self.p.run_model()
        deriv = self.p.compute_totals(of=['oas_scaneagle.AS_point_0.wing_perf.failure'],
                                      wrt=['oas_scaneagle.wing.twist_cp',
                                           'oas_scaneagle.wing.thickness_cp',
                                           'oas_scaneagle.wing.sweep',
                                           'oas_scaneagle.alpha'])
        n_twist_cp = self.input_dict['n_twist_cp']
        n_cp = n_twist_cp + self.input_dict['n_thickness_cp']
        dcon_failure = np.zeros(self.input_dict['ndv'])
        dcon_failure[0:n_twist_cp] = deriv['oas_scaneagle.AS_point_0.wing_perf.failure', 'oas_scaneagle.wing.twist_cp']
        dcon_failure[n_twist_cp:n_cp] = deriv['oas_scaneagle.AS_point_0.wing_perf.failure', 'oas_scaneagle.wing.thickness_cp']
        dcon_failure[n_cp] = deriv['oas_scaneagle.AS_point_0.wing_perf.failure', 'oas_scaneagle.wing.sweep']
        dcon_failure[n_cp+1] = deriv['oas_scaneagle.AS_point_0.wing_perf.failure', 'oas_scaneagle.alpha']

        return dcon_failure

#-------------------------------------------------------------------------------

class LiftConstraint(QuantityOfInterest):
    def __init__(self, systemsize, oas_object, data_type=np.float):
        QuantityOfInterest.__init__(self, systemsize, data_type=data_type)
        self.data_type = data_type
        self.p = oas_object.p
        self.rvs = oas_object.rvs
        self.rv_dict = oas_object.input_dict['rv_dict']
        self.update_rv = oas_object.update_rv

    def eval_QoI(self, mu, xi):
        rv = mu + xi
        self.update_rv(rv)
        self.p.run_model()
        return self.p['oas_scaneagle.AS_point_0.L_equals_W']

    def eval_QoIGradient(self, mu, xi):
        rv = mu + xi
        deriv_arr = np.zeros(self.systemsize, dtype=self.data_type)
        self.update_rv(rv)
        self.p.run_model()
        rv_name_list = list(self.rv_dict.keys())
        deriv = self.p.compute_totals(of=['oas_scaneagle.AS_point_0.L_equals_W'],
                                      wrt=rv_name_list)
        # Populate deriv_arr
        ctr = 0
        for rvs in self.rv_dict:
            deriv_arr[ctr] = deriv['oas_scaneagle.AS_point_0.L_equals_W', rvs]
            ctr += 1
        # deriv = self.p.compute_totals(of=['oas_scaneagle.AS_point_0.L_equals_W'],
        #                     wrt=['Mach_number', 'CT', 'W0', 'E', 'G', 'mrho'])
        # deriv_arr[0] = deriv['oas_scaneagle.AS_point_0.L_equals_W', 'Mach_number']
        # deriv_arr[1] = deriv['oas_scaneagle.AS_point_0.L_equals_W', 'CT']
        # deriv_arr[2] = deriv['oas_scaneagle.AS_point_0.L_equals_W', 'W0']
        # deriv_arr[3] = deriv['oas_scaneagle.AS_point_0.L_equals_W', 'E']
        # deriv_arr[4] = deriv['oas_scaneagle.AS_point_0.L_equals_W', 'G']
        # deriv_arr[5] = deriv['oas_scaneagle.AS_point_0.L_equals_W', 'mrho']

        return deriv_arr

    def eval_QoIGradient_dv(self, mu, xi):
        rv = mu + xi
        self.update_rv(rv)
        self.p.run_model()
        deriv = self.p.compute_totals(of=['oas_scaneagle.AS_point_0.L_equals_W'],
                                      wrt=['oas_scaneagle.wing.twist_cp',
                                           'oas_scaneagle.wing.thickness_cp',
                                           'oas_scaneagle.wing.sweep',
                                           'oas_scaneagle.alpha'])
        dcon_lift = np.zeros(self.input_dict['ndv'])
        dcon_lift[0:n_twist_cp] = deriv['oas_scaneagle.AS_point_0.wing_perf.failure', 'oas_scaneagle.wing.twist_cp'][0]
        dcon_lift[n_twist_cp:n_cp] = deriv['oas_scaneagle.AS_point_0.wing_perf.failure', 'oas_scaneagle.wing.thickness_cp'][0]
        dcon_lift[n_cp] = deriv['oas_scaneagle.AS_point_0.wing_perf.failure', 'oas_scaneagle.wing.sweep'][0,0]
        dcon_lift[n_cp+1] = deriv['oas_scaneagle.AS_point_0.wing_perf.failure', 'oas_scaneagle.alpha'][0,0]

        return dcon_lift
#-------------------------------------------------------------------------------

class MomentConstraint(QuantityOfInterest):
    def __init__(self, systemsize, oas_object, data_type=np.float):
        QuantityOfInterest.__init__(self, systemsize, data_type=data_type)
        self.data_type = data_type
        self.p = oas_object.p
        self.rvs = oas_object.rvs
        self.rv_dict = oas_object.input_dict['rv_dict']
        self.update_rv = oas_object.update_rv

    def eval_QoI(self, mu, xi):
        rv = mu + xi
        self.update_rv(rv)
        self.p.run_model()
        return self.p['oas_scaneagle.AS_point_0.CM']

    def eval_QoIGradient(self, mu, xi):
        rv = mu + xi
        deriv_arr = np.zeros(self.systemsize, dtype=self.data_type)
        self.update_rv(rv)
        self.p.run_model()
        rv_name_list = list(self.rv_dict.keys())
        deriv = self.p.compute_totals(of=['oas_scaneagle.AS_point_0.CM'],
                                      wrt=rv_name_list)
        # Populate deriv_arr
        ctr = 0
        for rvs in self.rv_dict:
            deriv_arr[ctr] = deriv['oas_scaneagle.AS_point_0.CM', rvs][1,0]
            ctr += 1
        # deriv = self.p.compute_totals(of=['oas_scaneagle.AS_point_0.CM'],
        #                     wrt=['Mach_number', 'CT', 'W0', 'E', 'G', 'mrho'])
        # # We will only consider the longitudnal moment for this derivative
        # deriv_arr[0] = deriv['oas_scaneagle.AS_point_0.CM', 'Mach_number'][1,0]
        # deriv_arr[1] = deriv['oas_scaneagle.AS_point_0.CM', 'CT'][1,0]
        # deriv_arr[2] = deriv['oas_scaneagle.AS_point_0.CM', 'W0'][1,0]
        # deriv_arr[3] = deriv['oas_scaneagle.AS_point_0.CM', 'E'][1,0]
        # deriv_arr[4] = deriv['oas_scaneagle.AS_point_0.CM', 'G'][1,0]
        # deriv_arr[5] = deriv['oas_scaneagle.AS_point_0.CM', 'mrho'][1,0]

        return deriv_arr

    def eval_QoIGradient_dv(self, mu, xi):
        rv = mu + xi
        self.update_rv(rv)
        self.p.run_model()
        deriv = self.p.compute_totals(of=['oas_scaneagle.AS_point_0.CM'],
                                      wrt=['oas_scaneagle.wing.twist_cp',
                                           'oas_scaneagle.wing.thickness_cp',
                                           'oas_scaneagle.wing.sweep',
                                           'oas_scaneagle.alpha'])
        dcon_CM = np.zeros(self.input_dict['ndv'])
        dcon_CM[0:n_twist_cp] = deriv['oas_scaneagle.AS_point_0.CM', 'oas_scaneagle.wing.twist_cp'][0]
        dcon_CM[n_twist_cp:n_cp] = deriv['oas_scaneagle.AS_point_0.CM', 'oas_scaneagle.wing.thickness_cp'][0]
        dcon_CM[n_cp] = deriv['oas_scaneagle.AS_point_0.CM', 'oas_scaneagle.wing.sweep'][0,0]
        dcon_CM[n_cp+1] = deriv['oas_scaneagle.AS_point_0.CM', 'oas_scaneagle.alpha'][0,0]

        return dcon_CM

class ThicknessIntersectsConstraint(QuantityOfInterest):
    def __init__(self, systemsize, oas_object, data_type=np.float):
        QuantityOfInterest.__init__(self, systemsize, data_type=data_type)
        self.data_type = data_type
        self.p = oas_object.p
        self.rvs = oas_object.rvs
        self.rv_dict = oas_object.input_dict['rv_dict']
        self.update_rv = oas_object.update_rv

    def eval_QoI(self, mu, xi):
        rv = mu + xi
        self.update_rv(rv)
        self.p.run_model()
        return self.p['oas_scaneagle.AS_point_0.wing_perf.thickness_intersects']

    def eval_QoIGradient_dv(self, mu, xi):
        rv = mu + xi
        self.update_rv(rv)
        self.p.run_model()
        deriv = self.p.compute_totals(of=['oas_scaneagle.AS_point_0.wing_perf.thickness_intersects'],
                            wrt=['oas_scaneagle.wing.thickness_cp'])
        return deriv['oas_scaneagle.AS_point_0.wing_perf.thickness_intersects', 'oas_scaneagle.wing.thickness_cp']
