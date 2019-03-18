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

class OASScanEagleWrapper2(QuantityOfInterest):
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
            self.rvs.add_output('CT', val=self.rv_dict['CT']['mean'], units='1/s') # TSFC
            self.p.model.connect('CT', 'oas_scaneagle.CT')

        if 'W0' in self.rv_dict:
            self.rvs.add_output('W0', val=self.rv_dict['W0']['mean'],  units='kg')
            self.p.model.connect('W0', 'oas_scaneagle.W0')

        if 'R' in self.rv_dict:
            self.rvs.add_output('R', val=self.rv_dict['R']['mean'], units='m')
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
            self.rvs.add_output('altitude', val=4.57e3, units='m')
            self.p.model.connect('altitude', 'oas_scaneagle.altitude')

        self.p.setup(check=False)

        # Set up reusable arrays
        self.dJ_ddv = np.zeros(self.input_dict['ndv'], dtype=self.data_type) # Used in eval_ObjGradient_dv
        self.con_arr = np.zeros(self.input_dict['n_constraints'], dtype=self.data_type) # Used in eval_ConstraintQoI
        self.con_jac = np.zeros((self.input_dict['n_constraints'], self.input_dict['ndv']), dtype=self.data_type)

    def eval_QoI(self, mu, xi):
        """
        Computes the Quantity of Interest (in this case the objective function)
        for a given random variable realization.
        """
        rv = mu + xi
        self.update_rv(rv)
        self.p.run_model()
        return self.p['oas_scaneagle.AS_point_0.fuelburn']

    def eval_QoIGradient(self, mu, xi):
        """
        Computes the gradient of the QoI w.r.t the random variables
        """
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

        # OLD CODE
        # deriv = self.p.compute_totals(of=['oas_scaneagle.AS_point_0.fuelburn'],
        #                     wrt=['Mach_number', 'CT', 'W0', 'E', 'G', 'mrho', 'R', 'load_factor'])
        # deriv_arr[0] = deriv['oas_scaneagle.AS_point_0.fuelburn', 'Mach_number']
        # deriv_arr[1] = deriv['oas_scaneagle.AS_point_0.fuelburn', 'CT']
        # deriv_arr[2] = deriv['oas_scaneagle.AS_point_0.fuelburn', 'W0']
        # deriv_arr[3] = deriv['oas_scaneagle.AS_point_0.fuelburn', 'E']
        # deriv_arr[4] = deriv['oas_scaneagle.AS_point_0.fuelburn', 'G']
        # deriv_arr[5] = deriv['oas_scaneagle.AS_point_0.fuelburn', 'mrho']
        # deriv_arr[6] = deriv['oas_scaneagle.AS_point_0.fuelburn', 'R']
        # deriv_arr[7] = deriv['oas_scaneagle.AS_point_0.fuelburn', 'load_factor']

        return deriv_arr

    def eval_ObjGradient_dv(self, mu, xi):
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

        return self.dJ_ddv

    def eval_AllConstraintQoI(self, mu, xi):
        """
        Evaluates ALL the constraint function for a given realization of random variables.
        """
        rv = mu + xi
        self.update_rv(rv)
        self.p.run_model()

        # Since the current stochastic collocation method expects a single array
        # as the functional output, we need to assemble the constraints into a
        # single array and redistribute it in the OUU function
        self.con_arr.fill(0.0) # = np.zeros(self.input_dict['n_constraints'])
        n_thickness_intersects = self.p['oas_scaneagle.AS_point_0.wing_perf.thickness_intersects'].size
        n_CM = self.p['oas_scaneagle.AS_point_0.CM'].size
        self.con_arr[0] = self.p['oas_scaneagle.AS_point_0.wing_perf.failure']
        self.con_arr[1:n_thickness_intersects+1] = self.p['oas_scaneagle.AS_point_0.wing_perf.thickness_intersects']
        self.con_arr[n_thickness_intersects+1] = self.p['oas_scaneagle.AS_point_0.L_equals_W']
        self.con_arr[n_thickness_intersects+2:n_thickness_intersects+2+n_CM] = self.p['oas_scaneagle.AS_point_0.CM']
        self.con_arr[n_thickness_intersects+2+n_CM:] = self.p['oas_scaneagle.wing.twist_cp']

        return self.con_arr

    def eval_confailureQoI(self, mu, xi):
        """
        Evaluates only the failure constraint for a given realization of random variabels.
        """
        rv = mu + xi
        self.update_rv(rv)
        self.p.run_model()

        return self.p['oas_scaneagle.AS_point_0.wing_perf.failure']

    def eval_ConGradient_dv(self, mu, xi):
        """
        Evaluates the gradient of the constraint function with respect to the
        design variables for a given realization of random variabels
        """
        rv = mu + xi
        self.update_rv(rv)
        self.p.run_model()

        # Compute all the derivatives
        # Compute derivatives
        deriv = self.p.compute_totals(of=['oas_scaneagle.AS_point_0.wing_perf.failure',
                                               'oas_scaneagle.AS_point_0.wing_perf.thickness_intersects',
                                               'oas_scaneagle.AS_point_0.L_equals_W',
                                               'oas_scaneagle.AS_point_0.CM'],
                                           wrt=['oas_scaneagle.wing.twist_cp',
                                                'oas_scaneagle.wing.thickness_cp',
                                                'oas_scaneagle.wing.sweep',
                                                'oas_scaneagle.alpha'])
        # In the interest of implicity, I will create a dense constraint jacobian
        # for now and then investigate sparseness. Also, this is a small matrix
        n_twist_cp = self.input_dict['n_twist_cp']
        n_cp = n_twist_cp + self.input_dict['n_thickness_cp']
        n_CM = self.input_dict['n_CM']
        n_thickness_intersects = self.p['oas_scaneagle.AS_point_0.wing_perf.thickness_intersects'].size
        self.con_jac.fill(0.0) # = np.zeros((self.input_dict['n_constraints'], self.input_dict['ndv']))
        # Populate con_jac
        self.con_jac[0,0:n_twist_cp] = deriv['oas_scaneagle.AS_point_0.wing_perf.failure', 'oas_scaneagle.wing.twist_cp']
        self.con_jac[0,n_twist_cp:n_cp] = deriv['oas_scaneagle.AS_point_0.wing_perf.failure', 'oas_scaneagle.wing.thickness_cp']
        self.con_jac[0,n_cp] = deriv['oas_scaneagle.AS_point_0.wing_perf.failure', 'oas_scaneagle.wing.sweep']
        self.con_jac[0,n_cp+1] = deriv['oas_scaneagle.AS_point_0.wing_perf.failure', 'oas_scaneagle.alpha']

        # con_jac[1:n_thickness_intersects+1,0:n_twist_cp] = deriv['oas_scaneagle.AS_point_0.wing_perf.thickness_intersects', 'oas_scaneagle.wing.twist_cp']
        self.con_jac[1:n_thickness_intersects+1,n_twist_cp:n_cp] = deriv['oas_scaneagle.AS_point_0.wing_perf.thickness_intersects', 'oas_scaneagle.wing.thickness_cp']
        # con_jac[1:n_thickness_intersects+1,n_cp] = deriv['oas_scaneagle.AS_point_0.wing_perf.thickness_intersects', 'oas_scaneagle.wing.sweep'][:,0]
        # con_jac[1:n_thickness_intersects+1,n_cp+1] = deriv['oas_scaneagle.AS_point_0.wing_perf.thickness_intersects', 'oas_scaneagle.alpha'][:,0]

        self.con_jac[n_thickness_intersects+1,0:n_twist_cp] = deriv['oas_scaneagle.AS_point_0.L_equals_W', 'oas_scaneagle.wing.twist_cp']
        self.con_jac[n_thickness_intersects+1,n_twist_cp:n_cp] = deriv['oas_scaneagle.AS_point_0.L_equals_W', 'oas_scaneagle.wing.thickness_cp']
        self.con_jac[n_thickness_intersects+1,n_cp] = deriv['oas_scaneagle.AS_point_0.L_equals_W', 'oas_scaneagle.wing.sweep'] # [:,0]
        self.con_jac[n_thickness_intersects+1,n_cp+1] = deriv['oas_scaneagle.AS_point_0.L_equals_W', 'oas_scaneagle.alpha']

        idx = n_thickness_intersects + 2
        self.con_jac[idx:idx+n_CM,0:n_twist_cp] = deriv['oas_scaneagle.AS_point_0.CM', 'oas_scaneagle.wing.twist_cp']
        self.con_jac[idx:idx+n_CM,n_twist_cp:n_cp] = deriv['oas_scaneagle.AS_point_0.CM', 'oas_scaneagle.wing.thickness_cp']
        self.con_jac[idx:idx+n_CM,n_cp] = deriv['oas_scaneagle.AS_point_0.CM', 'oas_scaneagle.wing.sweep'][:,0]
        self.con_jac[idx:idx+n_CM,n_cp+1] = deriv['oas_scaneagle.AS_point_0.CM', 'oas_scaneagle.alpha'][:,0]

        idx = n_thickness_intersects + 2 + n_CM
        self.con_jac[idx:,0:n_twist_cp] = np.eye(self.input_dict['n_twist_cp']) # deriv['oas_scaneagle.wing.twist_cp', 'oas_scaneagle.wing.twist_cp']
        # con_jac[idx:,n_twist_cp:n_cp] = deriv['oas_scaneagle.wing.twist_cp', 'oas_scaneagle.wing.thickness_cp']
        # con_jac[idx:,n_cp] = deriv['oas_scaneagle.wing.twist_cp', 'oas_scaneagle.wing.sweep'][:,0]
        # con_jac[idx:,n_cp+1] = deriv['oas_scaneagle.wing.twist_cp', 'oas_scaneagle.alpha'][:,0]

        return self.con_jac

    def eval_ConFailureGradient_dv(self, mu, xi):
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

    def update_rv(self, rv):

        ctr = 0
        for rvs in self.rv_dict:
            self.p[rvs] = rv[ctr]
            ctr += 1
        # self.p['Mach_number'] = rv[0]
        # self.p['CT'] = rv[1]
        # self.p['W0'] = rv[2]
        # self.p['E'] = rv[3]
        # self.p['G'] = rv[4]
        # self.p['mrho'] = rv[5]
        # self.p['R'] = rv[6]
        # self.p['load_factor'] = rv[7]
