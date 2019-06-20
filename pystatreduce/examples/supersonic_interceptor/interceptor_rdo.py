from __future__ import division, print_function
import os, sys, errno, copy

# pyStatReduce specific imports
import numpy as np
import chaospy as cp
from pystatreduce.stochastic_collocation import StochasticCollocation
from pystatreduce.quantity_of_interest import QuantityOfInterest
from pystatreduce.dimension_reduction import DimensionReduction
from pystatreduce.stochastic_arnoldi.arnoldi_sample import ArnoldiSampling
import pystatreduce.examples as examples

#pyoptsparse sepecific imports
from scipy import sparse
import argparse
from pyoptsparse import Optimization, OPT, SNOPT

from openmdao.api import Problem, Group, IndepVarComp, pyOptSparseDriver, DirectSolver
from openmdao.utils.assert_utils import assert_rel_error

import dymos as dm
from dymos.examples.min_time_climb.min_time_climb_ode import MinTimeClimbODE
from pystatreduce.examples.supersonic_interceptor.min_time_climb_ode import MinTimeClimbODE

class DymosInterceptorQoI(QuantityOfInterest):
    def __init__(self, systemsize, input_dict, data_type=np.float):
        QuantityOfInterest.__init__(self, systemsize, data_type=data_type)
        self.input_dict = input_dict
        num_segments = input_dict['num_segments']
        transcription_order = input_dict['transcription_order']
        transcription_type = input_dict['transcription_type']
        solve_segments = input_dict['solve_segments']
        self.interceptor_obj = InterceptorWrapper(num_segments=num_segments,
                                                  transcription_order=transcription_order,
                                                  transcription_type=transcription_type,
                                                  solve_segments=solve_segments)

    def eval_QoI(self, mu, xi):
        rv = mu + xi
        self.interceptor_obj.set_random_perturbations(rv)
        self.interceptor_obj.p.run_driver()
        return self.p.get_val('traj.phase0.t_duration')[0]

    def eval_QoIGradient(self, mu, xi):
        assert interceptor_obj.transcription_type == 'LGR'
        rv = mu + xi
        self.interceptor_obj.set_random_perturbations(mu)
        self.interceptor_obj.p.run_driver()
        t_orig = self.p.get_val('traj.phase0.t_duration')[0]

        pert_val = 1.e-6
        pert_arr = np.zeros(mu.size)
        n_nodes = interceptor_obj.num_segments * (interceptor_obj.transcription_order+1)
        dtf_drho_all = np.zeros(1.e-6)

        # This is only for LGR



#------------------------------------------------------------------------------#
#
# The following class actually instantiates the dymos object for analysis
#
#------------------------------------------------------------------------------#

class InterceptorWrapper(object):
    def __init__(self, num_segments=15, transcription_order=3, transcription_type='LGR', solve_segments=False):
        self.num_segments = num_segments
        self.transcription_order = transcription_order
        self.transcription_type = transcription_type
        self.solve_segments = solve_segments

        print('self.transcription_type = ', self.transcription_type)
        print('self.solve_segments = ', self.solve_segments)

        self.p = Problem(model=Group())

        self.p.driver = pyOptSparseDriver()
        self.p.driver.options['optimizer'] = 'SNOPT'
        self.p.driver.options['dynamic_simul_derivs'] = False

        self.p.driver.opt_settings['Major iterations limit'] = 1000
        # self.p.driver.opt_settings['iSumm'] = 6
        self.p.driver.opt_settings['Major feasibility tolerance'] = 1.0E-9
        self.p.driver.opt_settings['Major optimality tolerance'] = 1.0E-9
        self.p.driver.opt_settings['Function precision'] = 1.0E-12
        self.p.driver.opt_settings['Linesearch tolerance'] = 0.1
        self.p.driver.opt_settings['Major step limit'] = 0.5
        self.p.driver.options['print_results'] = False

        # Add an indep_var_comp that will talk to external calls from pyStatReduce
        seed_perturbation = np.zeros(60)
        # seed_perturbation[1] = 1.e-6
        random_perturbations = self.p.model.add_subsystem('random_perturbations', IndepVarComp())
        random_perturbations.add_output('rho_pert', val=seed_perturbation, units='kg/m**3',
                                        desc="perturbations introduced into the density data")

        lgl =  dm.GaussLobatto(num_segments=self.num_segments, order=self.transcription_order, solve_segments=self.solve_segments)
        lgr = dm.Radau(num_segments=self.num_segments,
                       order=self.transcription_order,
                       solve_segments=self.solve_segments)
        rk4 = dm.RungeKutta(num_segments=100)

        traj = dm.Trajectory()
        if self.transcription_type is 'RK4':
            phase = dm.Phase(ode_class=MinTimeClimbODE, transcription=rk4)
        elif self.transcription_type is 'LGL':
            phase = dm.Phase(ode_class=MinTimeClimbODE, transcription=lgl)
        elif self.transcription_type is 'LGR':
            phase = dm.Phase(ode_class=MinTimeClimbODE, transcription=lgr)
        else:
            raise NotImplementedError

        traj.add_phase('phase0', phase)
        self.p.model.add_subsystem('traj', traj)

        phase.set_time_options(fix_initial=True, duration_bounds=(50, 400),
                               duration_ref=100.0)

        phase.set_state_options('r', fix_initial=True, lower=0, upper=1.0E6,
                                ref=1.0E3, defect_ref=1.0E3, units='m',solve_segments=self.solve_segments)

        phase.set_state_options('h', fix_initial=True, lower=0, upper=20000.0,
                                ref=1.0E2, defect_ref=1.0E2, units='m', solve_segments=self.solve_segments)

        phase.set_state_options('v', fix_initial=True, lower=10.0,
                                ref=1.0E2, defect_ref=1.0E2, units='m/s', solve_segments=self.solve_segments)

        phase.set_state_options('gam', fix_initial=True, lower=-1.5, upper=1.5,
                                ref=1.0, defect_ref=1.0, units='rad', solve_segments=self.solve_segments)

        phase.set_state_options('m', fix_initial=True, lower=10.0, upper=1.0E5,
                                ref=1.0E3, defect_ref=1.0E3, solve_segments=self.solve_segments)

        phase.add_polynomial_control('alpha', units='deg', lower=-8., upper=8., order=5)

        # Add the random parameters to dymos
        phase.add_input_parameter('rho_pert', shape=(60,), dynamic=False, units='kg/m**3')

        self.p.model.connect('random_perturbations.rho_pert', 'traj.phase0.input_parameters:rho_pert')

        phase.add_design_parameter('S', val=49.2386, units='m**2', opt=False)
        phase.add_design_parameter('Isp', val=1600.0, units='s', opt=False)
        phase.add_design_parameter('throttle', val=1.0, opt=False)

        phase.add_boundary_constraint('h', loc='final', equals=20000, scaler=1.0E-3, units='m')
        phase.add_boundary_constraint('aero.mach', loc='final', equals=1.0)
        phase.add_boundary_constraint('gam', loc='final', equals=0.0, units='rad')

        phase.add_path_constraint(name='h', lower=100.0, upper=20000, ref=20000)
        phase.add_path_constraint(name='aero.mach', lower=0.1, upper=1.8)

        # Minimize time at the end of the phase
        phase.add_objective('time', loc='final', ref=1.0)

        self.p.model.linear_solver = DirectSolver()

        self.p.setup()

        self.p['traj.phase0.t_initial'] = 0.0
        self.p['traj.phase0.t_duration'] = 300.0

        self.p['traj.phase0.states:r'] = phase.interpolate(ys=[0.0, 111319.54], nodes='state_input')
        self.p['traj.phase0.states:h'] = phase.interpolate(ys=[100.0, 20000.0], nodes='state_input')
        self.p['traj.phase0.states:v'] = phase.interpolate(ys=[135.964, 283.159], nodes='state_input')
        self.p['traj.phase0.states:gam'] = phase.interpolate(ys=[0.0, 0.0], nodes='state_input')
        self.p['traj.phase0.states:m'] = phase.interpolate(ys=[19030.468, 16841.431], nodes='state_input')

        if self.transcription_type is 'RK4' or self.transcription_type is 'LGR':
            self.p['traj.phase0.polynomial_controls:alpha'] = np.array([[4.86918595],
                                                                        [1.30322324],
                                                                        [1.41897019],
                                                                        [1.10227365],
                                                                        [3.58780732],
                                                                        [5.36233472]])
            self.p['traj.phase0.t_duration'] = 346.13171325
        elif self.solve_segments is True:
            self.p['traj.phase0.t_duration'] = 346.13171325

    def set_random_perturbations(self, std_pert):
        expanded_pert = np.expand_dims(std_pert, axis=0)
        self.p.set_val('random_perturbations.rho_pert', std_pert, 'kg/m**3')
