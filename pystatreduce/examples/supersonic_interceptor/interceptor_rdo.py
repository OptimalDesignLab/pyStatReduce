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
# from dymos.examples.min_time_climb.min_time_climb_ode import MinTimeClimbODE
from pystatreduce.examples.supersonic_interceptor.min_time_climb_ode import MinTimeClimbODE
import gc

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
        # expanded_rv = self.__setNodalEntries(rv)
        # self.interceptor_obj.set_random_perturbations(expanded_rv)
        self.update_rv(rv)
        self.interceptor_obj.p.run_driver()
        return self.interceptor_obj.p.get_val('traj.phase0.t_duration')[0]

    def eval_QoIGradient(self, mu, xi):
        assert self.interceptor_obj.transcription_type == 'LGR', 'The method is only implemented for LGR transcription.'
        rv = mu + xi
        # expanded_rv = self.__setNodalEntries(rv)
        # self.interceptor_obj.set_random_perturbations(expanded_rv)
        self.update_rv(rv)
        self.interceptor_obj.p.run_driver()
        t_orig = self.interceptor_obj.p.get_val('traj.phase0.t_duration')[0]

        transcription_order = self.input_dict['transcription_order']
        pert_val = 1.e-6
        pert_arr = np.zeros(rv.size)
        n_nodes = self.interceptor_obj.num_segments * (transcription_order+1)
        # dtf_drho_all = np.zeros(n_nodes)
        dtf_drho = np.zeros(rv.size)

        # This is only valid for LGR
        for i in range(rv.size):
            pert_arr[i] += pert_val
            self.update_rv(pert_arr)
            self.interceptor_obj.p.run_driver()
            t_pert = self.interceptor_obj.p.get_val('traj.phase0.t_duration')[0]
            dtf_drho[i] = (t_pert - t_orig) / pert_val
            pert_arr[i] -= pert_val

        # # This is only for LGR
        # segment_id = 0
        # i = 0
        # while i < n_nodes:
        #     print('i = ', i)
        #     if i is (segment_id+1)*(transcription_order+1) -1 and i != (n_nodes-1):
        #         print('     i = ', i)
        #         pert_arr[i] += pert_val
        #         pert_arr[i+1] += pert_val
        #         self.interceptor_obj.set_random_perturbations(pert_arr)
        #         t_pert = self.evaluate_time(alpha)
        #         # print('t_pert = ', t_pert)
        #         dtf_drho_all[i] = (t_pert - t_orig) / pert_val
        #         dtf_drho_all[i+1] = dtf_drho_all[i]
        #         segment_id += 1
        #         pert_arr[i] -= pert_val
        #         pert_arr[i+1] -= pert_val
        #         i +=2
        #     else:
        #         pert_arr[i] += pert_val
        #         self.interceptor_obj.set_random_perturbations(pert_arr)
        #         t_pert = self.evaluate_time(alpha)
        #         # print('t_pert = ', t_pert)
        #         dtf_drho_all[i] = (t_pert - t_orig) / pert_val
        #         pert_arr[i] -= pert_val
        #         i += 1

        # # Now that we have to get rid of the duplicates in dtf_drho_all
        # dtf_drho = self.__getUniqueNodalEntries(dtf_drho_all)
        gc.collect()
        return dtf_drho

    def __getUniqueNodalEntries(self, input_array):
        output_list = []
        for x in input_array:
            if x not in output_list:
                output_list.append(x)
            elif x != output_list[-1]:
                # Ensures if two distinct points have the same gradient values,
                # we still capture the values correctly
                output_list.append(x)

        return np.asarray(output_list)

    def __setNodalEntries(self, input_array):
        num_segments = self.input_dict['num_segments']
        transcription_order = self.input_dict['transcription_order']
        n_nodes = (transcription_order+1)*num_segments
        output_arr = np.zeros(n_nodes)
        ctr = 0
        segment_id = 0
        i = 0
        while i < n_nodes:
            if i is (segment_id+1)*(transcription_order+1) -1 and i != (n_nodes-1):
                output_arr[i] = input_array[ctr]
                output_arr[i+1] = input_array[ctr]
                i += 2
                ctr += 1
                segment_id += 1
            else:
                output_arr[i] = input_array[ctr]
                i += 1
                ctr += 1

        return output_arr

    def update_rv(self, rv):
        expanded_rv = self.__setNodalEntries(rv)
        # print('expanded_rv = ', expanded_rv)
        self.interceptor_obj.set_random_perturbations(expanded_rv)

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
        self.p.driver.options['dynamic_simul_derivs'] = True

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


if __name__ == '__main__':

    systemsize = 45 + 1
    input_dict = {'num_segments': 15,
                  'transcription_order' : 3,
                  'transcription_type': 'LGR',
                  'solve_segments': False}
    dymos_obj = DymosInterceptorQoI(systemsize, input_dict)

    # t_f = dymos_obj.eval_QoI(np.zeros(systemsize), np.zeros(systemsize))
    # print('t_f = ', t_f)

    grad_tf = dymos_obj.eval_QoIGradient(np.zeros(systemsize), np.zeros(systemsize))
    print('grad_tf = ', grad_tf)
