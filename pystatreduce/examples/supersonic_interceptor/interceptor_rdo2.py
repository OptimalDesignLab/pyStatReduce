# This file contains the second implementation of the interceptor problem. This
# Implementation will create a glue that instantiates multiplae instances of the
# an OpenMDAO problem

from __future__ import division, print_function
import os, sys, errno, copy
import warnings

# pyStatReduce specific imports
import numpy as np
import chaospy as cp
from pystatreduce.stochastic_collocation import StochasticCollocation
from pystatreduce.quantity_of_interest import QuantityOfInterest
from pystatreduce.dimension_reduction import DimensionReduction
from pystatreduce.stochastic_arnoldi.arnoldi_sample import ArnoldiSampling
import pystatreduce.utils as utils
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

class DymosInterceptorGlue(QuantityOfInterest):
    """
    This is a part of the glue that connects the OpenMDAO dymos interface with
    pyStatReduce. It creates dymos objects on the fly but does not store them.
    In other words, whenever you need to get information about a trajectory, you
    need to create a dymos object, run the optimization, and then collect the
    necessary information. Therefore one must be careful while creating the
    functions.
    """
    def __init__(self, systemsize, input_dict, data_type=np.float):
        QuantityOfInterest.__init__(self, systemsize, data_type=data_type)

        # Default attributes
        self.input_dict = input_dict
        self._write_files = False
        self._aggregate_solutions = False

        num_segments = input_dict['num_segments']
        transcription_order = input_dict['transcription_order']
        transcription_type = input_dict['transcription_type']
        solve_segments = input_dict['solve_segments']
        use_polynomial_control = input_dict['use_polynomial_control']

        if 'write_files' in input_dict:
            self._write_files = True
            self._ctr1 = 0
            self._target_directory = input_dict['target_output_directory']

        if 'aggregate_solutions' in input_dict:
            self._aggregate_solutions = True
            self._ctr2 = 0
            self.altitude_aggregate = {} # np.zeros(transcription_order*num_segments + 1)
            self.aoa_aggregate = {} # np.zeros(transcription_order*num_segments)


    def eval_QoI(self, mu, xi):
        rv = mu + xi
        interceptor_obj = self.__createInterceptorObj(rv)
        interceptor_obj.p.run_driver()

        if self._write_files:
            fname = self._target_directory + '/sample_' + str(self._ctr1)
            alt_schedule = interceptor_obj.p.get_val('traj.phase0.timeseries.states:h')
            aoa_schedule = interceptor_obj.p.get_val('traj.phase0.timeseries.controls:alpha')
            np.savez(fname, altitude=alt_schedule, alpha=aoa_schedule)
            self._ctr1 += 1

        if self._aggregate_solutions:
            # self.altitude_aggregate[:] += np.squeeze(interceptor_obj.p.get_val('traj.phase0.states:h'), axis=1)
            # self.aoa_aggregate[:] += np.squeeze(interceptor_obj.p.get_val('traj.phase0.controls:alpha'), axis=1)

            self.altitude_aggregate[self._ctr2] = np.squeeze(interceptor_obj.p.get_val('traj.phase0.timeseries.states:h'), axis=1)
            self.aoa_aggregate[self._ctr2] = np.squeeze(interceptor_obj.p.get_val('traj.phase0.timeseries.controls:alpha'), axis=1)
            # print('altitude_aggregate = ', self.altitude_aggregate)
            # print('aoa_aggregate = ', self.aoa_aggregate)
            self._ctr2 += 1

        return interceptor_obj.p.get_val('traj.phase0.t_duration')[0]

    def eval_QoIGradient(self, mu, xi, fd_pert=1.e-2):
        # Check a few importhant things
        if 'write_files' in self.input_dict or 'aggregate_solutions' in self.input_dict:
            if self.input_dict['write_files'] is True or self.input_dict['aggregate_solutions'] is True:
                warnings.warn("WARNING: 'write_files' and/or 'aggregate_solutions' is turned on at every QoI solve using input_dict. Turn it off so as to prevent a possible overwrite of your saved data")

        def func(x):
            return self.eval_QoI(x, np.zeros(np.size(x)))

        rv = mu + xi
        dfdrv = utils.central_fd(func, rv, output_dimensions=1, fd_pert=fd_pert)
        return dfdrv

    def compute_schedule_statistics(self):
        num_segments = self.input_dict['num_segments']
        transcription_order = self.input_dict['transcription_order']
        # mean_altitude = self.altitude_aggregate / self._ctr1
        # mean_alpha = self.aoa_aggregate / self._ctr1
        # variance_alpha =
        mean_altitude = np.zeros(self.altitude_aggregate[0].size) # np.zeros(transcription_order*num_segments + 1)
        var_altitude = np.zeros(mean_altitude.size)
        mean_alpha = np.zeros(self.aoa_aggregate[0].size)
        var_alpha = np.zeros(mean_alpha.size)

        # Get the altitude statistics
        for i in self.altitude_aggregate:
            mean_altitude[:] += self.altitude_aggregate[i]
        mean_altitude[:] = mean_altitude / self._ctr2
        for i in self.altitude_aggregate:
            var_altitude[:] += (self.altitude_aggregate[i] - mean_altitude) ** 2
        var_altitude[:] = var_altitude / (self._ctr2 - 1)

        # Get the control statistics
        for i in self.aoa_aggregate:
            mean_alpha[:] += self.aoa_aggregate[i]
        mean_alpha[:] = mean_alpha / self._ctr2
        for i in self.aoa_aggregate:
            var_alpha[:] += (self.aoa_aggregate[i] - mean_alpha) ** 2
        var_alpha[:] = var_alpha / (self._ctr2 - 1)

        return_dict = {'altitude': {'mean' : mean_altitude,
                                    'variance' : var_altitude,},
                       'alpha': {'mean': mean_alpha,
                                 'variance': var_alpha,},
                      }

        return return_dict

    def get_time_series(self):
        """
        Gets the time series of altitude, angle of atttack, and the time for
        plotting results.
        """
        transcription_order = self.input_dict['transcription_order']
        num_segments = self.input_dict['num_segments']
        rv = np.zeros(transcription_order*num_segments)
        interceptor_obj = self.__createInterceptorObj(rv)
        interceptor_obj.p.run_driver()
        # Get the time series
        aoa_series = np.squeeze(interceptor_obj.p.get_val('traj.phase0.timeseries.controls:alpha'), axis=1)
        alt_series = np.squeeze(interceptor_obj.p.get_val('traj.phase0.timeseries.states:h'), axis=1)
        time_series = np.squeeze(interceptor_obj.p.get_val('traj.phase0.timeseries.time'), axis=1)

        return_dict = { 'aoa' : aoa_series,
                        'altitude' : alt_series,
                        'time' : time_series,
                       }
        # return np.squeeze(interceptor_obj.p.get_val('traj.phase0.timeseries.time'), axis=1)
        return return_dict

    def get_nominal_density(self, units='kg/m**3'):
        """
        Gets the nominal density values at the different quadrature points for
        the supersonic interceptor problem
        """
        transcription_order = self.input_dict['transcription_order']
        num_segments = self.input_dict['num_segments']
        rv = np.zeros(transcription_order*num_segments)
        interceptor_obj = self.__createInterceptorObj(rv)
        interceptor_obj.p.run_driver()
        # print(interceptor_obj.p.get_val('traj.phase0.rhs_all.atmos.rho'))
        rho_arr = interceptor_obj.p.get_val('traj.phase0.rhs_all.atmos.rho', units=units)
        alt_arr = interceptor_obj.p.get_val('traj.phases.phase0.rhs_all.atmos.density.h', units='m')
        schedule_dict = {'altitude' : alt_arr,
                         'density'  : rho_arr,
                         }
        # We need to also supply altitude because the density values extracted
        # at this point have repeated values across different time segments.
        return schedule_dict


    def __createInterceptorObj(self, rv):
        # self.input_dict = input_dict
        num_segments = self.input_dict['num_segments']
        transcription_order = self.input_dict['transcription_order']
        transcription_type = self.input_dict['transcription_type']
        solve_segments = self.input_dict['solve_segments']
        use_polynomial_control = self.input_dict['use_polynomial_control']

        interceptor_obj = InterceptorWrapper(num_segments=num_segments,
                                             transcription_order=transcription_order,
                                             transcription_type=transcription_type,
                                             solve_segments=solve_segments,
                                             use_polynomial_control=use_polynomial_control,
                                             ivc_pert=np.expand_dims(rv, axis=0))
        return interceptor_obj

    # def __getExpandedArray(self, input_array):
    #     num_segments = self.input_dict['num_segments']
    #     transcription_order = self.input_dict['transcription_order']
    #     n_nodes = (transcription_order+1)*num_segments
    #     output_arr = np.zeros(n_nodes)
    #     ctr = 0
    #     segment_id = 0
    #     i = 0
    #     while i < n_nodes:
    #         if i is (segment_id+1)*(transcription_order+1) -1 and i != (n_nodes-1):
    #             output_arr[i] = input_array[ctr]
    #             output_arr[i+1] = input_array[ctr]
    #             i += 2
    #             ctr += 1
    #             segment_id += 1
    #         else:
    #             output_arr[i] = input_array[ctr]
    #             i += 1
    #             ctr += 1
    #     return output_arr

#------------------------------------------------------------------------------#

class InterceptorWrapper(object):
    def __init__(self, num_segments=15, transcription_order=3,
                 transcription_type='LGR', solve_segments=False,
                 ivc_pert=None, use_polynomial_control=True,
                 read_coloring_file=True):

        self.num_segments = num_segments
        self.transcription_order = transcription_order
        self.transcription_type = transcription_type
        self.solve_segments = solve_segments
        self.use_polynomial_control = use_polynomial_control

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
        if read_coloring_file:
            fname = os.environ['HOME'] + '/UserApps/pyStatReduce/pystatreduce/examples/supersonic_interceptor/coloring_files/total_coloring.pkl'
            self.p.driver.use_fixed_coloring(fname)
            # self.p.driver.use_fixed_coloring('/users/pandak/UserApps/pyStatReduce/pystatreduce/examples/supersonic_interceptor/coloring_files/total_coloring.pkl')

        # Add an indep_var_comp that will talk to external calls from pyStatReduce
        if ivc_pert is None:
            seed_perturbation = np.zeros(self.transcription_order*self.num_segments)
        else:
            seed_perturbation = ivc_pert
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

        if self.use_polynomial_control:
            phase.add_polynomial_control('alpha', units='deg', lower=-8., upper=8., order=5)
        else:
            phase.add_control('alpha', units='deg', lower=-8.0, upper=8.0, scaler=1.0,
                              rate_continuity=True, rate_continuity_scaler=100.0,
                              rate2_continuity=False)

        # # Add the random parameters to dymos
        # phase.add_input_parameter('rho_pert', shape=(60,), dynamic=False, units='kg/m**3')

        # Add the density perturbation as a control
        phase.add_control('rho_pert', units='kg/m**3', opt=False)

        self.p.model.connect('random_perturbations.rho_pert', 'traj.phase0.controls:rho_pert')

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

            if self.use_polynomial_control:
                self.p['traj.phase0.polynomial_controls:alpha'] = np.array([[4.86918595],
                                                                            [1.30322324],
                                                                            [1.41897019],
                                                                            [1.10227365],
                                                                            [3.58780732],
                                                                            [5.36233472]])
                self.p['traj.phase0.t_duration'] = 346.13171325
            else:
                # self.p['traj.phase0.controls:alpha'] = phase.interpolate(ys=[0.0, 0.0], nodes='control_input')
                self.p['traj.phase0.controls:alpha'] = np.array([[ 5.28001465],
                                                               [ 3.13975533],
                                                               [ 1.98865951],
                                                               [ 2.05967779],
                                                               [ 2.22378148],
                                                               [ 1.66812216],
                                                               [ 1.30331958],
                                                               [ 0.69713879],
                                                               [ 0.95481437],
                                                               [ 1.30067776],
                                                               [ 1.89992733],
                                                               [ 1.61608848],
                                                               [ 1.25793436],
                                                               [ 0.61321823],
                                                               [ 0.78469243],
                                                               [ 1.09529382],
                                                               [ 1.75985378],
                                                               [ 2.06015107],
                                                               [ 2.00622047],
                                                               [ 1.80482513],
                                                               [ 1.54044227],
                                                               [ 1.46002774],
                                                               [ 1.31279412],
                                                               [ 1.22493615],
                                                               [ 1.22498241],
                                                               [ 1.2379623 ],
                                                               [ 1.24779616],
                                                               [ 1.24895759],
                                                               [ 1.24532979],
                                                               [ 1.22320548],
                                                               [ 1.21206765],
                                                               [ 1.18520735],
                                                               [ 1.15116935],
                                                               [ 1.14112692],
                                                               [ 1.22312977],
                                                               [ 1.67973464],
                                                               [ 1.90722158],
                                                               [ 2.4858537 ],
                                                               [ 3.32375899],
                                                               [ 3.59849829],
                                                               [ 3.9384917 ],
                                                               [ 3.44095692],
                                                               [ 3.04996246],
                                                               [ 1.07581437],
                                                               [-4.76838553]])
        elif self.solve_segments is True:
            self.p['traj.phase0.t_duration'] = 346.13171325

if __name__ == '__main__':
    import time
    input_dict = {'num_segments': 15,
                  'transcription_order' : 3,
                  'transcription_type': 'LGR',
                  'solve_segments': False,
                  'use_polynomial_control': False,
                  # 'aggregate_solutions' : False,
                  # 'write_files': False
                  }
    systemsize = input_dict['num_segments'] * input_dict['transcription_order']

    qoi = DymosInterceptorGlue(systemsize, input_dict)

    eval_QoI = False
    if eval_QoI:
        start_time = time.time()
        dummy_vec = np.zeros(systemsize)
        t_f =   qoi.eval_QoI(dummy_vec, np.zeros(systemsize))
        end_time = time.time()
        # print('altitude_history = \n', repr(qoi.altitude_aggregate[0]))
        # print('\naoa history = \n', repr(qoi.aoa_aggregate[0]))

    eval_QoI_gradient = False
    if eval_QoI_gradient:
        grad_start_time = time.time()
        grad_tf = qoi.eval_QoIGradient(np.zeros(systemsize), np.zeros(systemsize))
        # print('grad_tf = \n', grad_tf)
        grad_end_time = time.time()

        print('qoi evaluation time = ', end_time-start_time)
        print('qoi gradient evaluation time = ', grad_end_time - grad_start_time)

    get_nominal_density = True
    if get_nominal_density:
        density_dict = qoi.get_nominal_density(units='kg/m**3')
        print('nominal density = \n', repr(density_dict['density']))
        print('nominal_altitude = \n', repr(density_dict['altitude']))

    get_time_series = False
    if get_time_series:
        # Get the time series
        series_dict = qoi.get_time_series()
        print('altitude = \n', repr(series_dict['altitude']))
        print('aoa = \n', repr(series_dict['aoa']))
        print('time = \n', repr(series_dict['time']))
