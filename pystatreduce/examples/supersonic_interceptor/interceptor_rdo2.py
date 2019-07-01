# This file contains the second implementation of the interceptor problem. This
# Implementation will create a glue that instantiates multiplae instances of the
# an OpenMDAO problem

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

class DymosInterceptorGlue(QuantityOfInterest):
    def __init__(self, systemsize, input_dict, data_type=np.float):
        QuantityOfInterest.__init__(self, systemsize, data_type=data_type)
        self.input_dict = input_dict
        num_segments = input_dict['num_segments']
        transcription_order = input_dict['transcription_order']
        transcription_type = input_dict['transcription_type']
        solve_segments = input_dict['solve_segments']
        use_polynomial_control = input_dict['use_polynomial_control']

        use_for_collocation = input_dict['use_for_collocation']
        if use_for_collocation:
            n_qoi_samples = input_dict['n_collocation_samples']
            self.collocation_samples = {}
            for i in range(n_qoi_samples):
                self.collocation_samples[i] = InterceptorWrapper(num_segments=num_segments,
                                                          transcription_order=transcription_order,
                                                          transcription_type=transcription_type,
                                                          solve_segments=solve_segments,
                                                          use_polynomial_control=use_polynomial_control)

    def eval_QoI(self, mu, xi):
        rv = mu + xi
        interceptor_obj = self.__createInterceptorObj(rv)
        interceptor_obj.p.run_driver()

        return interceptor_obj.p.get_val('traj.phase0.t_duration')[0]

    """
    def eval_QoIGradient_forward(self, mu, xi, fd_pert=1.e-6):
        print('fd_pert = ', fd_pert)
        rv = mu + xi
        baseline_obj = self.__createInterceptorObj(rv)
        baseline_obj.p.run_driver()
        t_orig = baseline_obj.p.get_val('traj.phase0.t_duration')[0]
        print('t_orig = ', t_orig)
        dtf_drho = np.zeros(rv.size)

        for i in range(rv.size):
            print('\ni = ', i)
            rv[i] += fd_pert
            # print('rv = ', rv)
            pert_obj = self.__createInterceptorObj(rv)
            pert_obj.p.run_driver()
            t_pert = pert_obj.p.get_val('traj.phase0.t_duration')[0]
            print('t_pert = ', t_pert)
            dtf_drho[i] = (t_pert - t_orig) / fd_pert
            print('dtf_drho[i] = ', dtf_drho[i])
            rv[i] -= fd_pert

        return dtf_drho
    """

    def eval_QoIGradient(self, mu, xi, fd_pert=1.e-2):
        print('fd_pert = ', fd_pert)
        rv = mu + xi
        temp_arr1 = mu + xi # np.zeros(rv.size)
        temp_arr2 = mu + xi
        dt_fdrho = np.zeros(rv.size)
        for i in range(rv.size):
            temp_arr1[i] += fd_pert
            temp_arr2[i] -= fd_pert
            obj1 = self.__createInterceptorObj(temp_arr1)
            obj2 = self.__createInterceptorObj(temp_arr2)
            obj1.p.run_driver()
            obj2.p.run_driver()
            t_1 = obj1.p.get_val('traj.phase0.t_duration')[0]
            t_2 = obj2.p.get_val('traj.phase0.t_duration')[0]
            # print('t_1 = ', t_1)
            # print('t_2 = ', t_2)
            dt_fdrho[i] = (t_1 - t_2) / (2*fd_pert)
            # print('dt_fdrho[i] = ', dt_fdrho[i])
            temp_arr1[i] -= fd_pert
            temp_arr2[i] += fd_pert

        return dt_fdrho

    def evaluateSCQoIs(self, QoI_dict, QoI_dict_key):
        # Evaluates the QoI for stochastic collocation
        target_dict = QoI_dict[QoI_dict_key]
        for i in range(self.input_dict['n_collocation_samples']):
            self.update_rv(q_val, i)
            self.collocation_samples[i].p.run_driver()
            target_dict['fval'][i] = self.collocation_samples[i].p.get_val('traj.phase0.t_duration')[0]

    def update_rv(self, rv_val, collocation_id):
        expanded_rv = self.__setNodalEntries(rv_val)

    def __createInterceptorObj(self, rv):
        # self.input_dict = input_dict
        num_segments = self.input_dict['num_segments']
        transcription_order = self.input_dict['transcription_order']
        transcription_type = self.input_dict['transcription_type']
        solve_segments = self.input_dict['solve_segments']
        use_polynomial_control = self.input_dict['use_polynomial_control']

        # expanded_arr = self.__getExpandedArray(rv)
        # print('expanded_arr = \n', expanded_arr)
        interceptor_obj = InterceptorWrapper(num_segments=num_segments,
                                             transcription_order=transcription_order,
                                             transcription_type=transcription_type,
                                             solve_segments=solve_segments,
                                             use_polynomial_control=use_polynomial_control,
                                             ivc_pert=np.expand_dims(rv, axis=0))
        return interceptor_obj

    def __getExpandedArray(self, input_array):
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

    input_dict = {'num_segments': 15,
                  'transcription_order' : 3,
                  'transcription_type': 'LGR',
                  'solve_segments': False,
                  'use_for_collocation' : False,
                  'n_collocation_samples': 20,
                  'use_polynomial_control': False}
    systemsize = input_dict['num_segments'] * input_dict['transcription_order']

    qoi = DymosInterceptorGlue(systemsize, input_dict)
    dummy_vec = np.zeros(systemsize)
    # dummy_vec[2] = 1.e-6
    # t_f =   qoi.eval_QoI(dummy_vec, np.zeros(systemsize))
    # print('t_f = ', t_f)

    # grad_tf = qoi.eval_QoIGradient(np.zeros(systemsize), np.zeros(systemsize), fd_pert=1.e-1)
    # print('grad_tf = \n', grad_tf)

    grad_tf = qoi.eval_QoIGradient(np.zeros(systemsize), np.zeros(systemsize))
    print('grad_tf = \n', grad_tf)
