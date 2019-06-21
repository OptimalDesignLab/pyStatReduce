import matplotlib.pyplot as plt

import numpy as np
import numdifftools as nd
import copy

from openmdao.api import Problem, Group, IndepVarComp, pyOptSparseDriver, DirectSolver
from openmdao.utils.assert_utils import assert_rel_error

import dymos as dm
from dymos.examples.min_time_climb.min_time_climb_ode import MinTimeClimbODE
from pystatreduce.examples.supersonic_interceptor.min_time_climb_ode import MinTimeClimbODE
# from dymos.examples.plotting import plot_results

class IterceptorWrapper2(object):
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

    def evaluate_time(self, alpha):
        self.update_alpha(alpha)
        if self.transcription_type is 'RK4':
            self.p.run_driver()
        else:
            if self.solve_segments is True:
                self.p.run_model()
            else:
                self.p.run_driver()

        return self.p.get_val('traj.phase0.t_duration')[0]

    def evaluate_rho(self, alpha):
        self.update_alpha(alpha)
        if self.transcription_type is 'RK4':
            self.p.run_model()
            return self.p['traj.phase0.ode.atmos.rho']
        elif self.transcription_type is 'LGR':
            if self.solve_segments is True:
                self.p.run_model()
            else:
                self.p.run_driver()
            return self.p.get_val('traj.phase0.rhs_all.atmos.rho')
        elif self.transcription_type is 'LGL':
            if self.solve_segments is True:
                self.p.run_model()
            else:
                self.p.run_driver()
            return self.p.get_val('traj.phase0.rhs_disc.atmos.rho')

    def update_alpha(self, alpha):
        self.p['traj.phase0.polynomial_controls:alpha'] = alpha

    def set_random_perturbations(self, std_pert):
        expanded_pert = np.expand_dims(std_pert, axis=0)
        self.p.set_val('random_perturbations.rho_pert', std_pert, 'kg/m**3')

    def evaluate_dtf_drho(self, alpha):
        t_orig = self.evaluate_time(alpha)
        print('t_orig = ', t_orig)
        n_nodes = self.num_segments * (self.transcription_order+1)
        pert_arr = np.zeros(n_nodes)
        pert_val = 1.e-6
        dtf_drho = np.zeros(n_nodes)
        for i in range(n_nodes):
            pert_arr[i] += pert_val
            self.set_random_perturbations(pert_arr)
            t_pert = self.evaluate_time(alpha)
            dtf_drho[i] = (t_pert - t_orig) / pert_val
            pert_arr[i] -= pert_val

        # # Only for LGR
        # segment_id = 0
        # i = 0
        # while i < n_nodes:
        #     print('i = ', i)
        #     if i is (segment_id+1)*(self.transcription_order+1) -1 and i != (n_nodes-1):
        #         print('     i = ', i)
        #         pert_arr[i] += pert_val
        #         pert_arr[i+1] += pert_val
        #         self.set_random_perturbations(pert_arr)
        #         t_pert = self.evaluate_time(alpha)
        #         # print('t_pert = ', t_pert)
        #         dtf_drho[i] = (t_pert - t_orig) / pert_val
        #         dtf_drho[i+1] = dtf_drho[i]
        #         segment_id += 1
        #         pert_arr[i] -= pert_val
        #         pert_arr[i+1] -= pert_val
        #         i +=2
        #     else:
        #         pert_arr[i] += pert_val
        #         self.set_random_perturbations(pert_arr)
        #         t_pert = self.evaluate_time(alpha)
        #         # print('t_pert = ', t_pert)
        #         dtf_drho[i] = (t_pert - t_orig) / pert_val
        #         pert_arr[i] -= pert_val
        #         i += 1

        return dtf_drho

if __name__ == '__main__':
    optimal_alpha = np.array([[4.86918595],
                              [1.30322324],
                              [1.41897019],
                              [1.10227365],
                              [3.58780732],
                              [5.36233472]])

    interceptor_obj = IterceptorWrapper2(transcription_type='LGR', solve_segments=False)
    # tf = interceptor_obj.evaluate_time(optimal_alpha)
    # print('tf = ', tf)
    # tf = interceptor_obj.evaluate_time(optimal_alpha)
    # print('tf = ', tf)
    # print('h = ', interceptor_obj.p.driver.get_design_var_values()['traj.phases.phase0.indep_states.states:h'].size)
    # print('h_schedule = ', interceptor_obj.p['traj.phase0.states:h'])

    dtf_drho = interceptor_obj.evaluate_dtf_drho(optimal_alpha)
    print('dtf_drho = \n', dtf_drho)


    # pert_arr = np.zeros(60)
    # pert_arr[0] = 1.e-6
    # interceptor_obj.set_random_perturbations(pert_arr)
    # tf = interceptor_obj.evaluate_time(optimal_alpha)
    # print('tf = ', tf)
