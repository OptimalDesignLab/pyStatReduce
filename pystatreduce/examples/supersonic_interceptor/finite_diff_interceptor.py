import matplotlib.pyplot as plt

import numpy as np
import numdifftools as nd

from openmdao.api import Problem, Group, IndepVarComp, pyOptSparseDriver, DirectSolver
from openmdao.utils.assert_utils import assert_rel_error

import dymos as dm
from dymos.examples.min_time_climb.min_time_climb_ode import MinTimeClimbODE
# from pystatreduce.examples.supersonic_interceptor.min_time_climb_ode import MinTimeClimbODE
from dymos.examples.plotting import plot_results

class IterceptorWrapper(object):
    def __init__(self, num_segments=15, transcription_order=3, transcription_type='LGR'):
        self.num_segments = num_segments
        self.transcription_order = transcription_order
        self.transcription_type = transcription_type

        self.p = Problem(model=Group())

        self.p.driver = pyOptSparseDriver()
        self.p.driver.options['optimizer'] = 'SNOPT'
        self.p.driver.options['dynamic_simul_derivs'] = True

        self.p.driver.opt_settings['Major iterations limit'] = 1000
        # self.p.driver.opt_settings['iSumm'] = 6
        self.p.driver.opt_settings['Major feasibility tolerance'] = 1.0E-6
        self.p.driver.opt_settings['Major optimality tolerance'] = 1.0E-6
        self.p.driver.opt_settings['Function precision'] = 1.0E-12
        self.p.driver.opt_settings['Linesearch tolerance'] = 0.1
        self.p.driver.opt_settings['Major step limit'] = 0.5
        self.p.driver.options['print_results'] = False

        lgl =  dm.GaussLobatto(num_segments=self.num_segments, order=self.transcription_order)
        lgr = dm.Radau(num_segments=self.num_segments, order=self.transcription_order)
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
                                ref=1.0E3, defect_ref=1.0E3, units='m')

        phase.set_state_options('h', fix_initial=True, lower=0, upper=20000.0,
                                ref=1.0E2, defect_ref=1.0E2, units='m')

        phase.set_state_options('v', fix_initial=True, lower=10.0,
                                ref=1.0E2, defect_ref=1.0E2, units='m/s')

        phase.set_state_options('gam', fix_initial=True, lower=-1.5, upper=1.5,
                                ref=1.0, defect_ref=1.0, units='rad')

        phase.set_state_options('m', fix_initial=True, lower=10.0, upper=1.0E5,
                                ref=1.0E3, defect_ref=1.0E3)

        phase.add_polynomial_control('alpha', units='deg', lower=-8., upper=8., order=5)

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

        if self.transcription_type is 'RK4':
            self.p['traj.phase0.polynomial_controls:alpha'] = np.array([[4.86918595],
                                                                        [1.30322324],
                                                                        [1.41897019],
                                                                        [1.10227365],
                                                                        [3.58780732],
                                                                        [5.36233472]])
            self.p['traj.phase0.t_duration'] = 346.13171325

    def evaluate_time(self, alpha):
        self.update_alpha(alpha)
        if self.transcription_type is 'RK4':
            self.p.run_model()
            print(self.p.get_val('traj.phase0.t_duration'))
        else:
            self.p.run_driver()

        return self.p.get_val('traj.phase0.t_duration')

    def evaluate_rho(self, alpha):
        self.update_alpha(alpha)
        if self.transcription_type is 'RK4':
            self.p.run_model()
            return self.p['traj.phase0.ode.atmos.rho']
        elif self.transcription_type is 'LGR':
            self.p.run_driver()
            return self.p.get_val('traj.phase0.rhs_all.atmos.rho')
        elif self.transcription_type is 'LGL':
            self.p.run_driver()
            return self.p.get_val('traj.phase0.rhs_disc.atmos.rho')

    def evaluate_dtf_dalpha(self, alpha):
        self.update_alpha(alpha)
        grad = nd.Jacobian(self.evaluate_time)(alpha)
        return grad

    def evaluate_drhodalpha(self, alpha):
        self.update_alpha(alpha)
        jac = nd.Jacobian(self.evaluate_rho)(alpha)
        return jac

    def update_alpha(self, alpha):
        self.p['traj.phase0.polynomial_controls:alpha'] = alpha

if __name__ == '__main__':
    optimal_alpha = np.array([[4.86918595],
                              [1.30322324],
                              [1.41897019],
                              [1.10227365],
                              [3.58780732],
                              [5.36233472]])

    interceptor_obj = IterceptorWrapper(transcription_type='RK4')
    # dtda = interceptor_obj.evaluate_dtf_dalpha(optimal_alpha)
    # print('dtf_dalpha = \n', dtda)

    # print('rho = ', interceptor_obj.evaluate_rho(optimal_alpha))
    drhoda = interceptor_obj.evaluate_drhodalpha(optimal_alpha)
    print('drho_dalpha = \n', drhoda)
