import matplotlib.pyplot as plt

import numpy as np

from openmdao.api import Problem, Group, IndepVarComp, pyOptSparseDriver, DirectSolver
from openmdao.utils.assert_utils import assert_rel_error

import dymos as dm
# from dymos.examples.min_time_climb.min_time_climb_ode import MinTimeClimbODE
from pystatreduce.examples.supersonic_interceptor.min_time_climb_ode import MinTimeClimbODE
from dymos.examples.plotting import plot_results

#
# Instantiate the problem and configure the optimization driver
#
p = Problem(model=Group())

p.driver = pyOptSparseDriver()
p.driver.options['optimizer'] = 'SNOPT'
p.driver.options['dynamic_simul_derivs'] = False

p.driver.opt_settings['Major iterations limit'] = 1000
p.driver.opt_settings['iSumm'] = 6
p.driver.opt_settings['Major feasibility tolerance'] = 1.0E-9
p.driver.opt_settings['Major optimality tolerance'] = 1.0E-9
p.driver.opt_settings['Function precision'] = 1.0E-12
p.driver.opt_settings['Linesearch tolerance'] = 0.1
p.driver.opt_settings['Major step limit'] = 0.5
p.driver.options['print_results'] = False

n_nodes = 60
seed_perturbation = np.zeros(n_nodes)
# seed_perturbation[2] = 1.e-6

# Add an indep_var_comp that will talk to external calls from pyStatReduce
random_perturbations = p.model.add_subsystem('random_perturbations', IndepVarComp())
random_perturbations.add_output('rho_pert', val=seed_perturbation, units='kg/m**3',
                                desc="perturbations introduced into the density data")

#
# Instantiate the trajectory and phase
#
traj = dm.Trajectory()

phase = dm.Phase(ode_class=MinTimeClimbODE,
                 transcription=dm.Radau(num_segments=15, compressed=True))

traj.add_phase('phase0', phase)

p.model.add_subsystem('traj', traj)

#
# Set the options on the optimization variables
#
phase.set_time_options(fix_initial=True, duration_bounds=(50, 400),
                       duration_ref=100.0)

phase.set_state_options('r', fix_initial=True, lower=0, upper=1.0E6,
                        ref=1.0E3, defect_ref=1.0E3, units='m')

phase.set_state_options('h', fix_initial=True, lower=0, upper=20000.0,
                        ref=1.0E2, defect_ref=1.0E2, units='m')

phase.set_state_options('v', fix_initial=True, lower=10.0,
                        ref=1.0E2, defect_ref=1.0E2, units='m/s')

phase.set_state_options('gam', fix_initial=True, lower=-1.5, upper=1.5,
                        ref=1.0, defect_scaler=1.0, units='rad')

phase.set_state_options('m', fix_initial=True, lower=10.0, upper=1.0E5,
                        ref=1.0E3, defect_ref=1.0E3)

# phase.add_control('alpha', units='deg', lower=-8.0, upper=8.0, scaler=1.0,
#                   rate_continuity=True, rate_continuity_scaler=100.0,
#                   rate2_continuity=False)

phase.add_polynomial_control('alpha', units='deg', lower=-8., upper=8., order=5)

phase.add_design_parameter('S', val=49.2386, units='m**2', opt=False)
phase.add_design_parameter('Isp', val=1600.0, units='s', opt=False)
phase.add_design_parameter('throttle', val=1.0, opt=False)

# Add the random parameters to dymos
phase.add_input_parameter('rho_pert', shape=(n_nodes,), dynamic=False, units='kg/m**3')

p.model.connect('random_perturbations.rho_pert', 'traj.phase0.input_parameters:rho_pert')

#
# Setup the boundary and path constraints
#
phase.add_boundary_constraint('h', loc='final', equals=20000, scaler=1.0E-3, units='m')
phase.add_boundary_constraint('aero.mach', loc='final', equals=1.0, shape=(1,))
phase.add_boundary_constraint('gam', loc='final', equals=0.0, units='rad')

phase.add_path_constraint(name='h', lower=100.0, upper=20000, ref=20000)
phase.add_path_constraint(name='aero.mach', lower=0.1, upper=1.8, shape=(1,))

# Minimize time at the end of the phase
phase.add_objective('time', loc='final', ref=1.0)

p.model.linear_solver = DirectSolver()

#
# Setup the problem and set the initial guess
#
p.setup(check=True)

p['traj.phase0.t_initial'] = 0.0
p['traj.phase0.t_duration'] = 500

p['traj.phase0.states:r'] = phase.interpolate(ys=[0.0, 50000.0], nodes='state_input')
p['traj.phase0.states:h'] = phase.interpolate(ys=[100.0, 20000.0], nodes='state_input')
p['traj.phase0.states:v'] = phase.interpolate(ys=[135.964, 283.159], nodes='state_input')
p['traj.phase0.states:gam'] = phase.interpolate(ys=[0.0, 0.0], nodes='state_input')
p['traj.phase0.states:m'] = phase.interpolate(ys=[19030.468, 10000.], nodes='state_input')
# p['traj.phase0.controls:alpha'] = phase.interpolate(ys=[0.0, 0.0], nodes='control_input')

p['traj.phase0.polynomial_controls:alpha'] = np.array([[4.86918595],
                                                        [1.30322324],
                                                        [1.41897019],
                                                        [1.10227365],
                                                        [3.58780732],
                                                        [5.36233472]])
p['traj.phase0.t_duration'] = 346.13171325
#
# Solve for the optimal trajectory
#
# p.run_model()
p.run_driver()
print(p.get_val('traj.phase0.t_duration')[0])
# print(p.get_val('traj.phase0.rhs_all.atmos.rho'))
# p.run_model()
# totals = p.compute_totals(of=['traj.phase0.states:m'], wrt=['traj.phase0.timeseries.controls:alpha'])
# print(totals['traj.phase0.states:m', 'traj.phase0.timeseries.controls:alpha'])
#
# Test the results
#
# print(p.driver.get_constraint_values().keys())


# print('len m = ', p.driver.get_constraint_values()['traj.phases.phase0.collocation_constraint.defects:m'].size)
# print('len gamma = ', p.driver.get_constraint_values()['traj.phases.phase0.collocation_constraint.defects:gam'].size)
# print('len v = ', p.driver.get_constraint_values()['traj.phases.phase0.collocation_constraint.defects:v'].size)
# print('len h = ', p.driver.get_constraint_values()['traj.phases.phase0.collocation_constraint.defects:h'].size)
# print('len r = ', p.driver.get_constraint_values()['traj.phases.phase0.collocation_constraint.defects:r'].size)
# print('len alpha = ', p.driver.get_constraint_values()['traj.phases.phase0.continuity_comp.defect_controls:alpha'].size)
# print('len alpha_dot = ', p.driver.get_constraint_values()['traj.phases.phase0.continuity_comp.defect_control_rates:alpha_rate'].size)
# print('len M_f = ', p.driver.get_constraint_values()[ 'traj.phases.phase0.final_boundary_constraints.final_value:mach'].size)
# print('len path_h = ', p.driver.get_constraint_values()['traj.phases.phase0.path_constraints.path:h'].size)
# print('len path_M = ', p.driver.get_constraint_values()['traj.phases.phase0.path_constraints.path:mach'].size)





# print('S = ', p['traj.phase0.design_parameters:S'])
# print('Isp = ', p['traj.phase0.design_parameters:Isp'])
# print('throttle = ', p['traj.phase0.design_parameters:throttle'])
# print(p.get_val('traj.phase0.t_duration'))


#
# Get the explicitly simulated solution and plot the results
#
# exp_out = traj.simulate()
# plot_results([('traj.phase0.timeseries.time', 'traj.phase0.timeseries.states:h',
#                'time (s)', 'altitude (m)'),
#               ('traj.phase0.timeseries.time', 'traj.phase0.timeseries.controls:alpha',
#                'time (s)', 'alpha (deg)')],
#              title='Supersonic Minimum Time-to-Climb Solution',
#              p_sol=p, p_sim=exp_out)
# plt.show()
