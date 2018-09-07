import numpy as np
import matplotlib.pyplot as plt

from openmdao.api import Problem, Group, pyOptSparseDriver, DirectSolver, IndepVarComp
from openmdao.utils.assert_utils import assert_rel_error

from dymos import Phase

from dymos.examples.aircraft_steady_flight.aircraft_ode import AircraftODE
from dymos.utils.lgl import lgl

p = Problem(model=Group())
p.driver = pyOptSparseDriver()
p.driver.options['optimizer'] = 'SLSQP'
p.driver.options['dynamic_simul_derivs'] = True

num_seg = 15
seg_ends, _ = lgl(num_seg + 1)

phase = Phase('gauss-lobatto',
              ode_class=AircraftODE,
              num_segments=num_seg,
              segment_ends=seg_ends,
              transcription_order=5,
              compressed=False)

# Pass design parameters in externally from an external source
assumptions = p.model.add_subsystem('assumptions', IndepVarComp())
assumptions.add_output('S', val=427.8, units='m**2')
assumptions.add_output('mass_empty', val=1.0, units='kg')
assumptions.add_output('mass_payload', val=1.0, units='kg')

p.model.add_subsystem('phase0', phase)

phase.set_time_options(fix_initial=True,
                       duration_bounds=(300, 10000),
                       duration_ref=3600)

phase.set_state_options('range', units='NM', fix_initial=True, fix_final=False,
                        scaler=0.001,
                        defect_scaler=1.0E-2)

phase.set_state_options('mass_fuel', units='lbm', fix_initial=True, fix_final=True,
                        upper=1.5E5, lower=0.0, scaler=1.0E-5, defect_scaler=1.0E-1)

phase.add_control('alt', units='kft', opt=True, lower=0.0, upper=50.0,
                  rate_param='climb_rate',
                  rate_continuity=True, rate_continuity_scaler=1.0,
                  rate2_continuity=True, rate2_continuity_scaler=1.0, ref=1.0,
                  fix_initial=True, fix_final=True)

phase.add_control('mach', units=None, opt=False)

phase.add_design_parameter('S', units='m**2', opt=False)
phase.add_design_parameter('mass_empty', units='kg', opt=False)
phase.add_design_parameter('mass_payload', units='kg', opt=False)

phase.add_path_constraint('propulsion.tau', lower=0.01, upper=1.0)
phase.add_path_constraint('alt_rate', units='ft/min', lower=-3000, upper=3000, ref=3000)

p.model.connect('assumptions.S', 'phase0.design_parameters:S')
p.model.connect('assumptions.mass_empty', 'phase0.design_parameters:mass_empty')
p.model.connect('assumptions.mass_payload', 'phase0.design_parameters:mass_payload')

phase.add_objective('range', loc='final', ref=-1.0)

p.model.linear_solver = DirectSolver(assemble_jac=True)

p.setup()

p['phase0.t_initial'] = 0.0
p['phase0.t_duration'] = 3600.0
p['phase0.states:range'] = phase.interpolate(ys=(0, 1000.0), nodes='state_input')
p['phase0.states:mass_fuel'] = phase.interpolate(ys=(30000, 0), nodes='state_input')

p['phase0.controls:mach'][:] = 0.8
p['phase0.controls:alt'][:] = 10.0

p['assumptions.S'] = 427.8
p['assumptions.mass_empty'] = 0.15E6
p['assumptions.mass_payload'] = 84.02869 * 400

p.run_driver()

exp_out = phase.simulate(times=np.linspace(0, p['phase0.t_duration'], 500), record=True,
                         record_file='test_doc_aircraft_steady_flight_rec.db')

print(phase.get_values('range', units='NM')[-1])

# Compute the derivatives
totals = p.compute_totals(of=['phase0.states:range'], wrt=['phase0.controls:alt'])
print(totals[('phase0.states:range','phase0.controls:alt')])
print(p['phase0.controls:alt'])
print(p['phase0.states:range'])