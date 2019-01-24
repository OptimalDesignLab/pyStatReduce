################################################################################
# This script runs a multipoint optimization for a scaneagle aircraft at regular
# cruise and 2.5G maneuver. It is based on the example in the OpenAeroStruct
# documentation at the link
# https://mdolab.github.io/OpenAeroStruct/aerostructural_wingbox_walkthrough.html
#
################################################################################

from __future__ import division, print_function
import numpy as np

from openaerostruct.geometry.utils import generate_mesh
from openaerostruct.integration.aerostruct_groups import AerostructGeometry, AerostructPoint
from openmdao.api import IndepVarComp, Problem, SqliteRecorder, pyOptSparseDriver

# Total number of nodes to use in the spanwise (num_y) and
# chordwise (num_x) directions. Vary these to change the level of fidelity.
num_y = 21 # 61 # 21
num_x = 3 # 7 # 3

# Create a mesh dictionary to feed to generate_mesh to actually create
# the mesh array.
mesh_dict = {'num_y' : num_y,
             'num_x' : num_x,
             'wing_type' : 'rect',
             'symmetry' : True,
             'span_cos_spacing' : 0.5,
             'span' : 3.11,
             'root_chord' : 0.3,
             }

mesh = generate_mesh(mesh_dict)

# Apply camber to the mesh
camber = 1 - np.linspace(-1, 1, num_x) ** 2
camber *= 0.3 * 0.05

for ind_x in range(num_x):
    mesh[ind_x, :, 2] = camber[ind_x]

# Introduce geometry manipulation variables to define the ScanEagle shape
zshear_cp = np.zeros(10)
zshear_cp[0] = .3

xshear_cp = np.zeros(10)
xshear_cp[0] = .15

chord_cp = np.ones(10)
chord_cp[0] = .5
chord_cp[-1] = 1.5
chord_cp[-2] = 1.3

radius_cp = 0.01  * np.ones(10)

# Define wing parameters
surface = {
            # Wing definition
            'name' : 'wing',        # name of the surface
            'symmetry' : True,     # if true, model one half of wing
                                    # reflected across the plane y = 0
            'S_ref_type' : 'wetted', # how we compute the wing area,
                                     # can be 'wetted' or 'projected'
            'fem_model_type' : 'tube',

            'taper' : 0.8,
            'zshear_cp' : zshear_cp,
            'xshear_cp' : xshear_cp,
            'chord_cp' : chord_cp,
            'sweep' : 20.,
            'twist_cp' : np.array([2.5, 2.5, 5.]), #np.zeros((3)),
            'thickness_cp' : np.ones((3))*.008,

            # Give OAS the radius and mesh from before
            'radius_cp' : radius_cp,
            'mesh' : mesh,

            # Aerodynamic performance of the lifting surface at
            # an angle of attack of 0 (alpha=0).
            # These CL0 and CD0 values are added to the CL and CD
            # obtained from aerodynamic analysis of the surface to get
            # the total CL and CD.
            # These CL0 and CD0 values do not vary wrt alpha.
            'CL0' : 0.0,            # CL of the surface at alpha=0
            'CD0' : 0.015,            # CD of the surface at alpha=0

            # Airfoil properties for viscous drag calculation
            'k_lam' : 0.05,         # percentage of chord with laminar
                                    # flow, used for viscous drag
            't_over_c_cp' : np.array([0.12]),      # thickness over chord ratio
            'c_max_t' : .303,       # chordwise location of maximum (NACA0015)
                                    # thickness
            'with_viscous' : True,
            'with_wave' : False,     # if true, compute wave drag

            # Material properties taken from http://www.performance-composites.com/carbonfibre/mechanicalproperties_2.asp
            'yield' : 350.e6,

            'fem_origin' : 0.35,    # normalized chordwise location of the spar
            'wing_weight_ratio' : 1., # multiplicative factor on the computed structural weight
            'struct_weight_relief' : True,    # True to add the weight of the structure to the loads on the structure
            'distributed_fuel_weight' : False,
            # Constraints
            'exact_failure_constraint' : False, # if false, use KS function
            }

# Create the problem and assign the model group
prob = Problem()

# Add problem information as an independent variables component
indep_var_comp = IndepVarComp()
indep_var_comp.add_output('Mach_number', val=mean_val_dict['mean_Ma'])
indep_var_comp.add_output('v', val=22.876, units='m/s')
indep_var_comp.add_output('re', val=1.e6, units='1/m')
indep_var_comp.add_output('rho', val=0.770816, units='kg/m**3')
indep_var_comp.add_output('speed_of_sound', val=322.2, units='m/s')
indep_var_comp.add_output('load_factor', val=1.)

indep_var_comp.add_output('alpha', val=5., units='deg')
indep_var_comp.add_output('R', val=1800e3, units='m')
indep_var_comp.add_output('CT', val=mean_val_dict['mean_TSFC'], units='1/s')
indep_var_comp.add_output('empty_cg', val=np.array([0.2, 0., 0.]), units='m')
indep_var_comp.add_output('W0', val=mean_val_dict['mean_W0'],  units='kg')
indep_var_comp.add_output('E', val=mean_val_dict['mean_E'], units='N/m**2')
indep_var_comp.add_output('G', val=mean_val_dict['mean_G'], units='N/m**2')
indep_var_comp.add_output('mrho', val=mean_val_dict['mean_mrho'], units='kg/m**3')

indep_var_comp.add_output('fuel_mass', val=10000., units='kg')

prob.model.add_subsystem('prob_vars',
     indep_var_comp,
     promotes=['*'])

# Add the AerostructGeometry group, which computes all the intermediary
# parameters for the aero and structural analyses, like the structural
# stiffness matrix and some aerodynamic geometry arrays
aerostruct_group = AerostructGeometry(surface=surface)
name = surface['name']
# Add the group to the problem
prob.model.add_subsystem(name, aerostruct_group)

# We will now loop through the number of multipoint cases
for i in range(2):
    point_name = 'AS_point_{}'.format(i)
    AS_point = AerostructPoint(surfaces=[surface], internally_connect_fuelburn=False)
    prob.model.add_subsystem(point_name, AS_point)
    # Connect flow properties to the analysis point
    prob.model.connect('v', point_name + '.v', src_indices=[i])
    prob.model.connect('Mach_number', point_name + '.Mach_number', src_indices=[i])
    prob.model.connect('re', point_name + '.re', src_indices=[i])
    prob.model.connect('rho', point_name + '.rho', src_indices=[i])
    prob.model.connect('CT', point_name + '.CT')
    prob.model.connect('R', point_name + '.R')
    prob.model.connect('W0', point_name + '.W0')
    prob.model.connect('speed_of_sound', point_name + '.speed_of_sound', src_indices=[i])
    prob.model.connect('empty_cg', point_name + '.empty_cg')
    prob.model.connect('load_factor', point_name + '.load_factor', src_indices=[i])
    prob.model.connect('fuel_mass', point_name + '.total_perf.L_equals_W.fuelburn')
    prob.model.connect('fuel_mass', point_name + '.total_perf.CG.fuelburn')

    name = surface['name'] # Reassigning for convenience of porting
    if surface['distributed_fuel_weight']:
        prob.model.connect('load_factor', point_name + '.coupled.load_factor', src_indices=[i])

    com_name = point_name + '.' + name + '_perf.'
    prob.model.connect(name + '.local_stiff_transformed', point_name + '.coupled.' + name + '.local_stiff_transformed')
    prob.model.connect(name + '.nodes', point_name + '.coupled.' + name + '.nodes')

    # Connect aerodyamic mesh to coupled group mesh
    prob.model.connect(name + '.mesh', point_name + '.coupled.' + name + '.mesh')

    # Connect performance calculation variables
    prob.model.connect(name + '.radius', com_name + '.radius')
    prob.model.connect(name + '.thickness', com_name + '.thickness')
    prob.model.connect(name + '.nodes', com_name + '.nodes')
    prob.model.connect(name + '.cg_location', point_name + '.' + 'total_perf.' + name + '_cg_location')
    prob.model.connect(name + '.structural_weight', point_name + '.' + 'total_perf.' + name + '_structural_weight')
    prob.model.connect(name + '.t_over_c', com_name + '.t_over_c')

    prob.model.connect('mrho', name + '.struct_setup.structural_weight.mrho')
    prob.model.connect('E', name + '.struct_setup.assembly.E')
    prob.model.connect('G', name + '.struct_setup.assembly.G')
    prob.model.connect('E', com_name + '.struct_funcs.vonmises.E')
    prob.model.connect('G', com_name + '.struct_funcs.vonmises.G')

prob.model.connect('alpha', 'AS_point_0' + '.alpha')
prob.model.connect('alpha_maneuver', 'AS_point_1' + '.alpha')

# The assumption for this implementation is that ScanEagle doesn't store any fuel
# within the wings (Which seems true from actual photos of the aircraft). Thus
# we don't need to add a fuel volume constraint. (i.e. we do not include the
# code between checkpoint 18 and 19 in the link enclosed above)

comp = ExecComp('fuel_diff = (fuel_mass - fuelburn) / fuelburn')
prob.model.add_subsystem('fuel_diff', comp,
    promotes_inputs=['fuel_mass'],
    promotes_outputs=['fuel_diff'])
prob.model.connect('AS_point_0.fuelburn', 'fuel_diff.fuelburn')

# Set the optimizer type
# from openmdao.api import ScipyOptimizeDriver
prob.driver = pyOptSparseDriver() # ScipyOptimizeDriver()
# prob.driver.options['tol'] = 1e-7
prob.driver.options['optimizer'] = 'SNOPT'
# prob.driver.options['gradient method'] = 'pyopt_fd' # 'snopt_fd'
prob.driver.options['print_results'] = True
prob.driver.opt_settings['Major feasibility tolerance'] = 1e-9

# Record data from this problem so we can visualize it using plot_wing
recorder = SqliteRecorder("aerostruct.db")
prob.driver.add_recorder(recorder)
prob.driver.recording_options['record_derivatives'] = True
prob.driver.recording_options['includes'] = ['*']

# Setup problem and add design variables.
# Here we're varying twist, thickness, sweep, and alpha.
prob.model.add_design_var('wing.twist_cp', lower=-5., upper=10.)
prob.model.add_design_var('wing.thickness_cp', lower=0.001, upper=0.01, scaler=1e3)
prob.model.add_design_var('wing.sweep', lower=10., upper=30.)
prob.model.add_design_var('alpha', lower=-10., upper=10.)

# Make sure the spar doesn't fail, we meet the lift needs, and the aircraft
# is trimmed through CM=0.
prob.model.add_constraint('AS_point_0.wing_perf.failure', upper=0.)
prob.model.add_constraint('AS_point_0.wing_perf.thickness_intersects', upper=0.)
prob.model.add_constraint('AS_point_0.L_equals_W', equals=0.)

# Instead of using an equality constraint here, we have to give it a little
# wiggle room to make SLSQP work correctly.
prob.model.add_constraint('AS_point_0.CM', lower=-0.001, upper=0.001)
prob.model.add_constraint('wing.twist_cp', lower=np.array([-1e20, -1e20, 5.]), upper=np.array([1e20, 1e20, 5.]))

# We're trying to minimize fuel burn
prob.model.add_objective('AS_point_0.fuelburn', scaler=.1)

# Set up the problem
prob.setup()
prob.run_driver()
