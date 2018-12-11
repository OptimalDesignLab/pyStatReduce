################################################################################
# This script runs an aerostructural optimization for the ScanEagle airplane,
# a small drone used for recon missions. The geometry definition comes from
# a variety of sources, including spec sheets and discussions with the
# manufacturer, Insitu.
#
# Results using this model were presented in this paper:
# https://arc.aiaa.org/doi/abs/10.2514/6.2018-1658
# which was presented at AIAA SciTech 2018.
#
# This specific file is for John's fork and branch
# https://github.com/johnjasa/OpenAeroStruct/tree/move_surface_vars
# commit hash ee10ee86e0aec273d8e4db9cfe2871426d2e57a8
################################################################################

from __future__ import division, print_function
import numpy as np

from openaerostruct.geometry.utils import generate_mesh
from openaerostruct.integration.aerostruct_groups import AerostructGeometry, AerostructPoint
from openmdao.api import IndepVarComp, Problem, SqliteRecorder, pyOptSparseDriver

import time

# Initial default values
mean_val_dict = {'mean_Ma' : 0.071,
                 'mean_TSFC' : 9.80665 * 8.6e-6,
                 'mean_W0' : 10.0,
                 'mean_E' : 85.e9,
                 'mean_G' : 25.e9,
                 'mean_mrho' : 1600,
                }


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
indep_var_comp.add_output('v', val=22.876, units='m/s')
indep_var_comp.add_output('alpha', val=5., units='deg')
indep_var_comp.add_output('re', val=1.e6, units='1/m')
indep_var_comp.add_output('rho', val=0.770816, units='kg/m**3')
indep_var_comp.add_output('R', val=1800e3, units='m')
indep_var_comp.add_output('speed_of_sound', val=322.2, units='m/s')
indep_var_comp.add_output('load_factor', val=1.)
indep_var_comp.add_output('empty_cg', val=np.array([0.2, 0., 0.]), units='m')

indep_var_comp.add_output('Mach_number', val=mean_val_dict['mean_Ma'])
indep_var_comp.add_output('CT', val=mean_val_dict['mean_TSFC'], units='1/s')
indep_var_comp.add_output('W0', val=mean_val_dict['mean_W0'],  units='kg')
indep_var_comp.add_output('E', val=mean_val_dict['mean_E'], units='N/m**2')
indep_var_comp.add_output('G', val=mean_val_dict['mean_G'], units='N/m**2')
indep_var_comp.add_output('mrho', val=mean_val_dict['mean_mrho'], units='kg/m**3')

prob.model.add_subsystem('prob_vars',
     indep_var_comp,
     promotes=['*'])

# Add the AerostructGeometry group, which computes all the intermediary
# parameters for the aero and structural analyses, like the structural
# stiffness matrix and some aerodynamic geometry arrays
aerostruct_group = AerostructGeometry(surface=surface)

name = 'wing'

# Add the group to the problem
prob.model.add_subsystem(name, aerostruct_group,
    promotes_inputs=['load_factor'])

point_name = 'AS_point_0'

# Create the aerostruct point group and add it to the model.
# This contains all the actual aerostructural analyses.
AS_point = AerostructPoint(surfaces=[surface])

prob.model.add_subsystem(point_name, AS_point,
    promotes_inputs=['v', 'alpha', 'Mach_number', 're', 'rho', 'CT', 'R',
        'W0', 'speed_of_sound', 'empty_cg', 'load_factor'])

# Issue quite a few connections within the model to make sure all of the
# parameters are connected correctly.
com_name = point_name + '.' + name + '_perf'
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

# # Use this if you just want to run analysis and not optimization
# prob.run_model()
# # print("fval = ", prob['AS_point_0.fuelburn'][0])
# # print("twist = ", prob['wing.geometry.twist'])
# # print("thickness =", prob['wing.thickness'])
# # print("sweep = ", prob['wing.sweep'])
# # print("aoa = ", prob['alpha'])
# # print("nodes = ", np.round(prob['AS_point_0.coupled.wing.struct_states.struct_weight_loads.nodes'][:,1], 5))

#-----------------------VALUE PLUGIN TEST---------------------------------------
# prob['wing.twist_cp'] = np.array([-1.59, 0.34, 4.50]) # np.array([2.60830137, 10., 5.])
# prob['wing.thickness_cp'] = np.array([1.0, 1.04, 3.41]) # np.array([0.001, 0.001, 0.001])
# prob['wing.sweep'] = 18.73 # [18.89098985]
# prob['alpha'] = 5.54 # [2.19244059]

# prob.run_model()
# print("fval = ", prob['AS_point_0.fuelburn'][0])

#------------------------SETUP BUG REPRODUCTION---------------------------------
# new_Ma = 0.071 + 0.005
# prob['Mach_number'] = new_Ma
# prob.run_model()

# prob.setup()
# prob.run_model()
# print("fval = ", prob['AS_point_0.fuelburn'][0])

# print("Mach_number = ", prob['Mach_number'])
# prob.run_model()
# print("fval = ", prob['AS_point_0.fuelburn'][0])

#--------------------FINITE DIFFERENCE -----------------------------------------
# fval1 = prob['AS_point_0.fuelburn'][0]
# # Check against finite difference
# pert = 1.e-6
# surface['mrho'] += pert
# prob.run_model()
# fval2 = prob['AS_point_0.fuelburn'][0]
# dfval = (fval2 - fval1) / pert
# print("dfval = ", dfval)

#----------------------------OPTIMIZATION--------------------------------------
# Actually run the optimization problem
start_time = time.time()
prob.run_driver()
elapsed_time = time.time() - start_time
print("fval = ", prob['AS_point_0.fuelburn'][0])
print("twist_cp = ", prob['wing.twist_cp'])
print("thickness_cp = ", prob['wing.thickness_cp'])
print("twist = ", prob['wing.geometry.twist'])
print("thickness =", prob['wing.thickness'])
print("sweep = ", prob['wing.sweep'])
print('alpha = ', prob['alpha'])
print('elapsed_time = ', elapsed_time)
