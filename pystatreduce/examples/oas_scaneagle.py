# Use the OpenMDAO framework for OUU for optimizing a Rosenbrock function
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

# Import the OpenMDAo shenanigans
from openmdao.api import IndepVarComp, Problem, Group, NewtonSolver, \
    ScipyIterativeSolver, LinearBlockGS, NonlinearBlockGS, \
    DirectSolver, LinearBlockGS, PetscKSP, SqliteRecorder, ScipyOptimizeDriver

from openaerostruct.geometry.utils import generate_mesh
from openaerostruct.integration.aerostruct_groups import AerostructGeometry, AerostructPoint
from openaerostruct.common.atmos_group import AtmosGroup
from openmdao.api import IndepVarComp, Problem, SqliteRecorder

class OASScanEagleWrapper(QuantityOfInterest):
    """
    Wrapper class for the ScanEagle problem in OpenAeroStruct, that is being called
    through the Group OASScanEagle below
    TODO: Add the dictionary externally to have better control

    NOTE: The Random variables as of this commit are split between
    `indep_var_comp` and the `surface` dictionary. Thus there exists a boolean
    option within the `OASScanEagleWrapper` constructor called `include_dict_rv`
    which, by default ( = False), only considers the random variables in
    `indep_var_comp`. If True, it will also consider the random variables in the
    `surface` dictionary.
    """
    def __init__(self, systemsize, input_dict, data_type=np.float):
        QuantityOfInterest.__init__(self, systemsize, data_type=data_type)
        self.input_dict = input_dict
        # self.include_dict_rv = include_dict_rv # Bool for including random variables in the durface dictionary
        self.rv_dict = self.input_dict['rv_dict']

        self.p = Problem()
        self.rvs = self.p.model.add_subsystem('random_variables', IndepVarComp(), promotes_outputs=['*'])
        self.p.model.add_subsystem('oas_scaneagle',
                                   OASScanEagle(mesh_dict=self.input_dict['mesh_dict'],
                                                surface_dict_rv=self.rv_dict))

        # Declare rvs units to ensure type stability
        # self.rvs.add_output('Mach_number', val=0.071)
        self.rvs.add_output('Mach_number', val=self.rv_dict['Mach_number'])
        self.p.model.connect('Mach_number', 'oas_scaneagle.Mach_number')

        # self.rvs.add_output('CT', val=9.80665 * 8.6e-6, units='1/s') # TSFC
        self.rvs.add_output('CT', val=self.rv_dict['CT'], units='1/s') # TSFC
        self.p.model.connect('CT', 'oas_scaneagle.CT')

        # self.rvs.add_output('W0', val=10.,  units='kg')
        self.rvs.add_output('W0', val=self.rv_dict['W0'],  units='kg')
        self.p.model.connect('W0', 'oas_scaneagle.W0')

        self.rvs.add_output('E', val=self.rv_dict['E'], units='N/m**2')
        self.p.model.connect('E', 'oas_scaneagle.wing.struct_setup.assembly.E')
        self.p.model.connect('E', 'oas_scaneagle.AS_point_0.wing_perf.struct_funcs.vonmises.E')

        self.rvs.add_output('G', val=self.rv_dict['G'], units='N/m**2')
        self.p.model.connect('G', 'oas_scaneagle.wing.struct_setup.assembly.G')
        self.p.model.connect('G', 'oas_scaneagle.AS_point_0.wing_perf.struct_funcs.vonmises.G')

        self.rvs.add_output('mrho', val=self.rv_dict['mrho'], units='kg/m**3')
        self.p.model.connect('mrho', 'oas_scaneagle.wing.struct_setup.structural_weight.mrho')

        self.p.setup()

        # Set up reusable arrays
        self.dJ_ddv = np.zeros(self.input_dict['ndv'], dtype=self.data_type) # Used in eval_ObjGradient_dv
        self.con_arr = np.zeros(self.input_dict['n_constraints'], dtype=self.data_type) # Used in eval_ConstraintQoI
        self.con_jac = np.zeros((self.input_dict['n_constraints'], self.input_dict['ndv']), dtype=self.data_type)

    def eval_QoI(self, mu, xi):
        """
        Computes the Quantity of Interest (in this case the objective function)
        for a given random variable realization.
        """
        rv = mu + xi
        self.update_rv(rv)
        self.p.run_model()
        return self.p['oas_scaneagle.AS_point_0.fuelburn']

    def eval_QoIGradient(self, mu, xi):
        """
        Computes the gradient of the QoI w.r.t the random variables
        """
        rv = mu + xi
        deriv_arr = np.zeros(self.systemsize, dtype=self.data_type)
        self.update_rv(rv)
        self.p.run_model()
        deriv = self.p.compute_totals(of=['oas_scaneagle.AS_point_0.fuelburn'],
                            wrt=['Mach_number', 'CT', 'W0', 'E', 'G', 'mrho'])
        deriv_arr[0] = deriv['oas_scaneagle.AS_point_0.fuelburn', 'Mach_number']
        deriv_arr[1] = deriv['oas_scaneagle.AS_point_0.fuelburn', 'CT']
        deriv_arr[2] = deriv['oas_scaneagle.AS_point_0.fuelburn', 'W0']
        deriv_arr[3] = deriv['oas_scaneagle.AS_point_0.fuelburn', 'E']
        deriv_arr[4] = deriv['oas_scaneagle.AS_point_0.fuelburn', 'G']
        deriv_arr[5] = deriv['oas_scaneagle.AS_point_0.fuelburn', 'mrho']

        return deriv_arr

    def eval_ObjGradient_dv(self, mu, xi):
        """
        Computes the gradient of the QoI (in this case the objective function)
        w.r.t the design variables for a given set of random variable
        realizations. The design variables for this implementation are NOT the
        random variables.
        """
        rv = mu + xi
        self.update_rv(rv)
        self.p.run_model()
        deriv = self.p.compute_totals(of=['oas_scaneagle.AS_point_0.fuelburn'],
                                      wrt=['oas_scaneagle.wing.twist_cp',
                                           'oas_scaneagle.wing.thickness_cp',
                                           'oas_scaneagle.wing.sweep',
                                           'oas_scaneagle.alpha'])

        self.dJ_ddv.fill(0.0)
        # dJ_ddv = np.zeros(self.input_dict['ndv'])
        n_twist_cp = self.input_dict['n_twist_cp']
        n_thickness_cp = self.input_dict['n_thickness_cp']
        n_cp = n_twist_cp + n_thickness_cp
        self.dJ_ddv[0:n_twist_cp] = deriv['oas_scaneagle.AS_point_0.fuelburn', 'oas_scaneagle.wing.twist_cp']
        self.dJ_ddv[n_twist_cp:n_cp] = deriv['oas_scaneagle.AS_point_0.fuelburn', 'oas_scaneagle.wing.thickness_cp']
        self.dJ_ddv[n_cp:n_cp+1] = deriv['oas_scaneagle.AS_point_0.fuelburn', 'oas_scaneagle.wing.sweep']
        self.dJ_ddv[n_cp+1:n_cp+2] = deriv['oas_scaneagle.AS_point_0.fuelburn', 'oas_scaneagle.alpha']

        return self.dJ_ddv

    def eval_AllConstraintQoI(self, mu, xi):
        """
        Evaluates ALL the constraint function for a given realization of random variables.
        """
        rv = mu + xi
        self.update_rv(rv)
        self.p.run_model()

        # Since the current stochastic collocation method expects a single array
        # as the functional output, we need to assemble the constraints into a
        # single array and redistribute it in the OUU function
        self.con_arr.fill(0.0) # = np.zeros(self.input_dict['n_constraints'])
        n_thickness_intersects = self.p['oas_scaneagle.AS_point_0.wing_perf.thickness_intersects'].size
        n_CM = self.p['oas_scaneagle.AS_point_0.CM'].size
        self.con_arr[0] = self.p['oas_scaneagle.AS_point_0.wing_perf.failure']
        self.con_arr[1:n_thickness_intersects+1] = self.p['oas_scaneagle.AS_point_0.wing_perf.thickness_intersects']
        self.con_arr[n_thickness_intersects+1] = self.p['oas_scaneagle.AS_point_0.L_equals_W']
        self.con_arr[n_thickness_intersects+2:n_thickness_intersects+2+n_CM] = self.p['oas_scaneagle.AS_point_0.CM']
        self.con_arr[n_thickness_intersects+2+n_CM:] = self.p['oas_scaneagle.wing.twist_cp']

        return self.con_arr

    def eval_confailureQoI(self, mu, xi):
        """
        Evaluates only the failure constraint for a given realization of random variabels.
        """
        rv = mu + xi
        self.update_rv(rv)
        self.p.run_model()

        return self.p['oas_scaneagle.AS_point_0.wing_perf.failure']

    def eval_ConGradient_dv(self, mu, xi):
        """
        Evaluates the gradient of the constraint function with respect to the
        design variables for a given realization of random variabels
        """
        rv = mu + xi
        self.update_rv(rv)
        self.p.run_model()

        # Compute all the derivatives
        # Compute derivatives
        deriv = self.p.compute_totals(of=['oas_scaneagle.AS_point_0.wing_perf.failure',
                                               'oas_scaneagle.AS_point_0.wing_perf.thickness_intersects',
                                               'oas_scaneagle.AS_point_0.L_equals_W',
                                               'oas_scaneagle.AS_point_0.CM'],
                                           wrt=['oas_scaneagle.wing.twist_cp',
                                                'oas_scaneagle.wing.thickness_cp',
                                                'oas_scaneagle.wing.sweep',
                                                'oas_scaneagle.alpha'])
        # In the interest of implicity, I will create a dense constraint jacobian
        # for now and then investigate sparseness. Also, this is a small matrix
        n_twist_cp = self.input_dict['n_twist_cp']
        n_cp = n_twist_cp + self.input_dict['n_thickness_cp']
        n_CM = self.input_dict['n_CM']
        n_thickness_intersects = self.p['oas_scaneagle.AS_point_0.wing_perf.thickness_intersects'].size
        self.con_jac.fill(0.0) # = np.zeros((self.input_dict['n_constraints'], self.input_dict['ndv']))
        # Populate con_jac
        self.con_jac[0,0:n_twist_cp] = deriv['oas_scaneagle.AS_point_0.wing_perf.failure', 'oas_scaneagle.wing.twist_cp']
        self.con_jac[0,n_twist_cp:n_cp] = deriv['oas_scaneagle.AS_point_0.wing_perf.failure', 'oas_scaneagle.wing.thickness_cp']
        self.con_jac[0,n_cp] = deriv['oas_scaneagle.AS_point_0.wing_perf.failure', 'oas_scaneagle.wing.sweep']
        self.con_jac[0,n_cp+1] = deriv['oas_scaneagle.AS_point_0.wing_perf.failure', 'oas_scaneagle.alpha']

        # con_jac[1:n_thickness_intersects+1,0:n_twist_cp] = deriv['oas_scaneagle.AS_point_0.wing_perf.thickness_intersects', 'oas_scaneagle.wing.twist_cp']
        self.con_jac[1:n_thickness_intersects+1,n_twist_cp:n_cp] = deriv['oas_scaneagle.AS_point_0.wing_perf.thickness_intersects', 'oas_scaneagle.wing.thickness_cp']
        # con_jac[1:n_thickness_intersects+1,n_cp] = deriv['oas_scaneagle.AS_point_0.wing_perf.thickness_intersects', 'oas_scaneagle.wing.sweep'][:,0]
        # con_jac[1:n_thickness_intersects+1,n_cp+1] = deriv['oas_scaneagle.AS_point_0.wing_perf.thickness_intersects', 'oas_scaneagle.alpha'][:,0]

        self.con_jac[n_thickness_intersects+1,0:n_twist_cp] = deriv['oas_scaneagle.AS_point_0.L_equals_W', 'oas_scaneagle.wing.twist_cp']
        self.con_jac[n_thickness_intersects+1,n_twist_cp:n_cp] = deriv['oas_scaneagle.AS_point_0.L_equals_W', 'oas_scaneagle.wing.thickness_cp']
        self.con_jac[n_thickness_intersects+1,n_cp] = deriv['oas_scaneagle.AS_point_0.L_equals_W', 'oas_scaneagle.wing.sweep'] # [:,0]
        self.con_jac[n_thickness_intersects+1,n_cp+1] = deriv['oas_scaneagle.AS_point_0.L_equals_W', 'oas_scaneagle.alpha']

        idx = n_thickness_intersects + 2
        self.con_jac[idx:idx+n_CM,0:n_twist_cp] = deriv['oas_scaneagle.AS_point_0.CM', 'oas_scaneagle.wing.twist_cp']
        self.con_jac[idx:idx+n_CM,n_twist_cp:n_cp] = deriv['oas_scaneagle.AS_point_0.CM', 'oas_scaneagle.wing.thickness_cp']
        self.con_jac[idx:idx+n_CM,n_cp] = deriv['oas_scaneagle.AS_point_0.CM', 'oas_scaneagle.wing.sweep'][:,0]
        self.con_jac[idx:idx+n_CM,n_cp+1] = deriv['oas_scaneagle.AS_point_0.CM', 'oas_scaneagle.alpha'][:,0]

        idx = n_thickness_intersects + 2 + n_CM
        self.con_jac[idx:,0:n_twist_cp] = np.eye(self.input_dict['n_twist_cp']) # deriv['oas_scaneagle.wing.twist_cp', 'oas_scaneagle.wing.twist_cp']
        # con_jac[idx:,n_twist_cp:n_cp] = deriv['oas_scaneagle.wing.twist_cp', 'oas_scaneagle.wing.thickness_cp']
        # con_jac[idx:,n_cp] = deriv['oas_scaneagle.wing.twist_cp', 'oas_scaneagle.wing.sweep'][:,0]
        # con_jac[idx:,n_cp+1] = deriv['oas_scaneagle.wing.twist_cp', 'oas_scaneagle.alpha'][:,0]

        return self.con_jac

    def eval_ConFailureGradient_dv(self, mu, xi):
        rv = mu + xi
        self.update_rv(rv)
        self.p.run_model()
        deriv = self.p.compute_totals(of=['oas_scaneagle.AS_point_0.wing_perf.failure'],
                                      wrt=['oas_scaneagle.wing.twist_cp',
                                           'oas_scaneagle.wing.thickness_cp',
                                           'oas_scaneagle.wing.sweep',
                                           'oas_scaneagle.alpha'])
        n_twist_cp = self.input_dict['n_twist_cp']
        n_cp = n_twist_cp + self.input_dict['n_thickness_cp']
        dcon_failure = np.zeros(self.input_dict['ndv'])
        dcon_failure[0:n_twist_cp] = deriv['oas_scaneagle.AS_point_0.wing_perf.failure', 'oas_scaneagle.wing.twist_cp']
        dcon_failure[n_twist_cp:n_cp] = deriv['oas_scaneagle.AS_point_0.wing_perf.failure', 'oas_scaneagle.wing.thickness_cp']
        dcon_failure[n_cp] = deriv['oas_scaneagle.AS_point_0.wing_perf.failure', 'oas_scaneagle.wing.sweep']
        dcon_failure[n_cp+1] = deriv['oas_scaneagle.AS_point_0.wing_perf.failure', 'oas_scaneagle.alpha']

        return dcon_failure

    def update_rv(self, rv):
        self.p['Mach_number'] = rv[0]
        self.p['CT'] = rv[1]
        self.p['W0'] = rv[2]
        self.p['E'] = rv[3]
        self.p['G'] = rv[4]
        self.p['mrho'] = rv[5]

#-------------------------------------------------------------------------------

class OASScanEagle(Group):
    """
    This is the OpenMDAO Group that get wraps in the class above for doing RDO
    under uncertainty.
    """
    def initialize(self):
        self.options.declare('mesh_dict', types=dict)
        self.options.declare('surface_dict_rv', types=dict)

    def setup(self):
        # Total number of nodes to use in the spanwise (num_y) and
        # chordwise (num_x) directions. Vary these to change the level of fidelity.

        mesh_dict = self.options['mesh_dict']
        num_y = mesh_dict['num_y']
        num_x = mesh_dict['num_x']

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

        # Add problem information as an independent variables component
        indep_var_comp = IndepVarComp()
        # indep_var_comp.add_output('v', val=22.876, units='m/s')
        indep_var_comp.add_output('alpha', val=5., units='deg')
        indep_var_comp.add_output('altitude', val=4.57e3, units='m')
        # indep_var_comp.add_output('re', val=1.e6, units='1/m')
        # indep_var_comp.add_output('rho', val=0.770816, units='kg/m**3')
        indep_var_comp.add_output('R', val=1800e3, units='m')
        # indep_var_comp.add_output('speed_of_sound', val=322.2, units='m/s')
        indep_var_comp.add_output('load_factor', val=1.)
        indep_var_comp.add_output('empty_cg', val=np.array([0.2, 0., 0.]), units='m')

        # indep_var_comp.add_output('Mach_number', val=mean_val_dict['mean_Ma'])
        # indep_var_comp.add_output('CT', val=mean_val_dict['mean_TSFC'], units='1/s')
        # indep_var_comp.add_output('W0', val=mean_val_dict['mean_W0'],  units='kg')
        # indep_var_comp.add_output('E', val=mean_val_dict['mean_E'], units='N/m**2')
        # indep_var_comp.add_output('G', val=mean_val_dict['mean_G'], units='N/m**2')
        # indep_var_comp.add_output('mrho', val=mean_val_dict['mean_mrho'], units='kg/m**3')

        self.add_subsystem('prob_vars', indep_var_comp, promotes=['*'])
        # Add atmosphere related properties
        self.add_subsystem('atmos', AtmosGroup(), promotes=['*'])

        # Add the AerostructGeometry group, which computes all the intermediary
        # parameters for the aero and structural analyses, like the structural
        # stiffness matrix and some aerodynamic geometry arrays
        aerostruct_group = AerostructGeometry(surface=surface)

        name = 'wing'

        # Add the group to the problem
        self.add_subsystem(name, aerostruct_group,
                           promotes_inputs=['load_factor'])

        point_name = 'AS_point_0'

        # Create the aerostruct point group and add it to the model.
        # This contains all the actual aerostructural analyses.
        AS_point = AerostructPoint(surfaces=[surface])

        self.add_subsystem(point_name, AS_point,
            promotes_inputs=['v', 'alpha', 'Mach_number', 're', 'rho', 'CT', 'R',
                'W0', 'speed_of_sound', 'empty_cg', 'load_factor'])

        # Issue quite a few connections within the model to make sure all of the
        # parameters are connected correctly.
        com_name = point_name + '.' + name + '_perf'
        self.connect(name + '.local_stiff_transformed', point_name + '.coupled.' + name + '.local_stiff_transformed')
        self.connect(name + '.nodes', point_name + '.coupled.' + name + '.nodes')

        # Connect aerodyamic mesh to coupled group mesh
        self.connect(name + '.mesh', point_name + '.coupled.' + name + '.mesh')

        # Connect performance calculation variables
        self.connect(name + '.radius', com_name + '.radius')
        self.connect(name + '.thickness', com_name + '.thickness')
        self.connect(name + '.nodes', com_name + '.nodes')
        self.connect(name + '.cg_location', point_name + '.' + 'total_perf.' + name + '_cg_location')
        self.connect(name + '.structural_weight', point_name + '.' + 'total_perf.' + name + '_structural_weight')
        self.connect(name + '.t_over_c', com_name + '.t_over_c')

        # prob.model.connect('mrho', name + '.struct_setup.structural_weight.mrho')
        # prob.model.connect('E', name + '.struct_setup.assembly.E')
        # prob.model.connect('G', name + '.struct_setup.assembly.G')
        # prob.model.connect('E', com_name + '.struct_funcs.vonmises.E')
        # prob.model.connect('G', com_name + '.struct_funcs.vonmises.G')
