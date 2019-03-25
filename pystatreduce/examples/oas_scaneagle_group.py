# oas_scaneagle_group.py
# The following code consists of a OpenMDAO group that contains the crux the
# OpnenAeroStruct ScanEagle model. This group needs to be connected to a component
# that contains the random variables for performing uncertainty propagation.
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

class OASScanEagle(Group):
    """
    This is the OpenMDAO Group that get wraps in the class above for doing RDO
    under uncertainty.
    """
    def initialize(self):
        self.options.declare('mesh_dict', types=dict)
        self.options.declare('rv_dict', types=dict)

    def setup(self):
        # Total number of nodes to use in the spanwise (num_y) and
        # chordwise (num_x) directions. Vary these to change the level of fidelity.

        mesh_dict = self.options['mesh_dict']
        rv_dict = self.options['rv_dict']
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
        # indep_var_comp.add_output('re', val=1.e6, units='1/m')
        # indep_var_comp.add_output('rho', val=0.770816, units='kg/m**3')
        # indep_var_comp.add_output('speed_of_sound', val=322.2, units='m/s')
        indep_var_comp.add_output('empty_cg', val=np.array([0.2, 0., 0.]), units='m')

        # Create independent input variables depending on which variables are
        # being considered as random variables
        if 'Mach_number' not in rv_dict:
            indep_var_comp.add_output('Mach_number', val=0.071)
        if 'CT' not in rv_dict:
            indep_var_comp.add_output('CT', val=9.80665 * 8.6e-6, units='1/s')
        if 'W0' not in rv_dict:
            indep_var_comp.add_output('W0', val=10.0,  units='kg')
        if 'R' not in rv_dict:
            indep_var_comp.add_output('R', val=1800, units='km')
        if 'load_factor' not in rv_dict:
            indep_var_comp.add_output('load_factor', val=1.)
        if 'E' not in rv_dict:
            indep_var_comp.add_output('E', val=85.e9, units='N/m**2')
        if 'G' not in rv_dict:
            # indep_var_comp.add_output('G', val=25.e9, units='N/m**2')
            indep_var_comp.add_output('G', val=5.e9, units='N/m**2')
        if 'mrho' not in rv_dict:
            indep_var_comp.add_output('mrho', val=1600, units='kg/m**3')
        if 'altitude' not in rv_dict:
            indep_var_comp.add_output('altitude', val=4.57, units='km')

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

        # Make connections based on whether a variable is a random variable or not
        if 'E' not in rv_dict:
            self.connect('E', com_name + '.struct_funcs.vonmises.E')
            self.connect('E', name + '.struct_setup.assembly.E')
        if 'G' not in rv_dict:
            self.connect('G', com_name + '.struct_funcs.vonmises.G')
            self.connect('G', name + '.struct_setup.assembly.G')
        if 'mrho' not in rv_dict:
            self.connect('mrho', name + '.struct_setup.structural_weight.mrho')
        if 'load_factor' not in rv_dict:
            self.connect('load_factor', point_name + '.coupled.' + name + '.load_factor')
