# Use the OpenMDAO framework for OUU for optimizing a Rosenbrock function
from __future__ import division, print_function
# import os
# import sys
# import errno
# sys.path.insert(0, '../../src')

# pyStatReduce specific imports
import numpy as np
import chaospy as cp
from pystatreduce.stochastic_collocation import StochasticCollocation
from pystatreduce.quantity_of_interest import QuantityOfInterest
from pystatreduce.dimension_reduction import DimensionReduction
from pystatreduce.stochastic_arnoldi.arnoldi_sample import ArnoldiSampling
import pystatreduce.examples

#pyoptsparse sepecific imports
from scipy import sparse
import argparse
from pyoptsparse import Optimization, OPT, SNOPT

# Import the OpenMDAo shenanigans
from openmdao.api import IndepVarComp, Problem, Group, NewtonSolver, \
    ScipyIterativeSolver, LinearBlockGS, NonlinearBlockGS, \
    DirectSolver, LinearBlockGS, PetscKSP, SqliteRecorder, ScipyOptimizeDriver

from openaerostruct.geometry.utils import generate_mesh
from openaerostruct.geometry.geometry_group import Geometry
from openaerostruct.aerodynamics.aero_groups import AeroPoint

class OASAerodynamicWrapper(QuantityOfInterest):
    """
    Wrapper class that sets up the OpenMDAO class of the first OAS problem
    """
    def __init__(self, systemsize, rv_dict):
        QuantityOfInterest.__init__(self, systemsize)
        # self.input_dict = input_dict # TODO: Add this feature
        # Default recommended values for the problem
        mean_v = 248.136 # Mean value of input random variable
        mean_alpha = 5   #
        mean_Ma = 0.84
        mean_re = 1.e6
        mean_rho = 0.38
        mean_cg = np.zeros((3))

        self.p = Problem() # Create problem object
        self.rvs = self.p.model.add_subsystem('random_variables', IndepVarComp(), promotes_outputs=['*'])
        self.rvs.add_output('mu', val=np.zeros(len(rv_dict)))
        self.p.model.add_subsystem('oas_example1', OASAerodynamic(rv_dict=rv_dict))

        # Figure out what are the random variables
        ctr = 0
        self.rv_array = np.array([])
        rv_indices = []
        if "v" in rv_dict:
            self.rv_array = np.append(self.rv_array, mean_v)
            rv_indices.append(ctr)
            self.p.model.connect('mu', 'oas_example1.v', src_indices=[ctr])
            ctr += 1
        if "alpha" in rv_dict:
            self.rv_array = np.append(self.rv_array, mean_alpha)
            rv_indices.append(ctr)
            self.p.model.connect('mu', 'oas_example1.alpha', src_indices=[ctr])
            ctr += 1
        if "Mach_number" in rv_dict:
            self.rv_array = np.append(self.rv_array, mean_Ma)
            rv_indices.append(ctr)
            self.p.model.connect('mu', 'oas_example1.Mach_number', src_indices=[ctr])
            ctr += 1
        if "re" in rv_dict:
            self.rv_array = np.append(self.rv_array, mean_re)
            rv_indices.append(ctr)
            self.p.model.connect('mu', 'oas_example1.re', src_indices=[ctr])
            ctr += 1
        if "rho" in rv_dict:
            self.rv_array = np.append(self.rv_array, mean_rho)
            rv_indices.append(ctr)
            self.p.model.connect('mu', 'oas_example1.rho', src_indices=[ctr])
            ctr += 1

        self.p.setup(force_alloc_complex=True)
        self.p['mu'] = self.rv_array

    def eval_QoI(self, mu, xi):
        rv = mu + xi
        self.p['mu'] = rv
        self.p.run_model()
        return self.p['oas_example1.aero_point_0.CD']

    def eval_QoIGradient(self, mu, xi):
        rv = mu + xi
        self.p['mu'] = rv
        self.p.run_model()
        self.deriv = self.p.compute_totals(of=['oas_example1.aero_point_0.CD',
                'oas_example1.aero_point_0.CL', 'oas_example1.aero_point_0.CM'], wrt=['mu'])

        return self.deriv['oas_example1.aero_point_0.CD', 'mu'][0]

    def eval_ObjGradient(self, mu, xi):
        rv = mu + xi
        self.p['mu'] = rv
        self.p.run_model()
        self.deriv = self.p.compute_totals(of=['oas_example1.aero_point_0.CD'],
                                           wrt=['oas_example1.wing.twist_cp'])
        return self.deriv['oas_example1.aero_point_0.CD', 'oas_example1.wing.twist_cp'][0]

    def eval_ConstraintQoI(self, mu, xi):
        rv = mu + xi
        self.p['mu'] = rv
        self.p.run_model()
        return self.p['oas_example1.aero_point_0.CL']

    def eval_ConstraintQoIGradient(self, mu, xi):
        rv = mu + xi
        self.p['mu'] = rv
        self.p.run_model()
        self.deriv = self.p.compute_totals(of=['oas_example1.aero_point_0.CL'],
                                           wrt=['oas_example1.wing.twist_cp'])
        return self.deriv['oas_example1.aero_point_0.CL', 'oas_example1.wing.twist_cp'][0]

#-------------------------------------------------------------------------------
class OASAerodynamic(Group):
    # First example from OpenAeroStruct packaged into a class
    def initialize(self):
        self.options.declare('rv_dict', types=dict)

    def setup(self):
        rv_dict = self.options['rv_dict']

        # Create a dictionary to store options about the mesh
        mesh_dict = {'num_y' : 7,
                     'num_x' : 2,
                     'wing_type' : 'CRM',
                     'symmetry' : True,
                     'num_twist_cp' : 5}

        # Generate the aerodynamic mesh based on the previous dictionary
        mesh, twist_cp = generate_mesh(mesh_dict)

        # Create a dictionary with info and options about the aerodynamic
        # lifting surface
        surface = {
                    # Wing definition
                    'name' : 'wing',        # name of the surface
                    'type' : 'aero',
                    'symmetry' : True,     # if true, model one half of wing
                                            # reflected across the plane y = 0
                    'S_ref_type' : 'wetted', # how we compute the wing area,
                                             # can be 'wetted' or 'projected'
                    'fem_model_type' : 'tube',

                    'twist_cp' : twist_cp,
                    'mesh' : mesh,
                    'num_x' : mesh.shape[0],
                    'num_y' : mesh.shape[1],

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
                    't_over_c_cp' : np.array([0.15]),      # thickness over chord ratio (NACA0015)
                    'c_max_t' : .303,       # chordwise location of maximum (NACA0015)
                                            # thickness
                    'with_viscous' : True,  # if true, compute viscous drag
                    'with_wave' : False,     # if true, compute wave drag
                    }

        indep_var_comp = IndepVarComp()
        if "v" not in rv_dict:
            indep_var_comp.add_output('v', val=248.136, units='m/s')
        if "alpha" not in rv_dict:
            indep_var_comp.add_output('alpha', val=5., units='deg')
        if "Mach_number" not in rv_dict:
            indep_var_comp.add_output('M', val=0.84)
        if "re" not in rv_dict:
            indep_var_comp.add_output('re', val=1.e6, units='1/m')
        if "rho" not in rv_dict:
            indep_var_comp.add_output('rho', val=0.38, units='kg/m**3')

        indep_var_comp.add_output('cg', val=np.zeros((3)), units='m')

        # Add this IndepVarComp to the problem model
        self.add_subsystem('prob_vars',
            indep_var_comp,
            promotes=['*'])

        # Create and add a group that handles the geometry for the
        # aerodynamic lifting surface
        geom_group = Geometry(surface=surface)
        self.add_subsystem(surface['name'], geom_group)

        # Create the aero point group, which contains the actual aerodynamic
        # analyses
        aero_group = AeroPoint(surfaces=[surface])
        point_name = 'aero_point_0'
        self.add_subsystem(point_name, aero_group,
            promotes_inputs=['v', 'alpha', 'Mach_number', 're', 'rho', 'cg'])

        name = surface['name']

        # Connect the mesh from the geometry component to the analysis point
        self.connect(name + '.mesh', point_name + '.' + name + '.def_mesh')

        # Perform the connections with the modified names within the
        # 'aero_states' group.
        self.connect(name + '.mesh', point_name + '.aero_states.' + name + '_def_mesh')
        self.connect(name + '.t_over_c', point_name + '.' + name + '_perf.' + 't_over_c')
