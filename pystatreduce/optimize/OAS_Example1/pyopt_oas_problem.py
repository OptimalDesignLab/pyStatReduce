# pyopt_oas_problem.py
# The following file contains the optimization of the deterministic problem shown
# in the quick example within OpenAeroStruct tutorial, the difference being,
# pyOptSparse is called separately, whereas the Quick example uses a driver.
import numpy as np
import chaospy as cp

# Import the OpenMDAo shenanigans
from openmdao.api import IndepVarComp, Problem, Group, NewtonSolver, \
    ScipyIterativeSolver, LinearBlockGS, NonlinearBlockGS, \
    DirectSolver, LinearBlockGS, PetscKSP, SqliteRecorder

from openaerostruct.geometry.utils import generate_mesh
from openaerostruct.geometry.geometry_group import Geometry
from openaerostruct.aerodynamics.aero_groups import AeroPoint
import pyoptsparse

class OASExample1Opt(object):
    """
    Trial test to see if I can ge the optimization running using my framework
    """
    def __init__(self):
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

        # Create the OpenMDAO problem
        self.prob = Problem()

        # Create an independent variable component that will supply the flow
        # conditions to the problem.
        indep_var_comp = IndepVarComp()
        indep_var_comp.add_output('v', val=248.136, units='m/s')
        indep_var_comp.add_output('alpha', val=5., units='deg')
        indep_var_comp.add_output('Mach_number', val=0.84)
        indep_var_comp.add_output('re', val=1.e6, units='1/m')
        indep_var_comp.add_output('rho', val=0.38, units='kg/m**3')
        indep_var_comp.add_output('cg', val=np.zeros((3)), units='m')

        # Add this IndepVarComp to the problem model
        self.prob.model.add_subsystem('prob_vars',
            indep_var_comp,
            promotes=['*'])

        # Create and add a group that handles the geometry for the
        # aerodynamic lifting surface
        geom_group = Geometry(surface=surface)
        self.prob.model.add_subsystem(surface['name'], geom_group)

        # Create the aero point group, which contains the actual aerodynamic
        # analyses
        aero_group = AeroPoint(surfaces=[surface])
        point_name = 'aero_point_0'
        self.prob.model.add_subsystem(point_name, aero_group,
            promotes_inputs=['v', 'alpha', 'Mach_number', 're', 'rho', 'cg'])

        name = surface['name']

        # Connect the mesh from the geometry component to the analysis point
        self.prob.model.connect(name + '.mesh', point_name + '.' + name + '.def_mesh')

        # Perform the connections with the modified names within the
        # 'aero_states' group.
        self.prob.model.connect(name + '.mesh', point_name + '.aero_states.' + name + '_def_mesh')
        self.prob.model.connect(name + '.t_over_c', point_name + '.' + name + '_perf.' + 't_over_c')
        self.prob.setup()
        # print "wing.twist_cp",  self.prob['wing.twist_cp']


def objfunc(xdict):
    dv = xdict['xvars']
    QoI.prob['wing.twist_cp'] = dv
    QoI.prob.run_model()
    funcs = {}
    funcs['obj'] = QoI.prob['aero_point_0.wing_perf.CD']
    conval = np.zeros(1)
    conval[0] = QoI.prob['aero_point_0.wing_perf.CL']
    funcs['con'] = conval
    fail = False
    return funcs, fail

def sens(xdict, funcs):
    dv = xdict['xvars']
    QoI.prob['wing.twist_cp'] = dv
    QoI.prob.run_model()
    deriv = QoI.prob.compute_totals(of=['aero_point_0.wing_perf.CD', 'aero_point_0.wing_perf.CL'], wrt=['wing.twist_cp'])
    funcsSens = {}
    funcsSens['obj', 'xvars'] = deriv['aero_point_0.wing_perf.CD', 'wing.twist_cp']
    funcsSens['con', 'xvars'] = deriv['aero_point_0.wing_perf.CL', 'wing.twist_cp']
    fail = False
    return funcsSens, fail

if __name__ == "__main__":

    # Lets do this
    QoI = OASExample1Opt()
    optProb = pyoptsparse.Optimization('OASExample1', objfunc)
    optProb.addVarGroup('xvars', 5, 'c', lower=-10., upper=15.)
    optProb.addConGroup('con', 1, lower=0.5, upper=0.5)
    optProb.addObj('obj')
    # print optProb
    # opt = pyoptsparse.SLSQP(tol=1.e-9)
    opt = pyoptsparse.SNOPT(optOptions = {'Major feasibility tolerance' : 1e-10})
    sol = opt(optProb, sens=sens)
    print sol
