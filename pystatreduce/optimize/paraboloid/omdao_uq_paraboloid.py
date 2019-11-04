# Use the OpenMDAO framework for OUU for optimizing a paraboloid
from __future__ import division, print_function
import os
import sys
import errno
sys.path.insert(0, '../../src')

# pyStatReduce specific imports
import numpy as np
import chaospy as cp
from stochastic_collocation import StochasticCollocation
from quantity_of_interest import QuantityOfInterest
from dimension_reduction import DimensionReduction
from stochastic_arnoldi.arnoldi_sample import ArnoldiSampling
import examples

# openMDAO specific imports
from openmdao.api import Problem, pyOptSparseDriver, SqliteRecorder, IndepVarComp, ExplicitComponent, Group
try:
    from openmdao.parallel_api import PETScVector
    vector = PETScVector
except:
    from openmdao.vectors.default_vector import DefaultVector
    vector = DefaultVector


class Paraboloid(ExplicitComponent):
    """
    Lets try OpenMDAO api for our problems
    """
    def setup(self):

        print("In Setup!!")

        self.std_dev_xi = np.array([0.3, 0.2, 0.1])
        system_size = 3
        mean_xi = np.ones(system_size)

        # Inputs
        self.add_input('mean_xi', val=mean_xi)

        # Intermediate operations
        jdist = cp.MvNormal(mean_xi, np.diag(self.std_dev_xi))
        self.collocation_QoI = StochasticCollocation(system_size, "Normal")
        self.collocation_QoI_grad = StochasticCollocation(system_size, "Normal", system_size) # Create a Stochastic collocation object
        self.QoI = examples.Paraboloid3D(system_size)          # Create QoI
        threshold_factor = 0.9
        self.dominant_space = DimensionReduction(threshold_factor, exact_Hessian=True)
        self.dominant_space.getDominantDirections(self.QoI, jdist)

        # Outputs
        self.add_output('mean_QoI', 0.0)

        # Partial derivatives
        self.declare_partials('mean_QoI', 'mean_xi')


    def compute(self, inputs, outputs):

        print("In compute")

        mu = inputs['mean_xi']
        # print("mu = ", mu)
        jdist = cp.MvNormal(mu, np.diag(self.std_dev_xi))
        QoI_func = self.QoI.eval_QoI
        outputs['mean_QoI'] = self.collocation_QoI.normal.reduced_mean(QoI_func, jdist, self.dominant_space)

    def compute_partials(self, inputs, J):

        print("In compute_partials")

        mu = inputs['mean_xi']
        jdist = cp.MvNormal(mu, np.diag(self.std_dev_xi))
        QoI_func = self.QoI.eval_QoIGradient
        val = self.collocation_QoI_grad.normal.reduced_mean(QoI_func, jdist, self.dominant_space)
        J['mean_QoI', 'mean_xi'] = val # self.collocation.normal.reduced_mean(QoI_func, jdist, self.dominant_space)


if __name__ == "__main__":

    #------------ Run model only
    # # Build the model
    # model = Group()
    # ivc = IndepVarComp()
    # ivc.add_output('mean_xi', 8*np.ones(3))
    # model.add_subsystem('des_vars', ivc)
    # model.add_subsystem('paraboloid', Paraboloid())
    # model.connect('des_vars.mean_xi', 'paraboloid.mean_xi')
    # prob = Problem(model)
    # prob.setup()
    # prob.run_model()
    # print(prob['paraboloid.mean_QoI'])
    # print(prob['paraboloid.mean_xi'])


    #------------ Run optimization using SNOPT
    prob = Problem()
    indeps = prob.model.add_subsystem('indeps', IndepVarComp(), promotes=['*'])
    indeps.add_output('mean_xi', 10*np.ones(3))
    prob.model.add_subsystem('paraboloid', Paraboloid(), promotes_inputs=['mean_xi'])

    # Set up the Optimization
    prob.driver = pyOptSparseDriver()
    prob.driver.options['optimizer'] = 'SNOPT'
    prob.driver.opt_settings['Major optimality tolerance'] = 3e-7
    prob.driver.opt_settings['Major feasibility tolerance'] = 3e-7
    prob.driver.opt_settings['Major iterations limit'] = 1000
    prob.driver.opt_settings['Verify level'] = -1
    prob.model.add_design_var('mean_xi', lower = -20*np.ones(3), upper = 20*np.ones(3))
    prob.model.add_objective('paraboloid.mean_QoI')

    prob.setup()
    prob.run_driver()
    print(prob['paraboloid.mean_QoI'])
    print(prob['paraboloid.mean_xi'])
