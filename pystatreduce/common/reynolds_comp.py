from __future__ import print_function
from openmdao.api import ExplicitComponent


class ReynoldsComp(ExplicitComponent):

    def setup(self):
        self.add_input('rho', val=1., units='kg/m**3')
        self.add_input('mu', val=1., units='kg * m **-1 / s')
        self.add_input('v', val=1., units='m/s')

        self.add_output('re', val=1., units='1/m')

        self.declare_partials('re', ['rho', 'mu', 'v'])

    def compute(self, inputs, outputs):
        outputs['re'] = inputs['rho'] * inputs['v'] / inputs['mu']

    def compute_partials(self, inputs, partials):
        partials['re', 'rho'] = inputs['v'] / inputs['mu']
        partials['re', 'v'] = inputs['rho'] / inputs['mu']
        partials['re', 'mu'] = -inputs['rho'] * inputs['v'] / inputs['mu']**-2
