import numpy as np
import numdifftools as nd
# import pyublas
import WingSpar as ws
# from kona.user import BaseVector, UserSolver

np.set_printoptions(linewidth=200)

# class SparSolver(UserSolver):
class SparSolver(object):

    def __init__(self, nelem, length=7.5, rho=1600.0, Young=70e9, Weight=0.5*500*9.8,
                 yield_stress=600e6, lb=0.01, up=0.05, minthick=0.0025, data_type=np.float):

        self.data_type = data_type # For complexifying code

        # Add replacements for replacing UserSolver from KONA
        self.num_design = 2*(nelem+1)
        self.num_state = 0 # Default within KONA is 0 and there are no keyword arguments either
        self.num_eq = 0
        self.num_nonlin_ineq = nelem+1 # 4*(nelem+1)
        self.num_lin_ineq = nelem + 1
        print('self.num_nonlin_ineq = ', self.num_nonlin_ineq)

        self.nelem = nelem
        self.length = length
        self.rho = rho
        self.E = Young
        self.yield_stress = yield_stress

        # idiot-proofing
        assert length > 0, "Length cannot be negative"
        assert rho > 0, "density cannot be negative"
        assert Young > 0, "Young's modulus cannot be negative"
        assert yield_stress > 0, "yield stress cannot be negative"

        self.force = (2*(2.5*Weight)/(length**2))*np.linspace(length,0.0,nelem+1, dtype=self.data_type)

        lb_arr = np.zeros(self.num_design, dtype=self.data_type)
        lb_arr[0:self.nelem+1] = lb
        lb_arr[self.nelem+1:] = minthick
        self.lb = lb # lb_arr
        self.up = up
        self.minthick = minthick
        self.xi = np.zeros(self.num_nonlin_ineq, dtype=self.data_type)

    def eval_obj(self, at_design, at_state=None):
        """
        Returns the estimated weight of the spar.
        """
        r_in = at_design[0:self.nelem+1]
        r_out = at_design[self.nelem+1:]
        y = r_out**2 - r_in**2
        obj = np.trapz(y)*np.pi*self.rho*self.length / self.nelem

        return obj

    def eval_stress_constraint(self, at_design, at_state=None):
        """
        Evaluate the stress constraints.
        """
        r_in = at_design[0:self.nelem+1]
        r_out = at_design[self.nelem+1:]

        Iyy = self.calc_second_moment_annulus(r_in, r_out)
        # print('Iyy = ', Iyy)
        pertforce = self.calc_pert_force()
        # print('pertforce = \n', pertforce)
        M = self.calcBeamMoment(pertforce)
        # print('M = \n', M)

        cineq = (M * r_out)/self.yield_stress - Iyy

        return cineq

    def calc_second_moment_annulus(self, r_in, r_out):
        # assert r_in.real.all() < r_out.real.all() # They should always be numpy arrays
        return np.pi * (r_out**4 - r_in**4) / 4

    def calc_pert_force(self):
        pertforce = np.zeros(self.xi.size, dtype=self.data_type)
        pertforce[:] = self.force
        y = np.linspace(0,self.length, self.num_nonlin_ineq, dtype=self.data_type)
        for i in range(0, self.xi.size):
            pertforce += self.xi[i]*np.cos((2*i-1)*np.pi*y/(2*self.length))

        return pertforce

    def calcBeamMoment(self, force):
        M = np.zeros(self.num_nonlin_ineq, dtype=self.data_type)
        for i in range(0, self.num_nonlin_ineq):
            x = i*self.length / self.nelem
            M[i] = force[0] * ((0.5* x**2 - x**3 / (6*self.length)) - 0.5*self.length*x + self.length**2/6)

        return M

    def eval_dFdX(self, at_design, at_state=None):
        """
        Compute the objective gradient
        """
        out_vec = np.zeros(self.num_design)
        out_vec = ws.wingspar.sparweight_rev(at_design, self.length, self.rho)

        return out_vec

    def eval_dCindX(self, at_design, at_state=None):
        """
        Compute the inequality constraint jacobian for the optimizer.
        """
        seed_matrix = np.eye(self.num_nonlin_ineq)
        output_matrix = np.zeros([self.num_design, self.num_nonlin_ineq])
        for i in range(self.num_nonlin_ineq):
            output_matrix[:,i] = self.multiply_dCINdX_T(at_design, at_state, seed_matrix[:,i])

        return output_matrix.T

    def init_design(self):
        """
        Set the initial spar radii distribution
        """
        x0 = np.zeros(self.num_design)
        x0[0:self.nelem+1] = 0.04625*np.ones(self.nelem+1)
        x0[self.nelem+1:] = x0[0:self.nelem+1] + 0.00375*np.ones(self.nelem+1)

        return x0

if __name__ == '__main__':

    spar_solver_obj = SparSolver(20) # 5 is the number of elements
    design_vars = spar_solver_obj.init_design()
    print('design_vars = \n', repr(design_vars))

    obj_val  = spar_solver_obj.eval_obj(design_vars)
    print('obj_val = ', obj_val)
    stress_con_val = spar_solver_obj.eval_stress_constraint(design_vars)
    print('stress_con_val = \n', repr(stress_con_val))

    matlab_stress_val = np.array([-3.90381453117098e-08,
                                  -2.21033588020043e-07,
                                  -3.84845436978377e-07,
                                  -5.31430723436710e-07,
                                  -6.61746478645043e-07,
                                  -7.76749733853377e-07,
                                  -8.77397520311710e-07,
                                  -9.64646869270043e-07,
                                  -1.03945481197838e-06,
                                  -1.10277837968671e-06,
                                  -1.15557460364504e-06,
                                  -1.19880051510338e-06,
                                  -1.23341314531171e-06,
                                  -1.26036952552004e-06,
                                  -1.28062668697838e-06,
                                  -1.29514166093671e-06,
                                  -1.30487147864504e-06,
                                  -1.31077317135338e-06,
                                  -1.31380377031171e-06,
                                  -1.31492030677004e-06,
                                  -1.31507981197838e-06])

    err = stress_con_val - matlab_stress_val
    print('\nerr = \n', err)
    # # Evaluate objective function gradient
    # dfdx = spar_solver_obj.eval_dFdX(design_vars)
    # print('dfdx = ', dfdx)

    # # Evaluate the inequality constraint derivatives
    # seed_vec = np.zeros(ineq_constr_val.size)
    # seed_vec[0] = 1.0
    # grad_col1 = spar_solver_obj.multiply_dCINdX_T(design_vars, None, seed_vec)
    # print('grad_col1 = \n', grad_col1)

    # dcin_dx = spar_solver_obj.eval_dCindX(design_vars)
    # print('\ndcin_dx = \n', np.round(dcin_dx, 2))
