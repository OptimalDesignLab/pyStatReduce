import numpy as np
import numdifftools as nd
# import pyublas
import WingSpar as ws
# from kona.user import BaseVector, UserSolver

np.set_printoptions(linewidth=200)

# class SparSolver(UserSolver):
class SparSolver(object):

    def __init__(self, nelem, length=7.5, rho=1600.0, Young=70e9, Weight=0.5*500*9.8,
                 yield_stress=600e6, lb=0.01, up=0.05, minthick=0.0025):
        # super(SparSolver, self).__init__(
        #     2*(nelem+1), num_eq=0, num_nonlin_ineq=4*(nelem+1))
        # #super(SparSolver, self).__init__(
        # #      2*(nelem+1), num_eq=2*(nelem+1), num_nonlin_ineq=0)
        print("__init__")

        # Add replacements for replacing UserSolver from KONA
        self.num_design = 2*(nelem+1)
        self.num_state = 0 # Default within KONA is 0 and there are no keyword arguments either
        self.num_eq = 0
        self.num_nonlin_ineq = 4*(nelem+1)
        self.num_lin_ineq = nelem + 1

        ws.wingspar.initialize(nelem)
        self.nelem = nelem
        self.length = length
        self.rho = rho
        self.E = Young
        self.force = (2*(2.5*Weight)/(length**2))*np.linspace(length,0.0,nelem+1)
        self.yield_stress = yield_stress

        lb_arr = np.zeros(self.num_design)
        lb_arr[0:self.nelem+1] = lb
        lb_arr[self.nelem+1:] = minthick
        self.lb = lb_arr
        self.up = up
        self.minthick = minthick
        self.xi = np.zeros(0)

    def eval_obj(self, at_design, at_state=None):
        """
        Returns the estimated weight of the spar.
        """
        obj = ws.wingspar.sparweight(at_design, self.length, self.rho)

        #cineq = np.zeros(self.nelem+1)
        #cineq[:] = ws.wingspar.stressconstraints(at_design, self.xi, self.length, self.E,
        #                                         self.force, self.yield_stress)
        #cineq[:] = np.minimum(0.0, cineq)
        return obj #+ 0.5*np.dot(cineq, cineq)

    # def eval_eq_cnstr(self, at_design, at_state):
    #     """
    #     Temp: debugging
    #     """
    #     print("eval_eq_cnstr")
    #     ceq = np.zeros(self.num_eq)
    #     ceq[0:(self.nelem+1)] = at_design[0:(self.nelem+1)] - self.lb
    #     ceq[(self.nelem+1):2*(self.nelem+1)] = at_design[(self.nelem+1):] - self.minthick
    #     ceq[:] *= 100.0
    #     print('ceq = ',ceq[:])
    #     return ceq

    def eval_ineq_cnstr(self, at_design, at_state=None):
        """
        Evaluate the stress constraints, bound constraints, and linear constraints
        """
        # print("eval_ineq_cnstr")

        bounds_ok = True
        # for i in range(0,self.nelem):
        #     if at_design[i] < 1e-4 or at_design[self.nelem+1+i] < 1e-4:
        #         bounds_ok = False

        cineq = np.zeros(self.num_nonlin_ineq)
        if bounds_ok:
            cineq[0:(self.nelem+1)] = ws.wingspar.stressconstraints(at_design, self.xi, self.length,
                                                                    self.force, self.yield_stress)

        cineq[(self.nelem+1):2*(self.nelem+1)] = at_design[0:(self.nelem+1)] - self.lb[0:(self.nelem+1)]
        cineq[2*(self.nelem+1):3*(self.nelem+1)] = at_design[(self.nelem+1):] - self.minthick
        cineq[3*(self.nelem+1):4*(self.nelem+1)] = self.up - (at_design[(self.nelem+1):] +
                                                              at_design[0:(self.nelem+1)])
        cineq[0:(self.nelem+1)] *= 100.0
        cineq[(self.nelem+1):] *= 10.0
        # print('cineq = ',cineq[:])
        return cineq

    def factor_linear_system(self, at_design, at_state):
        """
        Build and factor the stiffness matrix (which is conveniently independent of xi here)
        """
        Iyy = np.zeros(self.nelem+1)
        Iyy = ws.wingspar.calcsecondmomentannulus(at_design[0:self.nelem+1],
                                                  at_design[0:self.nelem+1] +
                                                  at_design[self.nelem+1:])
        ws.wingspar.buildandfactorstiffness(self.length, self.E, Iyy)

    def multiply_dCINdX(self, at_design, at_state, in_vec):
        """
        perform the constraint-Jacobian-vector product
        """
        return None

    # def multiply_dCEQdX_T(self, at_design, at_state, in_vec):
    #     """
    #     For debugging
    #     """
    #     print("multiply_dEQdX_T")
    #     out_vec = np.zeros(self.num_design)
    #     print("at_design = ",at_design[:])
    #     print("in_vec = ",in_vec[:])
    #     out_vec[0:(self.nelem+1)] += in_vec[0:(self.nelem+1)]*100.0
    #     out_vec[(self.nelem+1):] += in_vec[(self.nelem+1):2*(self.nelem+1)]*100.0
    #     return out_vec

    def multiply_dCINdX_T(self, at_design, at_state, in_vec):
        """
        Perform the vector-constraint-Jacobian product
        """
        # print("multiply_dCINdX_T")
        out_vec = np.zeros(self.num_design)
        # print("at_design = ",at_design[:])
        # print("in_vec = ",in_vec[:])

        bounds_ok = True
        # for i in range(0,self.nelem):
        #     if at_design[i] < 1e-4 or at_design[self.nelem+1+i] < 1e-4:
        #         bounds_ok = False

        if bounds_ok:
            out_vec = ws.wingspar.stressconstraints_rev(at_design, self.xi, self.length,
                                                        self.force, self.yield_stress,
                                                        in_vec[0:(self.nelem+1)])*100.0
        # print("after stress constraints")
        out_vec[0:(self.nelem+1)] += in_vec[(self.nelem+1):2*(self.nelem+1)]*10.0
        # print('out_vec = ', out_vec)
        # print('in_vec[2*(self.nelem+1):3*(self.nelem+1)]*10.0 = \n', in_vec[2*(self.nelem+1):3*(self.nelem+1)]*10.0)
        # print('out_vec[(self.nelem+1):] = \n', out_vec[(self.nelem+1):])
        out_vec[(self.nelem+1):] += in_vec[2*(self.nelem+1):3*(self.nelem+1)]*10.0
        out_vec[0:(self.nelem+1)] -= in_vec[3*(self.nelem+1):4*(self.nelem+1)]*10.0
        out_vec[(self.nelem+1):] -= in_vec[3*(self.nelem+1):4*(self.nelem+1)]*10.0
        return out_vec

    def eval_dFdX(self, at_design, at_state=None):
        """
        Compute the objective gradient
        """
        out_vec = np.zeros(self.num_design)
        out_vec = ws.wingspar.sparweight_rev(at_design, self.length, self.rho)
        # out_vec /= 10.0

        # cineq = np.zeros(self.nelem+1)
        # cineq[:] = ws.wingspar.stressconstraints(at_design, self.xi, self.length, self.E,
        #                                       self.force, self.yield_stress)
        # cineq[:] = np.minimum(0.0, cineq)
        # out_vec += ws.wingspar.stressconstraints_rev(at_design, self.xi, self.length, self.E,
        #                                              self.force, self.yield_stress, cineq)
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
        x0[self.nelem+1:] = 0.00375*np.ones(self.nelem+1)
        #x0[0:self.nelem+1] = self.lb*np.ones(self.nelem+1)
        #x0[self.nelem+1:] = 0.04*np.ones(self.nelem+1)
        return x0

if __name__ == '__main__':

    spar_solver_obj = SparSolver(10) # 5 is the number of elements
    design_vars = spar_solver_obj.init_design()
    print('design_vars = \n', repr(design_vars))

    """
    obj_val  = spar_solver_obj.eval_obj(design_vars)
    print('obj_val = ', obj_val)
    # eq_constr_val = spar_solver_obj.eval_eq_cnstr(design_vars, state_vars)
    ineq_constr_val = spar_solver_obj.eval_ineq_cnstr(design_vars)
    print('ineq_con_val = \n', repr(ineq_constr_val))

    # Evaluate objective function gradient
    dfdx = spar_solver_obj.eval_dFdX(design_vars)
    print('dfdx = ', dfdx)

    # Evaluate the inequality constraint derivatives
    seed_vec = np.zeros(ineq_constr_val.size)
    seed_vec[0] = 1.0
    grad_col1 = spar_solver_obj.multiply_dCINdX_T(design_vars, None, seed_vec)
    print('grad_col1 = \n', grad_col1)

    dcin_dx = spar_solver_obj.eval_dCindX(design_vars)
    print('\ndcin_dx = \n', np.round(dcin_dx, 2))

    # Verify the constraint derivatives against finite-difference
    def func(x):
        return spar_solver_obj.eval_ineq_cnstr(x)

    jac = nd.Jacobian(func)(design_vars)
    print('\nnd jac = \n', np.round(jac,2))
    ctr = 0
    err = jac - dcin_dx
    for i in err.flatten():
        if abs(i) > 1.e-6:
            ctr += 1

    print('violations = ', ctr)
    """
