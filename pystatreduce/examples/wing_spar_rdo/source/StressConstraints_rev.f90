subroutine StressConstraints_rev(x, x_b, xi, L, E, force, yield, Nelem, cineq_b)
    ! Computes the reverse mode of the nonlinear inequality constraints
    ! Inputs:
    !   x - the DVs; x(1:Nelem+1) inner and x(Nelem+2:2*(Nelem+1) outer radius
    !   xi - the uncertain force-perturbation coefficients
    !   L - length of the beam
    !   E - longitudinal elastic modulus
    !   force - the nominal force
    !   yield - the yield stress for the material
    !   Nelem - number of finite elements to use
    !   cineq_b - weights on inequality (stress) constraints
    ! Outputs:
    !   x_b - derivatives of cineq_b^T*cineq with respect to x
    !--------------------------------------------------------------------------
    implicit none
    double precision, intent(in) :: x(:), xi(:), L, E, force(:), yield, cineq_b(:)
    integer, intent(in) :: Nelem
    double precision, intent(out) :: x_b(:)
    double precision, allocatable :: pertforce(:), Iyy(:), Iyy_b(:)

    if ( (size(force) /= (Nelem+1)) .or. (size(x) /= (2*(Nelem+1))) ) then
        write (*,*) 'size of force/x inconsistent with Nelem'
    end
    allocate(pertforce(Nelem+1), Iyy(Nelem+1), Iyy_b(Nelem+1))

    ! need the displacements due to bilinear terms
    call CalcSecondMomentAnnulus(x(1:Nelem+1), x(Nelem+2:2*(Nelem+1)), Iyy)
    call CalcPertForce(force, xi, L, Nelem, pertforce)
    call CalcBeamDisplacement(L, E, Iyy, pertforce, Nelem, u)

    ! cineq(:) = cineq(:)/yield - 1.d0 ! do this at the end
    ! call CalcBeamStress(L, E, x(Nelem+2:2*(Nelem+1)), u, Nelem, cineq)
    x_b(:) = 0.d0
    u_b(:) = 0.d0
    call CalcBeamStress_rev(L, E, x(Nelem+2:2*(Nelem+1)), x_b(Nelem+2:2*(Nelem+1)), u, u_b, Nelem, &
                            cineq_b)
    !call CalcPertForce(force, xi, L, Nelem, pertforce)
    !call CalcBeamDisplacement(L, E, Iyy, pertforce, Nelem, u)
    Iyy_b(:) = 0.d0
    call CalcBeamDisplacement_rev(L, E, Iyy_b, pertforce, Nelem, u_b)
    !call CalcSecondMomentAnnulus(x(1:Nelem+1), x(Nelem+2:2*(Nelem+1)), Iyy)
    call CalcSecondMomentAnnulus_rev(x_b(1:Nelem+1), x_b(Nelem+2:2*(Nelem+1)), Iyy_b)
    x_b(:) = x_b(:)/yield ! did not do this earlier, so do it now
end
