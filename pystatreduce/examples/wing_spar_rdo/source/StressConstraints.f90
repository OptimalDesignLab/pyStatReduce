subroutine StressConstraints(x, xi, L, E, force, yield, Nelem, cineq)
    ! Computes the nonlinear inequality constraints for the wing-spar problem
    ! Inputs:
    !   x - the DVs; x(1:Nelem+1) inner and x(Nelem+2:2*(Nelem+1) outer radius
    !   xi - the uncertain force-perturbation coefficients
    !   L - length of the beam
    !   E - longitudinal elastic modulus
    !   force - the nominal force
    !   yield - the yield stress for the material
    !   Nelem - number of finite elements to use
    ! Outputs:
    !   cineq - inequality (stress) constraints
    !--------------------------------------------------------------------------
    implicit none
    double precision, intent(in) :: x(:), xi(:), L, E, force(:), yield
    integer, intent(in) :: Nelem
    double precision, intent(out) :: cineq(:)
    double precision, allocatable :: pertforce(:), Iyy(:)

    if ( (size(force) /= (Nelem+1)) .or. (size(x) /= (2*(Nelem+1))) ) then
        write (*,*) 'size of force/x inconsistent with Nelem'
    end
    allocate(pertforce(Nelem+1), Iyy(Nelem+1))
    call CalcSecondMomentAnnulus(x(1:Nelem+1), x(Nelem+2:2*(Nelem+1)), Iyy)
    call CalcPertForce(force, xi, L, Nelem, pertforce)
    call CalcBeamDisplacement(L, E, Iyy, pertforce, Nelem, u)
    call CalcBeamStress(L, E, x(Nelem+2:2*(Nelem+1)), u, Nelem, cineq)
    cineq(:) = cineq(:)/yield - 1.d0
end
