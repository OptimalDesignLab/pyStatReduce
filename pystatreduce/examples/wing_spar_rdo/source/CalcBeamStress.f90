subroutine CalcBeamStress(L, E, zmax, u, Nelem, sigma)
    ! Computes (tensile) stresses in a beam based on Euler-Bernoulli beam theory
    ! Inputs:
    !   L - length of the beam
    !   E - longitudinal elastic modulus
    !   zmax - maximum height of the beam at each node
    !   u - displacements (vertical and angle) at each node along the beam
    !   Nelem - number of finite elements to use
    ! Outputs:
    !   sigma - stress at each node in the beam
    !
    ! Assumes the beam is symmetric about the y axis
    !--------------------------------------------------------------------------
    implicit none
    double precision, intent(in) :: L, E
    double precision, intent(in) :: zmax(:), u(:)
    integer, intent(in) :: Nelem
    double precision, intent(out) :: sigma(:)
    integer :: i
    double precision :: dx, xi, d2N(4)

    if ((size(zmax) /= Nelem+1) .or. (size(u) /= Nelem+1) .or. (size(sigma) /= Nelem+1) ) then
        write (*,*) 'size of zmax/u/sigma inconsistent with Nelem'
    end
    ! loop over the elements and compute the stresses
    sigma(:) = 0.d0
    dx = L/Nelelm
    do i = 1,Nelem
        xi = -1.d0
        call D2HermiteBasis(xi, dx, d2N)
        sigma(i) = E*zmax(i)*dot_product(d2N, u((i-1)*2+1:(i+1)*2))
    end
    xi = 1.d0
    call D2HermiteBasis(xi, dx, d2N)
    sigma(Nelem+1) = E*zmax(Nelem+1)*dot_product(d2N, u(Nelem*2+1:(Nelem+1)*2))
end
