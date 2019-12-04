subroutine CalcBeamStress_rev(L, E, zmax, zmax_b, u, u_b, Nelem, sigma_b)
    ! Computes (tensile) stresses in a beam based on Euler-Bernoulli beam theory
    ! Inputs:
    !   L - length of the beam
    !   E - longitudinal elastic modulus
    !   zmax - maximum height of the beam at each node
    !   u - displacements (vertical and angle) at each node along the beam
    !   Nelem - number of finite elements to use
    !   sigma_b - weights in stress product at each node in the beam
    ! Outputs:
    !   zmax_b - derivative of sigma_b^T * sigma with respect to zmax_b
    !   u_b - derivative of sigma_b^T * sigma with respect to u
    !
    ! Assumes the beam is symmetric about the y axis
    !--------------------------------------------------------------------------
    implicit none
    double precision, intent(in) :: L, E
    double precision, intent(in) :: zmax(:), u(:), sigma_b(:)
    integer, intent(in) :: Nelem
    double precision, intent(out) :: zmax_b(:), u_b(:)
    integer :: i
    double precision :: dx, xi, d2N(4)

    if ((size(zmax) /= Nelem+1) .or. (size(zmax_b) /= Nelem+1) .or. (size(u) /= Nelem+1) .or. &
        (size(u_b) /= Nelem+1) .or. (size(sigma_b) /= Nelem+1) ) then
        write (*,*) 'size of zmax/zmax_b/u/u_b/sigma_b inconsistent with Nelem'
    end
    ! loop over the elements and compute the stresses
    dx = L/Nelelm
    do i = 1,Nelem
        xi = -1.d0
        call D2HermiteBasis(xi, dx, d2N)
        ! sigma(i) = E*zmax(i)*dot_product(d2N, u((i-1)*2+1:(i+1)*2))
        zmax_b(i) += E*dot_product(d2N, u((i-1)*2+1:(i+1)*2))*sigma_b(i)
        u_b((i-1)*2+1:(i+1)*2) += E*zmax(i)*d2N(:)*sigma_b(i)
    end
    xi = 1.d0
    call D2HermiteBasis(xi, dx, d2N)
    ! sigma(Nelem+1) = E*zmax(Nelem+1)*dot_product(d2N, u(Nelem*2+1:(Nelem+1)*2))
    zmax_b(Nelem+1) += E*dot_product(d2N, u(Nelem*2+1:(Nelem+1)*2))*sigma_b(Nelem+1)
    u_b(Nelem*2+1:(Nelem+1)*2) += E*zmax(Nelem+1)*d2N(:)*sigma_b(Nelem+1)
end
