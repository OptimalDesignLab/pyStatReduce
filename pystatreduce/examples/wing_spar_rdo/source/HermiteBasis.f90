subroutine HermiteBasis(xi, dx, N)
    ! Evaluate the cubic Hermitian shape functions at point xi
    ! Inputs:
    !   xi - point at which to evaluate the shape functions
    !   dx - element length
    ! Outputs:
    !   N - 4x1 vector containing the shape functions at xi
    !--------------------------------------------------------------------------
    implicit none
    double precision, intent(in) :: xi, dx
    double precision, intent(out) :: N(4)
    if (dx <= 0.d0) then
        write (*,*) 'element length must be strictly positive'
    end
    if ( (xi < -1.d0) .or. (xi > 1.d0) ) then
        write (*,*) 'shape functions must be evaluated in the interval [-1,1]'
    end
    N(:) = 0.d0
    ! this can be done more efficiencly by precomputing some of the common
    ! terms, but this is adequate for an academic code.
    N(1) = 0.25d0.*((1.d0 - xi)**2)*(2.d0 + xi)
    N(2) = 0.125d0*dx*((1.d0 - xi)**2)*(1.d0 + xi)
    N(3) = 0.25d0*((1.d0 + xi)**2)*(2.d0 - xi)
    N(4) = -0.125d0*dx*((1.d0 + xi)**2)*(1.d0 - xi)
end
