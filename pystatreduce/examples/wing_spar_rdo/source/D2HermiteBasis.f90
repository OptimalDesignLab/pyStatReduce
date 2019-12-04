subroutine D2HermiteBasis(xi, dx, B)
    ! Evaluate the second derivative of the Hermitian functions at point xi
    ! Inputs:
    !   xi - point at which to evaluate the shape functions
    !   dx - element length
    ! Outputs:
    !   B - 4x1 vector containing the shape function derivatives at xi
    !--------------------------------------------------------------------------
    implicit none
    double precision, intent(in) :: xi, dx
    double precision, intent(out) :: B(4)
    if (dx <= 0.d0) then
        write (*,*) 'element length must be strictly positive'
    end
    if ( (xi < -1.d0) .or. (xi > 1.d0) ) then
        write (*,*) 'shape functions must be evaluated in the interval [-1,1]'
    end
    B(:) = 0.d0
    B(1) = 6.d0*xi/dx
    B(2) = 3.d0*xi - 1.d0
    B(3) = -6.d0*xi/dx
    B(4) = 3.d0*xi + 1.d0
    B(:) = B(:)/dx
end
