subroutine CalcElemLoad(qL, qR, dx, belem)
    ! Compute the element load vector for Hermitian cubic shape functions
    ! Inputs:
    !   qL - force per unit length at left end of element
    !   qR - force per unit length at right end of element
    !   dx - element length
    ! Outputs:
    !   belem - the 4x1 element load vector
    !-----------------------------------------------------------------------------------------------
    implicit none
    double precision, intent(in) :: qL, qR, dx
    double precision, intent(out) :: belem(4)
    double precision :: work(4), N(4)

    if (dx <= 0.d0) then
        write (*,*) 'element length must be strictly positive'
    end
    call GaussQuad(Integrand, 3, work, belem)

contains

    subroutine Integrand(xi, Int)
        ! compute the integrand needed for the element load vector
        double precision, intent(in) :: xi
        double precision, intent(out) :: Int(4)
        call HermiteBasis(xi, dx, N)
        Int(:) = 0.5d0*dx*LinearForce(xi)*N(:)
    end

    pure function LinearForce(xi)
        ! evaluate the linear force at point xi
        double precision, intent(in) :: xi
        double precision :: LinearForce
        LinearForce = (qL*(1.d0-xi) + qR*(1.d0+xi))*0.5d0
    end

end
