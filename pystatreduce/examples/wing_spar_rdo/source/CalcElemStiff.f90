subroutine CalcElemStiff(E, IL, IR, dx, Aelem)
    ! Compute the element stiffness matrix using Hermitian cubic shape funcs.
    ! Inputs:
    !   E - longitudinal elastic modulus
    !   IL - moment of inertia at left side of element
    !   IR - moment of inertia at right side of element
    !   dx - length of the element
    ! Outputs:
    !   Aelem - the 4x4 element stiffness matrix
    !--------------------------------------------------------------------------
    implicit none
    double precision, intent(in) :: E, IL, IR, dx
    double precision, intent(inout) :: Aelem(4,4)
    double precision :: work(4,4), B(4)
    if ( (IL <= 0.0) .or. (IR <= 0.0) .or. (E <= 0.0) .or. (dx <= 0.0) )
        write (*,*) 'Inputs must all be strictly positive'
    end
    call GaussQuad(Integrad, 2, work, Aelem)

contains

    subroutine Integrad(xi, Int)
        ! compute the integrand needed for the element stiffness matrix
        double precision, intent(in) :: xi
        double precision, intent(out) :: Int(4,4)
        double precision :: fac
        integer :: i, j
        call D2HermiteBasis(xi, dx, B)
        fac = 0.5d0*E*dx*LinearMomentInertia(xi)
        do i = 1,4
            do j = 1,4
                Int(i,j) = fac*B(i)*B(j)
            end
        end
    end

    pure function LinearMomentInertia(xi)
        ! evaluate the linear moment of inertia at point xi
        double precision, intent(in) :: xi
        double precision :: LinearMomentInertia
        LinearMomentInertia = (IL*(1.d0-xi) + IR*(1.d0+xi))*0.5d0
    end

end
