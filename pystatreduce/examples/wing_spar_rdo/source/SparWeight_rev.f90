subroutine SparWeight_rev(x, L, rho, Nelem, dwdx)
    ! Reverse mode differentiated version of SparWeight
    ! Inputs:
    !   x - the DVs; x(1:Nelem+1) inner and x(Nelem+2:2*(Nelem+1) outer radius
    !   L - length of the beam
    !   rho - density of the metal alloy being used
    !   Nelem - number of elements
    ! Outputs:
    !   dwdx - derivative the weight of the spar w.r.t. w
    !--------------------------------------------------------------------------
    implicit none
    double precision, intent(in) :: x(:)
    double precision, intent(in) :: L, rho
    integer, intent(in) :: Nelem
    double precision, intent(out) :: dwdx(:)
    double precision :: r_in, r_out, fac
    integer :: i

    if ( (size(x) /= 2*(Nelem+1)) .or. (size(dwdx) /= 2*(Nelem+1)) ) then
        write (*,*) 'size of x and/or dwdx is inconsistent with Nelem'
    end
    fac = 3.14159265359d0*rho*L/Nelem
    r_in = x(1)
    r_out = x(Nelem+2)
    ! SparWeight = 0.5d0*(r_out**2 - r_in**2)*fac
    dwdx(1) = -r_in*fac
    dwdx(Nelem+2) = r_out*fac
    do i = 2,Nelem
        r_in = x(i)
        r_out = x(Nelem+i+1)
        ! SparWeight = SparWeight + (r_out**2 - r_in**2)*fac
        dwdx(i) = -2.d0*r_in*fac
        dwdx(Nelem+i+1) = 2.d0*r_out*fac
    end
    r_in = x(Nelem+1)
    r_out = x(2*(Nelem+1))
    ! SparWeight = 0.5d0*(r_out**2 - r_in**2)*fac
    dwdx(Nelem+1) = -r_in*fac
    dwdx(2*(Nelem+1)) = r_out*fac
end
