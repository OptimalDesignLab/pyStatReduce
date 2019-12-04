subroutine CalcSecondMomentAnnulus(r_inner, r_outer, Iyy)
    ! Computes the second-moment of area for an annular region
    ! Inputs:
    !   r_inner - the inner radius of the annular region
    !   r_outer - the outer radius of the annular region
    ! Outputs:
    !   Iyy - the second-moment of area
    !--------------------------------------------------------------------------
    implicit none
    double precision, intent(in) :: r_inner(:), r_outer(:)
    double precision, intent(out) :: Iyy(:)
    integer :: nNodes, i
    nNodes = size(r_inner)
    if ( (size(r_outer) /= nNodes) .or. (size(Iyy) /= nNodes) ) then
        write (*,*) 'size of r_inner/r_outer/Iyy inconsistent'
    end
    fac = 3.14159265359d0/4.d0
    do i = 1,nNodes
        Iyy(i) = fac*(r_outer(i)**4 - r_inner(i)**4)
    end
end
