subroutine CalcSecondMomentAnnulus_rev(r_inner, r_inner_b, r_outer, r_outer_b, Iyy_b)
    ! reverse mode of the second-moment of area for an annular region
    ! Inputs:
    !   r_inner - the inner radius of the annular region
    !   r_outer - the outer radius of the annular region
    !   Iyy_b - weight on Iyy product
    ! Outputs:
    !   r_inner_b - derivative of Iyy_b^T*Iyy with respect to r_inner
    !   r_outer_b - derivative of Iyy_b^T*Iyy with respect to r_outer
    !--------------------------------------------------------------------------
    implicit none
    double precision, intent(in) :: r_inner(:), r_outer(:), Iyy_b(:)
    double precision, intent(inout) :: r_inner_b(:), r_outer_b(:)
    integer :: nNodes, i
    nNodes = size(r_inner)
    if ( (size(r_outer) /= nNodes) .or. (size(Iyy) /= nNodes) ) then
        write (*,*) 'size of r_inner/r_outer/Iyy inconsistent'
    end
    fac = 3.14159265359d0/4.d0
    do i = 1,nNodes
        ! Iyy(i) = fac*(r_outer(i)**4 - r_inner(i)**4)
        r_outer_b(i) = r_outer_b(i) + Iyy_b(i)*4.d0*fac*r_outer(i)**3
        r_inner_b(i) = r_inner_b(i) - Iyy_b(i)*4.d0*fac*r_inner(i)**3
    end
end
