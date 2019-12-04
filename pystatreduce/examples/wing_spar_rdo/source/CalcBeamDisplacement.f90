subroutine CalcBeamDisplacement(L, E, Iyy, force, Nelem, u)
    ! Estimate beam displacements using Euler-Bernoulli beam theory
    ! Inputs:
    !   L - length of the beam
    !   E - longitudinal elastic modulus
    !   Iyy - moment of inertia with respect to the y axis, as function of x
    !   force - force per unit length along the beam axis x
    !   Nelem - number of finite elements to use
    ! Outputs:
    !   u - displacements (vertical and angle) at each node along the beam
    !
    ! Uses a cubic Hermitian finite-element basis to solve the Euler-Bernoulli
    ! beam equations.  The beam is assumed to lie along the x axis, with the
    ! force applied transversely in the xz plane.
    !--------------------------------------------------------------------------
    implicit none
    double precision, intent(in) :: L, E
    double precision, intent(in) :: Iyy(:), force(:)
    integer, intent(in) :: Nelem
    double precision, intent(out) :: u(:)
    integer :: i, info
    double precision :: dx, belem(4), Aelem(4,4)
    integer, allocatable :: ipiv(:)
    double precision, allocatable :: A(:,:), b(:)

    if ( (L <= 0) .or. (E <= 0) .or. (Nelem <= 0) ) then
        write (*,*) 'The inputs L, E, and Nelem must all be positive'
    end
    if ( (size(Iyy) /= Nelem+1) .or. (size(force) /= (Nelem+1)) .or. (size(u) /= 2*(Nelem+1)) ) then
        write (*,*) 'size of Iyy/force/u is inconsistent of Nelem'
    end

    ! There are Nelems elements and Nelem+1 nodes, but the DOF at the root are fixed; therefore,
    ! since there are 2 DOF per node, the system matrix is 2*Nelem by 2*Nelem.
    allocate(ipiv(2*Nelem), A(2*Nelem,2*Nelem), b(2*Nelem))
    A(:,:) = 0.d0
    b(:) = 0.d0

    ! loop over the interior elements
    dx = L/Nelem
    do i = 2,Nelem
        call CalcElemStiff(E, Iyy(i), Iyy(i+1), dx, Aelem)
        A((i-2)*2+1:i*2,(i-2)*2+1:i*2) = A((i-2)*2+1:i*2,(i-2)*2+1:i*2) + Aelem(:,:)
        call CalcElemLoad(force(i), force(i+1), dx, belem)
        b((i-2)*2+1:i*2) = b((i-2)*2+1:i*2) + belem(:)
    end
    ! handle the root element
    call CalcElemStiff(E, Iyy(1), Iyy(2), dx, Aelem)
    A(1:2,1:2) = A(1:2,1:2) + Aelem(3:4,3:4)
    call CalcElemLoad(force(1), force(2), dx, belem)
    b(1:2) = b(1:2) + belem(3:4)

    ! solve for the displacements
    call dgesv(2*Nelem, 1, A, 2*Nelem, ipiv, b, 2*Nelem, info)
    if (info /= 0) then
        write (*,*) '!!! DGESV exited with info = ',info
    end
    u(:) = b(:)
end
