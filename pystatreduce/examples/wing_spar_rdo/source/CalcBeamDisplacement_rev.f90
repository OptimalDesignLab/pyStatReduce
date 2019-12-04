subroutine CalcBeamDisplacement_rev(L, E, Iyy, Iyy_b, force, Nelem, u, u_b)
    ! Reverse mode of the displacements product with respect to Iyy
    ! Inputs:
    !   L - length of the beam
    !   E - longitudinal elastic modulus
    !   Iyy - moment of inertia with respect to the y axis, as function of x
    !   force - force per unit length along the beam axis x
    !   Nelem - number of finite elements to use
    !   u - displacements (vertical and angle) at each node along the beam
    !   u_b - weights on the displacements
    ! Outputs:
    !   Iyy_b - derivative of u_b^T*u with respect to Iyy
    !
    ! Uses a cubic Hermitian finite-element basis to solve the Euler-Bernoulli
    ! beam equations.  The beam is assumed to lie along the x axis, with the
    ! force applied transversely in the xz plane.
    !--------------------------------------------------------------------------
    implicit none
    double precision, intent(in) :: L, E
    double precision, intent(in) :: Iyy(:), force(:), u(:), u_b(:)
    integer, intent(in) :: Nelem
    double precision, intent(out) :: Iyy_b(:)
    integer :: i, info
    double precision :: dx, belem(4), Aelem_b(4,4), Aelem_L(4,4), Aelem_R(4,4)
    integer, allocatable :: ipiv(:)
    double precision, allocatable :: A(:,:), b(:)

    if ( (L <= 0) .or. (E <= 0) .or. (Nelem <= 0) ) then
        write (*,*) 'The inputs L, E, and Nelem must all be positive'
    end
    if ( (size(Iyy) /= Nelem+1) .or. (size(Iyy_b) /= Nelem+1) .or. &
        (size(force) /= (Nelem+1)) .or. (size(u) /= 2*(Nelem+1)) .or. &
        (size(u_b) /= 2*(Nelem+1)) ) then
        write (*,*) 'size of Iyy/Iyy_b/force/u/u_b is inconsistent of Nelem'
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

    ! start the reverse sweep
    ! u(:) = b(:)
    b(:) = u_b(:)
    ! solve for the adjoints
    ! call dgesv(2*Nelem, 1, A, 2*Nelem, ipiv, b, 2*Nelem, info)
    call dgesv(2*Nelem, 1, A, 2*Nelem, ipiv, b, 2*Nelem, info)
    if (info /= 0) then
        write (*,*) '!!! DGESV exited with info = ',info
    end

    call CalcElemStiff(E, 1.0, 0.0, dx, Aelem_L)
    call CalcElemStiff(E, 0.0, 1.0, dx, Aelem_R)

    ! loop over the interior elements
    do i = 2,Nelem
        ! A((i-2)*2+1:i*2,(i-2)*2+1:i*2) = A((i-2)*2+1:i*2,(i-2)*2+1:i*2) + Aelem(:,:)
        do j = 1,4
            do k = 1,4
                Aelem_b(j,k) = b((i-2)*2+j)*u((i-2)*2+k)
            end
        end
        ! call CalcElemStiff(E, Iyy(i), Iyy(i+1), dx, Aelem)
        do j = 1,4
            do k = 1,4
                Iyy_b(i) = Iyy_b(i) + Aelem_L(j,k)*Aelem_b(j,k)
                Iyy_b(i+1) = Iyy_b(i+1) + Aelem_R(j,k)*Aelem_b(j,k)
            end
        end
    end
    ! handle the root element
    ! A(1:2,1:2) = A(1:2,1:2) + Aelem(3:4,3:4)
    Aelem(:,:) = 0.d0
    do j = 1,2
        do k = 1,2
            Aelem_b(2+j,2+k) = b(j)*u(k)
        end
    end
    ! call CalcElemStiff(E, Iyy(1), Iyy(2), dx, Aelem)
    do j = 1,4
        do k = 1,4
            Iyy_b(1) = Iyy_b(1) + Aelem_L(j,k)*Aelem_b(j,k)
            Iyy_b(2) = Iyy_b(2) + Aelem_R(j,k)*Aelem_b(j,k)
        end
    end
end
