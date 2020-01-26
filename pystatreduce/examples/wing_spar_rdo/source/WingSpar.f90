module WingSpar

    ! get sigatures using:
    ! $ f2py WingSpar.f90 -m WingSpar -h WingSpar.pyf
    ! compile using:
    ! $ f2py -m WingSpar -c WingSpar.pyf WingSpar.f90

    implicit none
    double precision, private, parameter :: pi = 3.14159265359d0
    integer, private :: Nelem
    integer, private, allocatable :: ipiv(:)
    double precision, private, allocatable :: A(:,:), LU(:,:)

    public

    ! _______________________
    ! module subroutines
contains
    !
    ! ==============================================================================================
    !
    subroutine Initialize(numElement)
        ! Sets the number of elements and allocates some arrays
        ! Inputs:
        !   numElement - number of elements
        !--------------------------------------------------------------------------
        integer, intent(in) :: numElement
        if (numElement < 1) then
            write (*,*) 'Initialize: numElement must be >= 1'
        end if
        nElem = numElement
        allocate(ipiv(2*Nelem), A(2*Nelem,2*Nelem), LU(2*Nelem,2*Nelem))
    end subroutine Initialize
    !
    ! ==============================================================================================
    !
    subroutine HermiteBasis(xi, dx, N)
        ! Evaluate the cubic Hermitian shape functions at point xi
        ! Inputs:
        !   xi - point at which to evaluate the shape functions
        !   dx - element length
        ! Outputs:
        !   N - 4x1 vector containing the shape functions at xi
        !--------------------------------------------------------------------------
        double precision, intent(in) :: xi, dx
        double precision, intent(out) :: N(4)
        if (dx <= 0.d0) then
            write (*,*) 'HermiteBasis: element length must be strictly positive'
        end if
        if ( (xi < -1.d0) .or. (xi > 1.d0) ) then
            write (*,*) 'HermiteBasis: shape functions must be evaluated in the interval [-1,1]'
        end if
        N(:) = 0.d0
        ! this can be done more efficiencly by precomputing some of the common
        ! terms, but this is adequate for an academic code.
        N(1) = 0.25d0*((1.d0 - xi)**2)*(2.d0 + xi)
        N(2) = 0.125d0*dx*((1.d0 - xi)**2)*(1.d0 + xi)
        N(3) = 0.25d0*((1.d0 + xi)**2)*(2.d0 - xi)
        N(4) = -0.125d0*dx*((1.d0 + xi)**2)*(1.d0 - xi)
    end subroutine HermiteBasis
    !
    ! ==============================================================================================
    !
    subroutine D2HermiteBasis(xi, dx, B)
        ! Evaluate the second derivative of the Hermitian functions at point xi
        ! Inputs:
        !   xi - point at which to evaluate the shape functions
        !   dx - element length
        ! Outputs:
        !   B - 4x1 vector containing the shape function derivatives at xi
        !--------------------------------------------------------------------------
        double precision, intent(in) :: xi, dx
        double precision, intent(out) :: B(4)
        if (dx <= 0.d0) then
            write (*,*) 'D2HermiteBasis: element length must be strictly positive'
        end if
        if ( (xi < -1.d0) .or. (xi > 1.d0) ) then
            write (*,*) 'D2HermiteBasis: shape functions must be evaluated in the interval [-1,1]'
        end if
        B(:) = 0.d0
        B(1) = 6.d0*xi/dx
        B(2) = 3.d0*xi - 1.d0
        B(3) = -6.d0*xi/dx
        B(4) = 3.d0*xi + 1.d0
        B(:) = B(:)/dx
    end subroutine D2HermiteBasis
    !
    ! ==============================================================================================
    !
    subroutine GaussQuad_vec(func, n, work, integral)
        ! Integrates func over [-1,1] using n-point Gauss quadrature
        ! Inputs:
        !   func - function of the form func(x, y), where x is a scalar and y is a vector
        !   n - number of points to use for quadrature
        ! InOuts:
        !   work - work array of size of output for func
        ! Outputs:
        !   integral - result of integrating func numerically
        !--------------------------------------------------------------------------
        interface
            subroutine func(x, y)
                double precision, intent(in) :: x
                double precision, intent(out) :: y(:)
            end subroutine func
        end interface
        integer, intent(in) :: n
        double precision, intent(inout) :: work(:)
        double precision, intent(out) :: integral(:)

        if (n == 1) then
            call func(0.d0, work)
            integral(:) = 2.d0*work(:)
        elseif (n == 2) then
            call func(-1.d0/sqrt(3.d0), work)
            integral(:) = work(:)
            call func(1.d0/sqrt(3.d0), work)
            integral(:) = integral(:) + work(:)
        elseif (n == 3) then
            call func(0.d0, work)
            integral(:) = (8.d0/9.d0)*work(:)
            call func(-sqrt(3.d0/5.d0), work)
            integral(:) = integral(:) + (5.d0/9.d0)*work(:)
            call func(sqrt(3.d0/5.d0), work)
            integral(:) = integral(:) + (5.d0/9.d0)*work(:)
        else
            write (*,*) 'GuassQuad_vec: only implemented for n =1,2, or 3'
        end if
    end subroutine GaussQuad_vec
    !
    ! ==============================================================================================
    !
    subroutine GaussQuad_mat(func, n, work, integral)
        ! Integrates func over [-1,1] using n-point Gauss quadrature
        ! Inputs:
        !   func - function of the form func(x, y), where x is a scalar and y is a matrix
        !   n - number of points to use for quadrature
        ! InOuts:
        !   work - work array of size of output for func
        ! Outputs:
        !   integral - result of integrating func numerically
        !--------------------------------------------------------------------------
        interface
            subroutine func(x, y)
                double precision, intent(in) :: x
                double precision, intent(out) :: y(:,:)
            end subroutine func
        end interface
        integer, intent(in) :: n
        double precision, intent(inout) :: work(:,:)
        double precision, intent(out) :: integral(:,:)

        if (n == 1) then
            call func(0.d0, work)
            integral(:,:) = 2.d0*work(:,:)
        elseif (n == 2) then
            call func(-1.d0/sqrt(3.d0), work)
            integral(:,:) = work(:,:)
            call func(1.d0/sqrt(3.d0), work)
            integral(:,:) = integral(:,:) + work(:,:)
        elseif (n == 3) then
            call func(0.d0, work)
            integral(:,:) = (8.d0/9.d0)*work(:,:)
            call func(-sqrt(3.d0/5.d0), work)
            integral(:,:) = integral(:,:) + (5.d0/9.d0)*work(:,:)
            call func(sqrt(3.d0/5.d0), work)
            integral(:,:) = integral(:,:) + (5.d0/9.d0)*work(:,:)
        else
            write (*,*) 'GaussQuad_mat: only implemented for n =1,2, or 3'
        end if
    end subroutine GaussQuad_mat
    !
    ! ==============================================================================================
    !
    subroutine CalcElemLoad(qL, qR, dx, belem)
        ! Compute the element load vector for Hermitian cubic shape functions
        ! Inputs:
        !   qL - force per unit length at left end of element
        !   qR - force per unit length at right end of element
        !   dx - element length
        ! Outputs:
        !   belem - the 4x1 element load vector
        !--------------------------------------------------------------------------
        double precision, intent(in) :: qL, qR, dx
        double precision, intent(out) :: belem(4)
        double precision :: work(4), N(4)

        if (dx <= 0.d0) then
            write (*,*) 'CalcElemLoad: element length must be strictly positive'
        end if
        call GaussQuad_vec(Integrand, 3, work, belem)

    contains

        subroutine Integrand(xi, Int)
            ! compute the integrand needed for the element load vector
            double precision, intent(in) :: xi
            double precision, intent(out) :: Int(:)
            call HermiteBasis(xi, dx, N)
            Int(:) = 0.5d0*dx*LinearForce(xi)*N(:)
        end subroutine Integrand

        pure function LinearForce(xi)
            ! evaluate the linear force at point xi
            double precision, intent(in) :: xi
            double precision :: LinearForce
            LinearForce = (qL*(1.d0-xi) + qR*(1.d0+xi))*0.5d0
        end function LinearForce

    end subroutine CalcElemLoad
    !
    ! ==============================================================================================
    !
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
        if ( (IL < 0.0) .or. (IR < 0.0) .or. (E <= 0.0) .or. (dx <= 0.0) ) then
            write (*,*) 'CalcElemStiff: Inputs must all be strictly positive'
        end if
        call GaussQuad_mat(Integrand, 2, work, Aelem)

    contains

        subroutine Integrand(xi, Int)
            ! compute the integrand needed for the element stiffness matrix
            double precision, intent(in) :: xi
            double precision, intent(out) :: Int(:,:)
            double precision :: fac
            integer :: i, j
            call D2HermiteBasis(xi, dx, B)
            fac = 0.5d0*E*dx*LinearMomentInertia(xi)
            do i = 1,4
                do j = 1,4
                    Int(i,j) = fac*B(i)*B(j)
                end do
            end do
        end subroutine Integrand

        function LinearMomentInertia(xi)
            ! evaluate the linear moment of inertia at point xi
            double precision, intent(in) :: xi
            double precision :: LinearMomentInertia
            LinearMomentInertia = (IL*(1.d0-xi) + IR*(1.d0+xi))*0.5d0
        end function LinearMomentInertia

    end subroutine CalcElemStiff
    !
    ! ==============================================================================================
    !
    subroutine CalcBeamMoment(n, L, force, M)
        ! Computes the moment distribution in the spar
        ! Inputs:
        !   n - number of nodes
        !   L - length of the beam
        !   force - force distribution
        ! Outputs:
        !   M - moment distribution
        !--------------------------------------------------------------------------
        integer, intent(in) :: n
        double precision :: L
        double precision, intent(in) :: force(n)
        double precision, intent(out) :: M(n)
        integer :: i
        double precision :: x

        if (n /= Nelem+1) then
            write (*,*) 'CalcBeamMoment: n is inconsistent with Nelem'
        end if
        do i = 1,n
            x = (i-1)*L/Nelem
            M(i) = force(1)*((0.5d0*x**2 - x**3/(6.d0*L)) - (0.5d0*L)*x + L**2/6.d0)
        end do
    end
    !
    ! ==============================================================================================
    !
    subroutine CalcSecondMomentAnnulus(n, r_inner, r_outer, Iyy)
        ! Computes the second-moment of area for an annular region
        ! Inputs:
        !   n - number of nodes
        !   r_inner - the inner radius of the annular region
        !   r_outer - the outer radius of the annular region
        ! Outputs:
        !   Iyy - the second-moment of area
        !--------------------------------------------------------------------------
        integer, intent(in) :: n
        double precision, intent(in) :: r_inner(n), r_outer(n)
        double precision, intent(out) :: Iyy(n)
        integer :: i
        double precision :: fac

        if (n /= Nelem+1) then
            write (*,*) 'CalcSecondMomentAnnulus: n is inconsistent with Nelem'
        end if
        fac = pi/4.d0
        do i = 1,Nelem+1
            Iyy(i) = fac*(r_outer(i)**4 - r_inner(i)**4)
        end do
    end subroutine CalcSecondMomentAnnulus
    !
    ! ==============================================================================================
    !
    subroutine CalcSecondMomentAnnulus_rev(n, r_inner, r_inner_b, r_outer, r_outer_b, Iyy_b)
        ! reverse mode of the second-moment of area for an annular region
        ! Inputs:
        !   n - number of nodes
        !   r_inner - the inner radius of the annular region
        !   r_outer - the outer radius of the annular region
        !   Iyy_b - weight on Iyy product
        ! Outputs:
        !   r_inner_b - derivative of Iyy_b^T*Iyy with respect to r_inner
        !   r_outer_b - derivative of Iyy_b^T*Iyy with respect to r_outer
        !--------------------------------------------------------------------------
        integer, intent(in) :: n
        double precision, intent(in) :: r_inner(n), r_outer(n), Iyy_b(n)
        double precision, intent(inout) :: r_inner_b(n), r_outer_b(n)
        integer :: i
        double precision :: fac

        if (n /= Nelem+1) then
            write (*,*) 'CalcSecondMomentAnnulus_rev: n is inconsistent with Nelem'
        end if
        fac = pi/4.d0
        do i = 1,Nelem+1
            ! Iyy(i) = fac*(r_outer(i)**4 - r_inner(i)**4)
            r_outer_b(i) = r_outer_b(i) + Iyy_b(i)*4.d0*fac*r_outer(i)**3
            r_inner_b(i) = r_inner_b(i) - Iyy_b(i)*4.d0*fac*r_inner(i)**3
        end do
    end subroutine CalcSecondMomentAnnulus_rev
    !
    ! ==============================================================================================
    !
    subroutine CalcPertForce(force, xi, L, pertforce)
        ! Compute the perturbed force given a norminal and cosine perturbation
        ! Inputs:
        !   force - nominal force distribution
        !   xi - coefficients for the cosine series perturbation
        !   L - length of the beam
        !   Nelem - number of finite elements to use
        ! Outputs:
        !   pertforce - the perturbed force
        !--------------------------------------------------------------------------
        double precision, intent(in) :: force(:), xi(:), L
        double precision, intent(out) :: pertforce(:)
        integer :: i, k
        double precision :: y

        if ((size(force) /= Nelem+1) .or. (size(pertforce) /= Nelem+1) ) then
            write (*,*) 'CalcPertForce: size of force/pertforce is inconsistent with Nelem'
        end if
        pertforce(:) = force(:)
        do i = 1,Nelem+1
            y = (i-1)*L/Nelem
            do k = 1,size(xi)
                pertforce(i) = pertforce(i) + xi(k)*cos((2*k-1)*pi*y/(2.d0*L))
            end do
        end do
    end subroutine CalcPertForce
    !
    ! ==============================================================================================
    !
    subroutine BuildAndFactorStiffness(L, E, n, Iyy)
        ! Construct the stiffness matrix and then factor it.
        ! Inputs:
        !   L - length of the beam
        !   E - longitudinal elastic modulus
        !   n - number of nodes
        !   Iyy - moment of inertia with respect to the y axis, as function of x
        !--------------------------------------------------------------------------
        integer, intent(in) :: n
        double precision, intent(in) :: L, E, Iyy(n)
        integer :: i, info, d
        double precision :: dx, Aelem(4,4)

        if ( (L <= 0) .or. (E <= 0) ) then
            write (*,*) 'BuildAndFactorStiffness: inputs L and E must be positive'
        end if
        if (n /= Nelem+1) then
            write (*,*) 'BuildAndFactorStiffness: n is inconsistent with Nelem'
        end if
        A(:,:) = 0.d0
        ! loop over the interior elements
        dx = L/Nelem
        do i = 2,Nelem
            call CalcElemStiff(E, Iyy(i), Iyy(i+1), dx, Aelem)
            A((i-2)*2+1:i*2,(i-2)*2+1:i*2) = A((i-2)*2+1:i*2,(i-2)*2+1:i*2) + Aelem(:,:)
        end do
        ! handle the root element
        call CalcElemStiff(E, Iyy(1), Iyy(2), dx, Aelem)
        A(1:2,1:2) = A(1:2,1:2) + Aelem(3:4,3:4)
        ! factor the stiffness matrix for later use
        LU(:,:) = A(:,:)
        call LUDCMP(LU, 2*Nelem, ipiv, d, info)
        if (info /= 0) then
            write (*,*) 'BuildAndFactorStiffness: LUDCMP exited with info = ',info
        end if
    end subroutine BuildAndFactorStiffness
    !
    ! ==============================================================================================
    !
    subroutine BuildLoadVector(L, n, force, b)
        ! Construct the stiffness matrix and load vector
        ! Inputs:
        !   L - length of the beam
        !   n - number of nodes
        !   force - force per unit length along the beam axis x
        !--------------------------------------------------------------------------
        double precision, intent(in) :: L
        integer, intent(in) :: n
        double precision, intent(in) :: force(n)
        double precision, intent(out) :: b(2*(n-1))
        integer :: i
        double precision :: dx, belem(4)

        if (L <= 0) then
            write (*,*) 'BuildLoadVector: L must be positive'
        end if
        if (n /= Nelem+1) then
            write (*,*) 'BuildLoadVector: n is inconsistent with Nelem'
        end if
        b(:) = 0.d0
        ! loop over the interior elements
        dx = L/Nelem
        do i = 2,Nelem
            call CalcElemLoad(force(i), force(i+1), dx, belem)
            b((i-2)*2+1:i*2) = b((i-2)*2+1:i*2) + belem(:)
        end do
        ! handle the root element
        call CalcElemLoad(force(1), force(2), dx, belem)
        b(1:2) = b(1:2) + belem(3:4)
    end subroutine BuildLoadVector
    !
    ! ==============================================================================================
    !
    subroutine BuildLinearSystem(L, E, n, Iyy, force, b)
        ! Construct the stiffness matrix and load vector
        ! Inputs:
        !   L - length of the beam
        !   E - longitudinal elastic modulus
        !   n - number of nodes
        !   Iyy - moment of inertia with respect to the y axis, as function of x
        !   force - force per unit length along the beam axis x
        !--------------------------------------------------------------------------
        double precision, intent(in) :: L, E
        integer, intent(in) :: n
        double precision, intent(in) :: Iyy(n), force(n)
        double precision, intent(out) :: b(2*(n-1))
        integer :: i
        double precision :: dx, belem(4), Aelem(4,4)

        if (n /= Nelem+1) then
            write (*,*) 'BuildLinearSystem: n is inconsistent with Nelem'
        end if
        if ( (L <= 0) .or. (E <= 0) .or. (n <= 0) ) then
            write (*,*) 'BuildLinearSystem: The inputs L, E, and Nelem must all be positive'
        end if
        A(:,:) = 0.d0
        b(:) = 0.d0
        ! loop over the interior elements
        dx = L/Nelem
        do i = 2,Nelem
            call CalcElemStiff(E, Iyy(i), Iyy(i+1), dx, Aelem)
            A((i-2)*2+1:i*2,(i-2)*2+1:i*2) = A((i-2)*2+1:i*2,(i-2)*2+1:i*2) + Aelem(:,:)
            call CalcElemLoad(force(i), force(i+1), dx, belem)
            b((i-2)*2+1:i*2) = b((i-2)*2+1:i*2) + belem(:)
        end do
        ! handle the root element
        call CalcElemStiff(E, Iyy(1), Iyy(2), dx, Aelem)
        A(1:2,1:2) = A(1:2,1:2) + Aelem(3:4,3:4)
        call CalcElemLoad(force(1), force(2), dx, belem)
        b(1:2) = b(1:2) + belem(3:4)
    end subroutine BuildLinearSystem
    !
    ! ==============================================================================================
    !
    ! subroutine CalcResidual(u, res)
    !     ! returns the residual res = b - A*u, where A is the stiffness matrix
    !     ! Inputs:
    !     !   u - displacements (vertical and angle) at each node along the beam
    !     ! Outputs:
    !     !   res - residual
    !     !--------------------------------------------------------------------------
    !     double precision, intent(in) :: u(:)
    !     double precision, intent(out) :: res(:)
    !
    !     if ( (size(u) /= 2*(Nelem+1)) .or. (size(res) /= 2*(Nelem+1)) ) then
    !         write (*,*) 'size of u/res is inconsistent with Nelem'
    !     end if
    !     res(:) = 0.d0 ! first two elements are associated with the BCs
    !     ! use matmul for now; this can be sped up by taking advantage of sparsity of A
    !     res(3:2*(Nelem+1)) = b(:) - matmul(A, u(3:2*(Nelem+1)))
    ! end
    !
    ! ==============================================================================================
    !
    subroutine CalcBeamDisplacement(n, b, u)
        ! Estimate beam displacements using Euler-Bernoulli beam theory
        ! Inputs:
        !   n - number of nodes
        !   b - load vector (rhs)
        ! Outputs:
        !   u - displacements (vertical and angle) at each node along the beam
        !--------------------------------------------------------------------------
        integer, intent(in) :: n
        double precision, intent(in) :: b(2*(n-1))
        double precision, intent(out) :: u(2*n)

        if (n /= Nelem+1) then
            write (*,*) 'CalcBeamDisplacement: n is inconsistent with Nelem'
        end if
        ! solve for the displacements
        !call dgesv(2*Nelem, 1, A, 2*Nelem, ipiv, b, 2*Nelem, info)
        u(3:2*(Nelem+1)) = b(:)
        call LUBKSB(LU, 2*Nelem, ipiv, u(3:2*(Nelem+1)))
        u(1:2) = 0.d0
    end subroutine CalcBeamDisplacement
    !
    ! ==============================================================================================
    !
    subroutine CalcBeamDisplacement_rev(L, E, n, u, u_b, Iyy_b)
        ! Reverse mode of the displacements product with respect to Iyy
        ! Inputs:
        !   L - length of the beam
        !   E - longitudinal elastic modulus
        !   n - number of nodes
        !   u - displacements (vertical and angle) at each node along the beam
        !   u_b - weights on the displacements
        ! Outputs:
        !   Iyy_b - derivative of u_b^T*u with respect to Iyy
        !--------------------------------------------------------------------------
        double precision, intent(in) :: L, E
        integer, intent(in) :: n
        double precision, intent(in) :: u(2*n), u_b(2*n)
        double precision, intent(out) :: Iyy_b(n)
        integer :: i, j, k
        double precision :: dx, Aelem_b(4,4), Aelem_L(4,4), Aelem_R(4,4)
        double precision, allocatable :: psi(:)

        if ( (L <= 0) .or. (E <= 0) .or. (n <= 0) ) then
            write (*,*) 'CalcBeamDisplacement_rev: The inputs L, E, and n must all be positive'
        end if
        if (n /= Nelem+1) then
            write (*,*) 'CalcBeamDisplacement_rev: n is inconsistent with Nelem'
        end if

        ! start the reverse sweep
        ! u(3:2*(Nelem+1)) = b(:)
        allocate(psi(2*Nelem))
        psi(:) = -u_b(3:2*(Nelem+1))
        ! solve for the adjoints
        call LUBKSB(LU, 2*Nelem, ipiv, psi)

        dx = L/Nelem
        call CalcElemStiff(E, 1.d0, 0.d0, dx, Aelem_L)
        call CalcElemStiff(E, 0.d0, 1.d0, dx, Aelem_R)
        Iyy_b(:) = 0.d0
        ! loop over the interior elements
        do i = 2,Nelem
            ! A((i-2)*2+1:i*2,(i-2)*2+1:i*2) = A((i-2)*2+1:i*2,(i-2)*2+1:i*2) + Aelem(:,:)
            do j = 1,4
                do k = 1,4
                    Aelem_b(j,k) = psi((i-2)*2+j)*u((i-1)*2+k)
                end do
            end do
            ! call CalcElemStiff(E, Iyy(i), Iyy(i+1), dx, Aelem)
            do j = 1,4
                do k = 1,4
                    Iyy_b(i) = Iyy_b(i) + Aelem_L(j,k)*Aelem_b(j,k)
                    Iyy_b(i+1) = Iyy_b(i+1) + Aelem_R(j,k)*Aelem_b(j,k)
                end do
            end do
        end do
        ! handle the root element
        ! A(1:2,1:2) = A(1:2,1:2) + Aelem(3:4,3:4)
        Aelem_b(:,:) = 0.d0
        do j = 1,2
            do k = 1,4
                Aelem_b(2+j,k) = psi(j)*u(k)
            end do
        end do
        ! call CalcElemStiff(E, Iyy(1), Iyy(2), dx, Aelem)
        do j = 1,4
            do k = 1,4
                Iyy_b(1) = Iyy_b(1) + Aelem_L(j,k)*Aelem_b(j,k)
                Iyy_b(2) = Iyy_b(2) + Aelem_R(j,k)*Aelem_b(j,k)
            end do
        end do
        deallocate(psi)
    end subroutine CalcBeamDisplacement_rev
    !
    ! ==============================================================================================
    !
    function SparWeight(n, x, L, rho)
        ! Estimate the weight of the wing spar
        ! Inputs:
        !   n - number of nodes
        !   x - the DVs; x(1:Nelem+1) inner radii and x(Nelem+2:2*(Nelem+1) thickness
        !   L - length of the beam
        !   rho - density of the metal alloy being used
        ! Returns:
        !   SparWeight - the weight of the spar
        !--------------------------------------------------------------------------
        integer, intent(in) :: n
        double precision, intent(in) :: x(2*n)
        double precision, intent(in) :: L, rho
        double precision :: SparWeight
        double precision :: r_in, r_out, fac
        integer :: i

        if (n /= Nelem+1) then
            write (*,*) 'SparWeight: n is inconsistent with Nelem'
        end if
        fac = pi*rho*L/Nelem
        r_in = x(1)
        r_out = x(1) + x(Nelem+2)
        SparWeight = 0.5d0*(r_out**2 - r_in**2)*fac
        do i = 2,Nelem
            r_in = x(i)
            r_out = x(i) + x(Nelem+i+1)
            SparWeight = SparWeight + (r_out**2 - r_in**2)*fac
        end do
        r_in = x(Nelem+1)
        r_out = x(Nelem+1) + x(2*(Nelem+1))
        SparWeight = SparWeight + 0.5d0*(r_out**2 - r_in**2)*fac
    end function SparWeight
    !
    ! ==============================================================================================
    !
    subroutine SparWeight_rev(n, x, x_b, L, rho)
        ! Reverse mode differentiated version of SparWeight
        ! Inputs:
        !   n - number of nodes
        !   x - the DVs; x(1:Nelem+1) inner radii and x(Nelem+2:2*(Nelem+1) thickness
        !   L - length of the beam
        !   rho - density of the metal alloy being used
        ! Outputs:
        !   x_b - derivative the weight of the spar w.r.t. w
        !--------------------------------------------------------------------------
        integer, intent(in) :: n
        double precision, intent(in) :: x(2*n)
        double precision, intent(in) :: L, rho
        double precision, intent(out) :: x_b(2*n)
        double precision :: r_in, r_out, fac
        integer :: i

        if (n /= Nelem+1) then
            write (*,*) 'SparWeight_rev: n is inconsistent with Nelem'
        end if
        fac = pi*rho*L/Nelem
        r_in = x(1)
        r_out = x(1) + x(Nelem+2)
        ! SparWeight = 0.5d0*(r_out**2 - r_in**2)*fac
        x_b(1) = -r_in*fac + r_out*fac
        x_b(Nelem+2) = r_out*fac
        do i = 2,Nelem
            r_in = x(i)
            r_out = x(i) + x(Nelem+i+1)
            ! SparWeight = SparWeight + (r_out**2 - r_in**2)*fac
            x_b(i) = -2.d0*r_in*fac + 2.d0*r_out*fac
            x_b(Nelem+i+1) = 2.d0*r_out*fac
        end do
        r_in = x(Nelem+1)
        r_out = x(Nelem+1) + x(2*(Nelem+1))
        ! SparWeight = SparWeight + 0.5d0*(r_out**2 - r_in**2)*fac
        x_b(Nelem+1) = -r_in*fac + r_out*fac
        x_b(2*(Nelem+1)) = r_out*fac
    end subroutine SparWeight_rev
    !
    ! ==============================================================================================
    !
    subroutine CalcBeamStress(L, E, n, zmax, u, sigma)
        ! Computes (tensile) stresses in a beam based on Euler-Bernoulli beam theory
        ! Inputs:
        !   L - length of the beam
        !   E - longitudinal elastic modulus
        !   n - number of nodes
        !   zmax - maximum height of the beam at each node
        !   u - displacements (vertical and angle) at each node along the beam
        ! Outputs:
        !   sigma - stress at each node in the beam
        !
        ! Assumes the beam is symmetric about the y axis
        !--------------------------------------------------------------------------
        double precision, intent(in) :: L, E
        integer, intent(in) :: n
        double precision, intent(in) :: zmax(n), u(2*n)
        double precision, intent(out) :: sigma(n)
        integer :: i
        double precision :: dx, xi, d2N(4)

        if (n /= Nelem+1) then
            write (*,*) 'CalcBeamStress: n is inconsistent with Nelem'
        end if
        ! loop over the elements and compute the stresses
        sigma(:) = 0.d0
        dx = L/Nelem
        do i = 1,Nelem
            xi = -1.d0
            call D2HermiteBasis(xi, dx, d2N)
            sigma(i) = E*zmax(i)*dot_product(d2N, u((i-1)*2+1:(i+1)*2))
        end do
        xi = 1.d0
        call D2HermiteBasis(xi, dx, d2N)
        sigma(Nelem+1) = E*zmax(Nelem+1)*dot_product(d2N, u((Nelem-1)*2+1:(Nelem+1)*2))
    end subroutine CalcBeamStress
    !
    ! ==============================================================================================
    !
    subroutine CalcBeamStress_rev(L, E, n, zmax, zmax_b, u, u_b, sigma_b)
        ! reverse mode of CalcBeamStress
        ! Inputs:
        !   L - length of the beam
        !   E - longitudinal elastic modulus
        !   n - number of nodes
        !   zmax - maximum height of the beam at each node
        !   u - displacements (vertical and angle) at each node along the beam
        !   sigma_b - weights in stress product at each node in the beam
        ! Outputs:
        !   zmax_b - derivative of sigma_b^T * sigma with respect to zmax_b
        !   u_b - derivative of sigma_b^T * sigma with respect to u
        !
        ! Assumes the beam is symmetric about the y axis
        !--------------------------------------------------------------------------
        double precision, intent(in) :: L, E
        integer, intent(in) :: n
        double precision, intent(in) :: zmax(n), u(2*n), sigma_b(n)
        double precision, intent(out) :: zmax_b(n), u_b(2*n)
        integer :: i
        double precision :: dx, xi, d2N(4)

        if (n /= Nelem+1) then
            write (*,*) 'CalcBeamStress_rev: n is inconsistent with Nelem'
        end if
        ! loop over the elements and compute the stresses
        dx = L/Nelem
        do i = 1,Nelem
            xi = -1.d0
            call D2HermiteBasis(xi, dx, d2N)
            ! sigma(i) = E*zmax(i)*dot_product(d2N, u((i-1)*2+1:(i+1)*2))
            zmax_b(i) = zmax_b(i) + E*dot_product(d2N, u((i-1)*2+1:(i+1)*2))*sigma_b(i)
            u_b((i-1)*2+1:(i+1)*2) = u_b((i-1)*2+1:(i+1)*2) + E*zmax(i)*d2N(:)*sigma_b(i)
        end do
        xi = 1.d0
        call D2HermiteBasis(xi, dx, d2N)
        ! sigma(Nelem+1) = E*zmax(Nelem+1)*dot_product(d2N, u(Nelem*2+1:(Nelem+1)*2))
        zmax_b(Nelem+1) = zmax_b(Nelem+1) + &
            E*dot_product(d2N, u((Nelem-1)*2+1:(Nelem+1)*2))*sigma_b(Nelem+1)
        u_b((Nelem-1)*2+1:(Nelem+1)*2) = u_b((Nelem-1)*2+1:(Nelem+1)*2) + &
            E*zmax(Nelem+1)*d2N(:)*sigma_b(Nelem+1)
    end subroutine CalcBeamStress_rev
    !
    ! ==============================================================================================
    !
    subroutine StressConstraints_old(n, x, nxi, xi, L, E, force, yield, cineq)
        ! Computes the nonlinear inequality constraints for the wing-spar problem
        ! Inputs:
        !   n - number of nodes
        !   x - the DVs; x(1:Nelem+1) inner radii and x(Nelem+2:2*(Nelem+1) thickness
        !   nxi - number of coefficients in the force perturbation series
        !   xi - the uncertain force-perturbation coefficients
        !   L - length of the beam
        !   E - longitudinal elastic modulus
        !   force - the nominal force
        !   yield - the yield stress for the material
        ! Outputs:
        !   cineq - inequality (stress) constraints
        !--------------------------------------------------------------------------
        integer, intent(in) :: n, nxi
        double precision, intent(in) :: x(2*n), xi(nxi), L, E, force(n), yield
        double precision, intent(out) :: cineq(n)
        double precision, allocatable :: pertforce(:), zmax(:), u(:), b(:)

        if (n /= Nelem+1) then
            write (*,*) 'StressConstraints: n is inconsistent with Nelem'
        end if
        allocate(pertforce(n), zmax(n), u(2*n), b(2*Nelem))
        !call CalcSecondMomentAnnulus(n, x(1:Nelem+1), x(Nelem+2:2*(Nelem+1)), Iyy)
        call CalcPertForce(force, xi, L, pertforce)
        !call BuildLinearSystem(L, E, Iyy, pertforce)
        call BuildLoadVector(L, n, pertforce, b)
        call CalcBeamDisplacement(n, b, u)
        zmax(:) = x(1:Nelem+1) + x(Nelem+2:2*(Nelem+1))
        call CalcBeamStress(L, E, n, zmax, u, cineq)
        cineq(:) = 1.0 - cineq(:)/yield
        deallocate(pertforce, zmax, u, b)
    end subroutine StressConstraints_old
    !
    ! ==============================================================================================
    !
    subroutine StressConstraints(n, x, nxi, xi, L, force, yield, cineq)
        ! Computes the nonlinear inequality constraints for the wing-spar problem
        ! Inputs:
        !   n - number of nodes
        !   x - the DVs; x(1:Nelem+1) inner radii and x(Nelem+2:2*(Nelem+1) thickness
        !   nxi - number of coefficients in the force perturbation series
        !   xi - the uncertain force-perturbation coefficients
        !   L - length of the beam
        !   force - the nominal force
        !   yield - the yield stress for the material
        ! Outputs:
        !   cineq - inequality (stress) constraints
        !--------------------------------------------------------------------------
        integer, intent(in) :: n, nxi
        double precision, intent(in) :: x(2*n), xi(nxi), L, force(n), yield
        double precision, intent(out) :: cineq(n)
        integer :: i
        double precision, allocatable :: pertforce(:), zmax(:), Iyy(:), M(:)

        if (n /= Nelem+1) then
            write (*,*) 'StressConstraints: n is inconsistent with Nelem'
        end if
        allocate(pertforce(n), zmax(n), Iyy(n), M(n))
        zmax(:) = x(1:n) + x(n+1:2*n)
        call CalcSecondMomentAnnulus(n, x(1:n), zmax, Iyy)
        call CalcPertForce(force, xi, L, pertforce)
        print *, pertforce
        call CalcBeamMoment(n, L, pertforce, M)
        do i = 1,n
            ! cineq(i) = 1.0 - M(i)*zmax(i)/(yield*Iyy(i))
            cineq(i) = Iyy(i) - M(i)*zmax(i)/yield
        end do
        deallocate(pertforce, zmax, Iyy, M)
    end subroutine StressConstraints
    !
    ! ==============================================================================================
    !
    subroutine StressConstraints_old_rev(n, x, x_b, nxi, xi, L, E, force, yield, cineq_b)
        ! Computes the reverse mode of the nonlinear inequality constraints
        ! Inputs:
        !   n - number of nodes
        !   x - the DVs; x(1:Nelem+1) inner radii and x(Nelem+2:2*(Nelem+1) thickness
        !   nxi - number of basis function in the force perturbation series
        !   xi - the uncertain force-perturbation coefficients
        !   L - length of the beam
        !   E - longitudinal elastic modulus
        !   force - the nominal force
        !   yield - the yield stress for the material
        !   cineq_b - weights on inequality (stress) constraints
        ! Outputs:
        !   x_b - derivatives of cineq_b^T*cineq with respect to x
        !--------------------------------------------------------------------------
        integer, intent(in) :: n, nxi
        double precision, intent(in) :: x(2*n), xi(nxi), L, E, force(n), yield, cineq_b(n)
        double precision, intent(out) :: x_b(2*n)
        double precision, allocatable :: pertforce(:), zmax(:),  u(:), b(:), Iyy(:)
        double precision, allocatable :: zmax_b(:), u_b(:), Iyy_b(:)

        if (n /= Nelem+1) then
            write (*,*) 'StressConstraints_rev: n is inconsistent with Nelem'
        end if
        allocate(pertforce(n), zmax(n), zmax_b(n), u(2*n), u_b(2*n), b(2*Nelem), Iyy(n), Iyy_b(n))

        ! need the displacements due to bilinear terms
        zmax(:) = x(1:Nelem+1) + x(Nelem+2:2*(Nelem+1))
        call CalcSecondMomentAnnulus(n, x(1:Nelem+1), zmax, Iyy)
        call CalcPertForce(force, xi, L, pertforce)
        call BuildLoadVector(L, n, pertforce, b)
        call CalcBeamDisplacement(n, b, u)

        ! cineq(:) = 1.0 - cineq(:)/yield ! do this at the end
        ! call CalcBeamStress(L, E, n, x(Nelem+2:2*(Nelem+1)), u, Nelem, cineq)
        x_b(:) = 0.d0
        u_b(:) = 0.d0
        zmax_b(:) = 0.d0
        call CalcBeamStress_rev(L, E, n, zmax, zmax_b(:), u, u_b, &
                                cineq_b)
        !call CalcPertForce(force, xi, L, Nelem, pertforce)
        !call CalcBeamDisplacement(L, E, Iyy, pertforce, Nelem, u)
        Iyy_b(:) = 0.d0
        call CalcBeamDisplacement_rev(L, E, n, u, u_b, Iyy_b)
        !call CalcSecondMomentAnnulus(n, x(1:Nelem+1), x(Nelem+2:2*(Nelem+1)), Iyy)
        call CalcSecondMomentAnnulus_rev(n, x(1:Nelem+1), x_b(1:Nelem+1), zmax, zmax_b, Iyy_b)
        x_b(1:Nelem+1) = x_b(1:Nelem+1) + zmax_b(:)
        x_b(Nelem+2:2*(Nelem+1)) = x_b(Nelem+2:2*(Nelem+1)) + zmax_b(:)
        x_b(:) = -x_b(:)/yield ! did not do this earlier, so do it now
        deallocate(pertforce, zmax, zmax_b, u, u_b, b, Iyy, Iyy_b)
    end subroutine StressConstraints_old_rev
    !
    ! ==============================================================================================
    !
    subroutine StressConstraints_rev(n, x, x_b, nxi, xi, L, force, yield, cineq_b)
        ! Computes the reverse mode of the nonlinear inequality constraints
        ! Inputs:
        !   n - number of nodes
        !   x - the DVs; x(1:Nelem+1) inner radii and x(Nelem+2:2*(Nelem+1) thickness
        !   nxi - number of basis function in the force perturbation series
        !   xi - the uncertain force-perturbation coefficients
        !   L - length of the beam
        !   force - the nominal force
        !   yield - the yield stress for the material
        !   cineq_b - weights on inequality (stress) constraints
        ! Outputs:
        !   x_b - derivatives of cineq_b^T*cineq with respect to x
        !--------------------------------------------------------------------------
        integer, intent(in) :: n, nxi
        double precision, intent(in) :: x(2*n), xi(nxi), L, force(n), yield, cineq_b(n)
        double precision, intent(out) :: x_b(2*n)
        integer :: i
        double precision, allocatable :: pertforce(:), zmax(:), Iyy(:), M(:)
        double precision, allocatable :: zmax_b(:), Iyy_b(:)

        if (n /= Nelem+1) then
            write (*,*) 'StressConstraints_rev: n is inconsistent with Nelem'
        end if
        allocate(pertforce(n), zmax(n), zmax_b(n), Iyy(n), Iyy_b(n), M(n))
        ! forward sweep
        zmax(:) = x(1:n) + x(n+1:2*n)
        call CalcSecondMomentAnnulus(n, x(1:n), zmax, Iyy)
        call CalcPertForce(force, xi, L, pertforce)
        call CalcBeamMoment(n, L, pertforce, M)
        zmax_b(:) = 0.d0
        Iyy_b(:) = 0.d0
        x_b(:) = 0.d0
        do i = 1,n
            ! cineq(i) = 1.0 - M(i)*zmax(i)/(yield*Iyy(i))
            ! zmax_b(i) = zmax_b(i) - cineq_b(i)*M(i)/(yield*Iyy(i))
            ! Iyy_b(i) = Iyy_b(i) + cineq_b(i)*M(i)*zmax(i)/(yield*Iyy(i)*Iyy(i))
            ! cineq(i) = Iyy(i) - M(i)*zmax(i)/yield
            zmax_b(i) = zmax_b(i) - cineq_b(i)*M(i)/yield
            Iyy_b(i) = Iyy_b(i) + cineq_b(i)
        end do
        ! call CalcSecondMomentAnnulus(n, x(1:Nelem+1), x(Nelem+2:2*(Nelem+1)), Iyy)
        call CalcSecondMomentAnnulus_rev(n, x(1:n), x_b(1:n), zmax, zmax_b, Iyy_b)
        ! zmax(:) = x(1:n) + x(n+1:2*n)
        x_b(1:n) = x_b(1:n) + zmax_b(1:n)
        x_b(n+1:2*n) = x_b(n+1:2*n) + zmax_b(1:n)
        deallocate(pertforce, zmax, zmax_b, Iyy, Iyy_b, M)
    end subroutine StressConstraints_rev
    !
    ! ==============================================================================================
    !
    subroutine LUDCMP(A, N, INDX, D, CODE)
        !  ***************************************************************
        !  * Given an N x N matrix A, this routine replaces it by the LU *
        !  * decomposition of a rowwise permutation of itself. A and N   *
        !  * are input. INDX is an output vector which records the row   *
        !  * permutation effected by the partial pivoting; D is output   *
        !  * as -1 or 1, depending on whether the number of row inter-   *
        !  * changes was even or odd, respectively. This routine is used *
        !  * in combination with LUBKSB to solve linear equations or to  *
        !  * invert a matrix. Return code is 1, if matrix is singular.   *
        !  ***************************************************************
        double precision, intent(inout) :: A(:,:)
        integer, intent(in) :: N
        integer, intent(out) :: INDX(:)
        integer, intent(out) :: D, CODE

        integer, parameter :: NMAX=100
        double precision, parameter :: TINY=1.5D-16
        double precision :: AMAX, DUM, SUM, VV(NMAX)
        integer :: I, J, K, IMAX

        D=1; CODE=0

        DO I=1,N
            AMAX=0.d0
            DO J=1,N
                IF (DABS(A(I,J)).GT.AMAX) AMAX=DABS(A(I,J))
            END DO ! j loop
            IF(AMAX.LT.TINY) THEN
                CODE = 1
                RETURN
            END IF
            VV(I) = 1.d0 / AMAX
        END DO ! i loop

        DO J=1,N
            DO I=1,J-1
                SUM = A(I,J)
                DO K=1,I-1
                    SUM = SUM - A(I,K)*A(K,J)
                END DO ! k loop
                A(I,J) = SUM
            END DO ! i loop
            AMAX = 0.d0
            DO I=J,N
                SUM = A(I,J)
                DO K=1,J-1
                    SUM = SUM - A(I,K)*A(K,J)
                END DO ! k loop
                A(I,J) = SUM
                DUM = VV(I)*DABS(SUM)
                IF(DUM.GE.AMAX) THEN
                    IMAX = I
                    AMAX = DUM
                END IF
            END DO ! i loop

            IF(J.NE.IMAX) THEN
                DO K=1,N
                    DUM = A(IMAX,K)
                    A(IMAX,K) = A(J,K)
                    A(J,K) = DUM
                END DO ! k loop
                D = -D
                VV(IMAX) = VV(J)
            END IF

            INDX(J) = IMAX
            IF(DABS(A(J,J)) < TINY) A(J,J) = TINY

            IF(J.NE.N) THEN
                DUM = 1.d0 / A(J,J)
                DO I=J+1,N
                    A(I,J) = A(I,J)*DUM
                END DO ! i loop
            END IF
        END DO ! j loop

        RETURN
    END subroutine LUDCMP
    !
    ! ==============================================================================================
    !
    subroutine LUBKSB(A, N, INDX, B)
        !  ******************************************************************
        !  * Solves the set of N linear equations A . X = B.  Here A is     *
        !  * input, not as the matrix A but rather as its LU decomposition, *
        !  * determined by the routine LUDCMP. INDX is input as the permuta-*
        !  * tion vector returned by LUDCMP. B is input as the right-hand   *
        !  * side vector B, and returns with the solution vector X. A, N and*
        !  * INDX are not modified by this routine and can be used for suc- *
        !  * cessive calls with different right-hand sides. This routine is *
        !  * also efficient for plain matrix inversion.                     *
        !  ******************************************************************
        double precision, intent(in) :: A(:,:)
        integer, intent(in) :: N, INDX(:)
        double precision, intent(inout) :: B(:)

        double precision :: SUM
        integer :: I, J, II, LL

        II = 0

        DO I=1,N
            LL = INDX(I)
            SUM = B(LL)
            B(LL) = B(I)
            IF(II.NE.0) THEN
                DO J=II,I-1
                    SUM = SUM - A(I,J)*B(J)
                END DO ! j loop
            ELSE IF(SUM.NE.0.d0) THEN
                II = I
            END IF
            B(I) = SUM
        END DO ! i loop

        DO I=N,1,-1
            SUM = B(I)
            IF(I < N) THEN
                DO J=I+1,N
                    SUM = SUM - A(I,J)*B(J)
                END DO ! j loop
            END IF
            B(I) = SUM / A(I,I)
        END DO ! i loop

        RETURN
    END subroutine LUBKSB

end module WingSpar
