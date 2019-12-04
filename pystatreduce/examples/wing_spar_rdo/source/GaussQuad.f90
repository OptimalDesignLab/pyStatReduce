subroutine GaussQuad(func, n, work, integral)
    ! Integrates func over [-1,1] using n-point Gauss quadrature
    ! Inputs:
    !   func - function of the form func(x, y), where x is a scalar and y can be an array
    !   n - number of points to use for quadrature
    ! InOuts:
    !   work - work array of size of output for func
    ! Outputs:
    !   integral - result of integrating func numerically
    !-----------------------------------------------------------------------------------------------
    implicit none
    interface
        subroutine func(x, y)
            double precision, intent(in) :: x
            double precision, intent(inout) :: y(:)
        end subroutine func
    end interface
    integer, intent(in) :: n
    double precision, intent(inout) :: work(:)
    double precision, intent(out) :: integral(:)

    if (n == 1) then
        func(0.d0, work)
        integral(:) = 2.d0*work(:)
    elseif (n == 2) then
        func(-1.d0/sqrt(3.d0), work)
        integral(:) = work(:)
        func(1.d0/sqrt(3.d0), work)
        integral(:) = integral(:) + work(:)
    elseif (n == 3) then
        func(0.d0, work)
        integral(:) = (8.d0/9.d0)*work(:)
        func(-sqrt(3.d0/5.d0), work)
        integral(:) = integral(:) + (5.d0/9.d0)*work(:)
        func(sqrt(3.d0/5.d0), work)
        integral(:) = integral(:) + (5.d0/9.d0)*work(:)
    else
        write (*,*) 'GaussQuad is only implemented for n =1,2, or 3'
    end
end
