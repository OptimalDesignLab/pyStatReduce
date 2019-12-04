subroutine CalcPertForce(force, xi, L, Nelem, pertforce)
    ! Compute the perturbed force given a norminal and cosine perturbation
    ! Inputs:
    !   force - nominal force distribution
    !   xi - coefficients for the cosine series perturbation
    !   L - length of the beam
    !   Nelem - number of finite elements to use
    ! Outputs:
    !   pertforce - the perturbed force
    !--------------------------------------------------------------------------
    implicit none
    double precision, intent(in) :: force(:), xi(:), L
    integer, intent(in) :: Nelem
    double precision, intent(out) :: pertforce(:)
    integer :: i, n
    double precision :: pi, y

    if ( (size(force) /= (Nelem+1)) .or. (size(pertforce) /= (Nelem+1)) ) then
        write (*,*) 'size of force/pertforce inconsistent with Nelem'
    end if
    pertforce(:) = force(:)
    pi = 3.14159265359d0
    do i = 1,Nelem+1
        y = (i-1)*L/Nelem
        do n = 1,size(xi)
            pertforce(i) = pertforce(i) + xi(n)*cos((2*n-1)*pi*y/(2.d0*L))
        end do
    end do
end
