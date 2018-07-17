# utils.py
# Contains basic utility functions that can be used across multiple files to make
# life easier

import numpy as np

def isDiag(matrix):
    """
    Check if a matrix is diagonal, For this implementation, a scalar or 1
    element array is considered as diagonal
    """
    if np.isscalar(matrix) == True:
        return True
    elif type(matrix) is np.ndarray:
        if matrix.size == 1:
            return True
        elif np.count_nonzero(matrix - np.diag(np.diagonal(matrix))) == 0:
            return True
        else:
            return False
    else:
        raise NameError('isDiag only handles numpy arrays and scalars')

def mult_diag(d, mtx, left=True):
    """Multiply a full matrix by a diagonal matrix.
    This function should always be faster than dot.

    Input:
      d -- 1D (N,) array (contains the diagonal elements)
      mtx -- 2D (N,N) array

    Output:
      mult_diag(d, mts, left=True) == dot(diag(d), mtx)
      mult_diag(d, mts, left=False) == dot(mtx, diag(d))
    """
    if left:
        return (d*mtx.T).T
    else:
        return d*mtx
