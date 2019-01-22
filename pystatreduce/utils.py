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

def matvecprod(x, y, left=True):
    if left:
        # Check if they are scalar
        if np.isscalar(y) or y.size == 1:
            val = x * y
            return val.reshape(x.shape)
        elif len(x) > 1 and len(y) == 1 and y.size > 1:
            return np.dot(x, y)
    else:
        raise NotImplementedError

def compute_subspace_distance(S1, S2):
    """
    This function is to be used to compute the distance between two dominant
    subspaces.
    """
    assert S1.shape == S2.shape
    intmat = np.matmul(S1, S1.T) - np.matmul(S2, S2.T)
    # We need to compute the second norm of the above matrix. This is equivalent
    # to the square root of largest eigenvalue of the matrix itself.
    eigenvals, _ = np.linalg.eig(intmat)
    distance = np.sqrt(np.sort(eigenvals)[::-1])

    return distance

def compute_subspace_angles(S1, S2):
    """
    This function computes the angles between 2 subspaces and returns the
    corresponding angles in an array of radians.

    This implementation is based on Algorithm 12.4.3 on page 604 in the book
    Matrix Computations by Gene H. Golub and Charles F. Van Loan.
    """
    assert S1.shape == S2.shape
    # Do a QR Factorization of S1 and S2
    Q1, R1 = np.linalg.qr(S1)
    Q2, R2 = np.linalg.qr(S2)
    intmat = np.matmul(Q1.T, Q2)
    Y, s, Z = np.linalg.svd(intmat)
    s_radians = np.arccos(s)

    return s_radians
