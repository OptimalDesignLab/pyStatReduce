# utils.py
# Contains basic utility functions that can be used across multiple files to make
# life easier

import numpy as np
import copy

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

    # NaN prevention check
    indices = np.where(s > 1) # Get the indices where the violation exisits
    for i in indices: # Loop over these indices to fix the violation
        if s[i] - 1 < 1.e-13: # This violation limit is pulled out of thin air!
            s[i] = 1.0

    s_radians = np.arccos(s)

    return s_radians

def get_scaneagle_input_rv_statistics(rv_dict):

    """
    This function is specific to the ScanEagle problem, in that it creates the
    mean value and standard deviation arrays necessary to create a joint
    distribution object rom chaospy. It accepts a dictionary, and then parses it
    to create the arrays. The random varaiable dictionary should look like
    ```
    rv_dict = {'rv_name' : {'mean' : 0.0,
                            'std_dev' : 1.0},
              }
    ```
    """
    mu = np.zeros(len(rv_dict))
    std_dev = np.eye(len(rv_dict))
    i = 0
    for rvs in rv_dict:
        if rvs == 'Mach_number':
            mu[i] = rv_dict[rvs]['mean']
            std_dev[i,i] = rv_dict[rvs]['std_dev']
        elif rvs == 'CT':
            mu[i] = rv_dict[rvs]['mean']
            std_dev[i,i] = rv_dict[rvs]['std_dev']
        elif rvs == 'W0':
            mu[i] = rv_dict[rvs]['mean']
            std_dev[i,i] = rv_dict[rvs]['std_dev']
        elif rvs == 'mrho':
            mu[i] = rv_dict[rvs]['mean']
            std_dev[i,i] = rv_dict[rvs]['std_dev']
        elif rvs == 'R':
            mu[i] = rv_dict[rvs]['mean']
            std_dev[i,i] = rv_dict[rvs]['std_dev']
        elif rvs == 'load_factor':
            mu[i] = rv_dict[rvs]['mean']
            std_dev[i,i] = rv_dict[rvs]['std_dev']
        elif rvs == 'E':
            mu[i] = rv_dict[rvs]['mean']
            std_dev[i,i] = rv_dict[rvs]['std_dev']
        elif rvs == 'G':
            mu[i] = rv_dict[rvs]['mean']
            std_dev[i,i] = rv_dict[rvs]['std_dev']
        elif rvs == 'altitude':
            mu[i] = rv_dict[rvs]['mean']
            std_dev[i,i] = rv_dict[rvs]['std_dev']
        i += 1

    return mu, std_dev

def copy_qoi_dict(QoI_dict):
    """
    This function is used for copying the QoI dict, in instances where a simple
    copy.deepcopy cannot be used. This function is disctionary specific, and
    should not be used for copying any other dictionary
    """
    new_dict = dict.fromkeys(QoI_dict)
    for keys in QoI_dict:
        new_dict[keys] = dict.fromkeys(QoI_dict[keys])
        new_dict[keys]['QoI_func'] = QoI_dict[keys]['QoI_func']
        new_dict[keys]['output_dimensions'] = copy.deepcopy(QoI_dict[keys]['output_dimensions'])
        # Copy the deriv_dict now
        if 'deriv_dict' in QoI_dict[keys]:
            new_dict[keys]['deriv_dict'] = dict.fromkeys(QoI_dict[keys]['deriv_dict'])
            for key2 in new_dict[keys]['deriv_dict']:
                new_dict[keys]['deriv_dict'][key2] =  dict.fromkeys(QoI_dict[keys]['deriv_dict'][key2])
                new_dict[keys]['deriv_dict'][key2]['dQoI_func'] = QoI_dict[keys]['deriv_dict'][key2]['dQoI_func']
                new_dict[keys]['deriv_dict'][key2]['output_dimensions'] = copy.copy(QoI_dict[keys]['deriv_dict'][key2]['output_dimensions'])

    # Make sure that the dictionaries have been copied correctly
    assert new_dict == QoI_dict
    assert new_dict is not QoI_dict

    return new_dict
