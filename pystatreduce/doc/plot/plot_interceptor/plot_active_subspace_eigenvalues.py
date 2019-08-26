################################################################################
# plot_active_subspace_eigenvalues.py
#
# The following file plots the eigenvalues of the uncentered covariance of the
# Gradient vector for the Kriging surrogate of the interceptor
#
################################################################################

import numpy as np
from mpl_toolkits import mplot3d
import matplotlib
import matplotlib.pyplot as plt

eigenvals = np.array([25.459885555433928 , 20.5587126040947   ,  5.452218716756889 ,  3.53772505687236  ,  2.2535152219659462,  1.5671907743868985,
        0.3478431228269759,  0.2125105821708078,  0.1352951903477063,  0.0528137827295139,  0.0389375552678914,  0.0248665921597519,
        0.0208783998594401,  0.0163994094253895,  0.0147545582080962,  0.0113168226296197,  0.0107223319790792,  0.0077846215779989,
        0.0058294344695928,  0.0051149165821539])

idx = range(1, eigenvals.size+1)
fname = 'interceptor_kriging_active_eigenvals.pdf'
fig = plt.figure('eigenvalues', figsize=(7,4))
ax = plt.axes()
s = ax.scatter(idx, eigenvals)
ax.set_xlabel('eigenvalue index')
ax.set_ylabel('eigenvalue')
plt.tight_layout()
# plt.show()
plt.savefig(fname, format='pdf')
