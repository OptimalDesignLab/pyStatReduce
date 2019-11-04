# Plot hadamard approximation error all systemsizes for a given decay rate

import numpy as np
from mpl_toolkits import mplot3d
import matplotlib
import matplotlib.pyplot as plt

systemsize_arr = [16, 32, 64, 128, 256]
eigen_decayrate_arr = [2.0, 1.0, 0.5]
n_eigenmodes_arr = range(1,11)
n_stddev_samples = 10
n_e_sample = len(n_eigenmodes_arr)

avg_mu_err = np.zeros([len(systemsize_arr), n_e_sample])
max_mu_err = np.zeros([len(systemsize_arr), n_e_sample])
min_mu_err = np.zeros([len(systemsize_arr), n_e_sample])
errs = np.zeros([len(systemsize_arr), 2, n_e_sample])

j = 0
for i in xrange(0, len(systemsize_arr)):
    dirname = ''.join(['./plot_data/mean_accuracy/', str(systemsize_arr[i]), '/'])
    fname1 = ''.join([dirname, 'avg_err_decay', str(eigen_decayrate_arr[j]), '.txt'])
    fname2 = ''.join([dirname, 'max_err_decay', str(eigen_decayrate_arr[j]), '.txt'])
    fname3 = ''.join([dirname, 'min_err_decay', str(eigen_decayrate_arr[j]), '.txt'])

    # Read data
    avg_mu_err_vec = np.loadtxt(fname1, delimiter=',')
    max_mu_err_vec = np.loadtxt(fname2, delimiter=',')
    min_mu_err_vec = np.loadtxt(fname3, delimiter=',')

    avg_mu_err[i,:] = avg_mu_err_vec
    max_mu_err[i,:] = max_mu_err_vec
    min_mu_err[i,:] = min_mu_err_vec

    errs[i,0,:] = min_mu_err_vec
    errs[i,1,:] = max_mu_err_vec

# Plot
titlename = ''.join(['systemsize = ', str(systemsize_arr[i]), 'eigen_decayrate = ', str(eigen_decayrate_arr[j])])
plotname = ''.join(['./plot_data/mean_accuracy/approximation_error_bar_decay', str(eigen_decayrate_arr[j]), '.pdf'])

plt.rc('text', usetex=True)
matplotlib.rcParams['mathtext.fontset'] = 'cm'

f, axes = plt.subplots(5, sharex=True, sharey=True, figsize=(10,10))
plt.setp(axes, xticks=[0,1,2,3,4,5,6,7,8,9,10])
# plt.tight_layout()
if j == 0:
    f.suptitle(r'$\lambda_{i} = \frac{1}{i^2}$')
elif j==1:
    f.suptitle(r'$\lambda_{i} = \frac{1}{i}$')
elif j==2:
    f.suptitle(r'$\lambda_{i} = \frac{1}{\sqrt{i}}$')
i = 0
axes[i].errorbar(n_eigenmodes_arr, avg_mu_err[i,:], yerr=errs[i,:,:], fmt='-o', barsabove=True)
titlename = ''.join(['system size = ', str(systemsize_arr[i])])
axes[i].set_yscale("log", nonposy='clip')
axes[i].set_title(titlename)

i = 1
axes[i].errorbar(n_eigenmodes_arr, avg_mu_err[i,:], yerr=errs[i,:,:], fmt='-o', barsabove=True)
titlename = ''.join(['system size = ', str(systemsize_arr[i])])
axes[i].set_yscale("log", nonposy='clip')
axes[i].set_title(titlename)

i = 2
axes[i].errorbar(n_eigenmodes_arr, avg_mu_err[i,:], yerr=errs[i,:,:], fmt='-o', barsabove=True)
titlename = ''.join(['system size = ', str(systemsize_arr[i])])
axes[i].set_ylabel(r'approximation error, $\epsilon$')
axes[i].set_yscale("log", nonposy='clip')
axes[i].set_title(titlename)

i = 3
axes[i].errorbar(n_eigenmodes_arr, avg_mu_err[i,:], yerr=errs[i,:,:], fmt='-o', barsabove=True)
titlename = ''.join(['system size = ', str(systemsize_arr[i])])
axes[i].set_yscale("log", nonposy='clip')
axes[i].set_title(titlename)

i = 4
axes[i].errorbar(n_eigenmodes_arr, avg_mu_err[i,:], yerr=errs[i,:,:], fmt='-o', barsabove=True)
titlename = ''.join(['system size = ', str(systemsize_arr[i])])
axes[i].set_xlabel("Maximum allowable eigenmodes for collocation")
axes[i].set_yscale("log", nonposy='clip')
axes[i].set_title(titlename)

f.savefig(plotname, format='pdf')
