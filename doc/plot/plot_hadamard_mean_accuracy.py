# Plot hadamard
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
    dirname = ''.join(['./plot_data/', str(systemsize_arr[i]), '/'])
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
plotname = ''.join(['./plot_data/approximation_error_bar_decay', str(eigen_decayrate_arr[j]), '.pdf'])

f, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(5, sharex=True, figsize=(10,10))
matplotlib.rcParams['mathtext.fontset'] = 'cm'
i = 0
ax1.errorbar(n_eigenmodes_arr, avg_mu_err[i,:], yerr=errs[i,:,:], fmt='-o', barsabove=True)
titlename = ''.join(['systemsize = ', str(systemsize_arr[i]), 'eigen_decayrate = ', str(eigen_decayrate_arr[j])])
ax1.set_title(titlename)

i = 1
ax2.errorbar(n_eigenmodes_arr, avg_mu_err[i,:], yerr=errs[i,:,:], fmt='-o', barsabove=True)
titlename = ''.join(['systemsize = ', str(systemsize_arr[i]), 'eigen_decayrate = ', str(eigen_decayrate_arr[j])])
ax2.set_title(titlename)

i = 2
ax3.errorbar(n_eigenmodes_arr, avg_mu_err[i,:], yerr=errs[i,:,:], fmt='-o', barsabove=True)
titlename = ''.join(['systemsize = ', str(systemsize_arr[i]), 'eigen_decayrate = ', str(eigen_decayrate_arr[j])])
ax3.set_title(titlename)

i = 3
ax4.errorbar(n_eigenmodes_arr, avg_mu_err[i,:], yerr=errs[i,:,:], fmt='-o', barsabove=True)
titlename = ''.join(['systemsize = ', str(systemsize_arr[i]), 'eigen_decayrate = ', str(eigen_decayrate_arr[j])])
ax4.set_title(titlename)

i = 4
ax5.errorbar(n_eigenmodes_arr, avg_mu_err[i,:], yerr=errs[i,:,:], fmt='-o', barsabove=True)
titlename = ''.join(['systemsize = ', str(systemsize_arr[i]), 'eigen_decayrate = ', str(eigen_decayrate_arr[j])])
ax5.set_title(titlename)

f.savefig(plotname, format='pdf')
