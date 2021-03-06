################################################################################
#
# plot_hadamard_stddev_accuracy.py
#
# Plot hadamard approximation error in standard deviation for select 3
# systemsizes for given decay rates.
#
################################################################################
import numpy as np
from mpl_toolkits import mplot3d
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter

systemsize_arr = [16, 64, 256] # [16, 32, 64, 128, 256]
eigen_decayrate_arr = [2.0, 1.0, 0.5]
n_eigenmodes_arr = range(1,11)
n_stddev_samples = 10
n_e_sample = len(n_eigenmodes_arr)

avg_stddev_err = np.zeros([len(systemsize_arr), 3, n_e_sample])
max_stddev_err = np.zeros([len(systemsize_arr), 3, n_e_sample])
min_stddev_err = np.zeros([len(systemsize_arr), 3, n_e_sample])
errs = np.zeros([len(systemsize_arr), 3, 2, n_e_sample])

plot_std_dev = False

for i in range(0, len(systemsize_arr)):
    for j in range(0,3):
        dirname = ''.join(['./plot_data/std_dev_accuracy/', str(systemsize_arr[i]), '/'])
        fname1 = ''.join([dirname, 'avg_err_decay', str(eigen_decayrate_arr[j]), '.txt'])
        fname2 = ''.join([dirname, 'max_err_decay', str(eigen_decayrate_arr[j]), '.txt'])
        fname3 = ''.join([dirname, 'min_err_decay', str(eigen_decayrate_arr[j]), '.txt'])

        # Read data
        avg_stddev_err_vec = np.loadtxt(fname1, delimiter=',')
        max_stddev_err_vec = np.loadtxt(fname2, delimiter=',')
        min_stddev_err_vec = np.loadtxt(fname3, delimiter=',')

        avg_stddev_err[i,j,:] = avg_stddev_err_vec
        max_stddev_err[i,j,:] = max_stddev_err_vec
        min_stddev_err[i,j,:] = min_stddev_err_vec

        errs[i,j,0,:] = min_stddev_err_vec
        errs[i,j,1,:] = max_stddev_err_vec


# Plot data: We plot 16, 64, 256
titlename = ''.join(['systemsize = ', str(systemsize_arr[i]), 'eigen_decayrate = ', str(eigen_decayrate_arr[j])])

plt.rc('text', usetex=True)
matplotlib.rcParams['mathtext.fontset'] = 'cm'
matplotlib.rcParams.update({'font.size': 18})

f, axes = plt.subplots(1,3, sharey=True , figsize=(10,4))
plt.setp(axes, xticks=[0,1,2,3,4,5,6,7,8,9,10])
# plt.setp(axes, yticks=[1.e-4, 1.e-2, 1.0])
props = dict(boxstyle='round', facecolor='white')

i = 0 # Systemsize index. This needs
j = 0  # Eigen decay rate index
axes[j].set_yscale("log", nonposy='clip')
axes[j].errorbar(n_eigenmodes_arr, avg_stddev_err[i,j,:], yerr=errs[i,j,:,:], fmt='-o', capsize=6)
axes[j].set_ylabel(r'approximation error, $\epsilon_{\sigma}$')
axes[j].set_xlabel('dominant directions')
axes[j].text(0.5,1,r'$\lambda_i = \frac{1}{i^2}$', size=18, bbox=props, \
              transform=axes[j].transAxes, horizontalalignment='center', \
              verticalalignment='center')
axes[j].yaxis.grid(which='major', linestyle=':')
axes[j].minorticks_off()
# axes[j].set_yticks([1.e-4, 1.e-2, 1.0])
axes[j].set_ylim(1.e-5, 1)

j = 1  # Eigen decay rate index
axes[j].set_yscale("log", nonposy='clip')
axes[j].errorbar(n_eigenmodes_arr, avg_stddev_err[i,j,:], yerr=errs[i,j,:,:], fmt='-o', capsize=6)
axes[j].set_xlabel('dominant directions')
axes[j].text(0.5,1,r'$\lambda_i = \frac{1}{i}$', size=18, bbox=props, \
              transform=axes[j].transAxes, horizontalalignment='center', \
              verticalalignment='center')
axes[j].minorticks_off()
# axes[j].set_yticks([1.e-4, 1.e-2, 1.0])
axes[j].yaxis.grid(which='major', linestyle=':')

j = 2  # Eigen decay rate index
axes[j].set_yscale("log", nonposy='clip')
axes[j].errorbar(n_eigenmodes_arr, avg_stddev_err[i,j,:], yerr=errs[i,j,:,:], fmt='-o', capsize=6)
axes[j].set_xlabel('dominant directions')
axes[j].text(0.5,1,r'$\lambda_i = \frac{1}{\sqrt{i}}$', size=18, bbox=props, \
              transform=axes[j].transAxes, horizontalalignment='center', \
              verticalalignment='center')
axes[j].minorticks_off()
# axes[j].set_yticks([1.e-4, 1.e-2, 1.0])
axes[j].yaxis.grid(which='major', linestyle=':')

plt.tight_layout()
# plt.show()
plotname = ''.join(['./plot_data/std_dev_accuracy/std_dev_approximation_error_bar_systemsize', str(systemsize_arr[i]), '.pdf'])
f.savefig(plotname, format='pdf')
