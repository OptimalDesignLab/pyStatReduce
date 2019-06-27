# plot_hadamard_eigen_accuracy.py
import numpy as np
from mpl_toolkits import mplot3d
import matplotlib
import matplotlib.pyplot as plt

systemsize_arr = [64, 128, 256]
eigen_decayrate_arr = [2.0, 1.0, 0.5]
n_arnoldi_samples_arr = [11, 21, 31, 41, 51]
n_eigen_samples = [10, 20, 30, 40, 50]
# n_stddev_samples = 10

avg_err = np.zeros([len(systemsize_arr), len(eigen_decayrate_arr), len(n_arnoldi_samples_arr)])
err_bars = np.zeros([len(systemsize_arr), len(eigen_decayrate_arr), 2, len(n_arnoldi_samples_arr)])

for i in range(0, len(systemsize_arr)):
    for j in range(0, len(eigen_decayrate_arr)):
        dirname = ''.join(['./plot_data/eigen_accuracy/', str(systemsize_arr[i]), '/'])
        fname1 = ''.join([dirname, 'avg_err_decay', str(eigen_decayrate_arr[j]), '.txt'])
        fname2 = ''.join([dirname, 'max_err_decay', str(eigen_decayrate_arr[j]), '.txt'])
        fname3 = ''.join([dirname, 'min_err_decay', str(eigen_decayrate_arr[j]), '.txt'])

        # Read data
        avg_err_vec = np.loadtxt(fname1, delimiter=',')
        max_err_vec = np.loadtxt(fname2, delimiter=',')
        min_err_vec = np.loadtxt(fname3, delimiter=',')

        avg_err[i,j,:] = avg_err_vec
        err_bars[i,j,0,:] = min_err_vec
        err_bars[i,j,1,:] = max_err_vec

# Plot
titlename = ''.join(['systemsize = ', str(systemsize_arr[i]), 'eigen_decayrate = ', str(eigen_decayrate_arr[j])])
plotname = ''.join(['./plot_data/eigen_accuracy/approximation_error_bar_decay', str(systemsize_arr[j]), '.pdf'])

plt.rc('text', usetex=True)
matplotlib.rcParams['mathtext.fontset'] = 'cm'
matplotlib.rcParams.update({'font.size': 18})

i = 2
f1, axes1 = plt.subplots(1,3, sharey=True , figsize=(9,4))
plt.setp(axes1[:], xticks=[10, 20, 30, 40, 50], yticks=[1.e-9, 1.e-7, 1.e-5, 1.e-3, 0.1, 1])
#   f1.suptitle(''.join(['system size = ', str(systemsize_arr[i])]  ))
props = dict(boxstyle='round', facecolor='white')


j = 0
axes1[j].errorbar(n_arnoldi_samples_arr, avg_err[i,j,:], yerr=err_bars[i,j,:,:], fmt='-o', capsize=6)
axes1[j].set_yscale("log", nonposy='clip')
axes1[j].set_ylabel(r'eigenvalue accuracy, $\tau$')
axes1[j].set_ylim(1.e-10, 1.0)
axes1[j].set_yticks([1.e-9, 1.e-7, 1.e-5, 1.e-3, 0.1, 1])
axes1[j].text(0.5,1,r'$\lambda_i = \frac{1}{i^2}$', size=18, bbox=props, \
              transform=axes1[j].transAxes, horizontalalignment='center', \
              verticalalignment='center')
axes1[j].set_xlabel('Arnoldi iterations')
axes1[j].yaxis.grid(which='major', linestyle=':')

j = 1
axes1[j].errorbar(n_arnoldi_samples_arr, avg_err[i,j,:], yerr=err_bars[i,j,:,:], fmt='-o', capsize=6)
axes1[j].set_yscale("log", nonposy='clip')
titlename = ''.join(['system size = ', str(systemsize_arr[i])])
axes1[j].set_xlabel('Arnoldi iterations')
axes1[j].text(0.5,1,r'$\lambda_i = \frac{1}{i}$', size=18, bbox=props, \
              transform=axes1[j].transAxes, horizontalalignment='center', \
              verticalalignment='center')
axes1[j].yaxis.grid(which='major', linestyle=':')

j = 2
axes1[j].errorbar(n_arnoldi_samples_arr, avg_err[i,j,:], yerr=err_bars[i,j,:,:], fmt='-o', capsize=6)
axes1[j].set_yscale("log", nonposy='clip')
axes1[j].text(0.5,1,r'$\lambda_i = \frac{1}{\sqrt{i}}$', size=18, bbox=props, \
              transform=axes1[j].transAxes, horizontalalignment='center', \
              verticalalignment='center')
axes1[j].set_xlabel('Arnoldi iterations')
axes1[j].yaxis.grid(which='major', linestyle=':')

plt.tight_layout(pad=1, w_pad=0)# , w_pad=0.5, h_pad=0.5)
plotname = ''.join(['./plot_data/eigen_accuracy/approximation_error_bar_decay', str(systemsize_arr[i]), '.pdf'])
f1.savefig(plotname, format='pdf')
