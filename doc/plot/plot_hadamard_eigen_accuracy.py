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

for i in xrange(0, len(systemsize_arr)):
    for j in xrange(0, len(eigen_decayrate_arr)):
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
matplotlib.rcParams.update({'font.size': 22})

i = 1
f1, axes1 = plt.subplots(1,3, sharey=True , figsize=(9,5.5))
plt.setp(axes1, xticks=[10, 20, 30, 40, 50], yticks=[0.0, 0.25, 0.5, 0.75, 1])
f1.suptitle(''.join(['system size = ', str(systemsize_arr[i])]  ))

j = 0
axes1[j].errorbar(n_arnoldi_samples_arr, avg_err[i,j,:], yerr=err_bars[i,j,:,:], fmt='-o', barsabove=True)
# axes1[j].set_xlabel('Arnoldi estimated eigen pairs')
axes1[j].set_ylabel(r'$ || \lambda_{1:10} - \lambda_{1:10}^{exact} ||_2$')
axes1[j].set_ylim(-0.02, 0.75)
axes1[j].set_title(r'$\lambda_i = \frac{1}{i^2}$')

j = 1
axes1[j].errorbar(n_arnoldi_samples_arr, avg_err[i,j,:], yerr=err_bars[i,j,:,:], fmt='-o', barsabove=True)
titlename = ''.join(['system size = ', str(systemsize_arr[i])])
axes1[j].set_xlabel(r'No. of Arnoldi samples')
axes1[j].set_title(r'$\lambda_i = \frac{1}{i}$')

j = 2
axes1[j].errorbar(n_arnoldi_samples_arr, avg_err[i,j,:], yerr=err_bars[i,j,:,:], fmt='-o', barsabove=True)
# axes1[j].set_xlabel('Arnoldi estimated eigen pairs')
axes1[j].set_title(r'$\lambda_i = \frac{1}{\sqrt{i}}$')

plt.tight_layout(pad=2, w_pad=-1)# , w_pad=0.5, h_pad=0.5)
plotname = ''.join(['./plot_data/eigen_accuracy/approximation_error_bar_decay', str(systemsize_arr[i]), '.pdf'])
f1.savefig(plotname, format='pdf')




"""
f, axes = plt.subplots(3, 3, figsize=(8.3,11.7))
matplotlib.rcParams['mathtext.fontset'] = 'cm'

i = 0;j = 0
axes[i,j].errorbar(n_arnoldi_samples_arr, avg_err[i,j,:], yerr=err_bars[i,j,:,:], fmt='-o', barsabove=True)
titlename = ''.join(['systemsize = ', str(systemsize_arr[i]), ', eigen_decayrate = ', str(eigen_decayrate_arr[j])])
axes[i,j].set_title(titlename)

i = 0;j = 1
axes[i,j].errorbar(n_arnoldi_samples_arr, avg_err[i,j,:], yerr=err_bars[i,j,:,:], fmt='-o', barsabove=True)
titlename = ''.join(['systemsize = ', str(systemsize_arr[i]), ', eigen_decayrate = ', str(eigen_decayrate_arr[j])])
axes[i,j].set_title(titlename)

i = 0;j = 2
axes[i,j].errorbar(n_arnoldi_samples_arr, avg_err[i,j,:], yerr=err_bars[i,j,:,:], fmt='-o', barsabove=True)
titlename = ''.join(['systemsize = ', str(systemsize_arr[i]), ', eigen_decayrate = ', str(eigen_decayrate_arr[j])])
axes[i,j].set_title(titlename)

i = 1;j = 0
axes[i,j].errorbar(n_arnoldi_samples_arr, avg_err[i,j,:], yerr=err_bars[i,j,:,:], fmt='-o', barsabove=True)
titlename = ''.join(['systemsize = ', str(systemsize_arr[i]), ', eigen_decayrate = ', str(eigen_decayrate_arr[j])])
axes[i,j].set_title(titlename)

i = 1;j = 1
axes[i,j].errorbar(n_arnoldi_samples_arr, avg_err[i,j,:], yerr=err_bars[i,j,:,:], fmt='-o', barsabove=True)
titlename = ''.join(['systemsize = ', str(systemsize_arr[i]), ', eigen_decayrate = ', str(eigen_decayrate_arr[j])])
axes[i,j].set_title(titlename)

i = 1;j = 2
axes[i,j].errorbar(n_arnoldi_samples_arr, avg_err[i,j,:], yerr=err_bars[i,j,:,:], fmt='-o', barsabove=True)
titlename = ''.join(['systemsize = ', str(systemsize_arr[i]), ', eigen_decayrate = ', str(eigen_decayrate_arr[j])])
axes[i,j].set_title(titlename)

i = 2;j = 0
axes[i,j].errorbar(n_arnoldi_samples_arr, avg_err[i,j,:], yerr=err_bars[i,j,:,:], fmt='-o', barsabove=True)
titlename = ''.join(['systemsize = ', str(systemsize_arr[i]), ', eigen_decayrate = ', str(eigen_decayrate_arr[j])])
axes[i,j].set_title(titlename)

i = 2;j = 1
axes[i,j].errorbar(n_arnoldi_samples_arr, avg_err[i,j,:], yerr=err_bars[i,j,:,:], fmt='-o', barsabove=True)
titlename = ''.join(['systemsize = ', str(systemsize_arr[i]), ', eigen_decayrate = ', str(eigen_decayrate_arr[j])])
axes[i,j].set_title(titlename)

i = 2;j = 2
axes[i,j].errorbar(n_arnoldi_samples_arr, avg_err[i,j,:], yerr=err_bars[i,j,:,:], fmt='-o', barsabove=True)
titlename = ''.join(['systemsize = ', str(systemsize_arr[i]), ', eigen_decayrate = ', str(eigen_decayrate_arr[j])])
axes[i,j].set_title(titlename)

f.savefig(plotname, format='pdf')
"""
