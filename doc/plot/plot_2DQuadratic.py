# plot
import numpy as np
from mpl_toolkits import mplot3d
import matplotlib
import matplotlib.pyplot as plt


def lineplot_2DQuadratic():
    fname = "max_err_02_01.txt"
    # fname = "max_err_01_01.txt"
    approxn_error = np.loadtxt(fname)
    # approxn_error = np.array([0.00492413, 0.00510523, 0.00569139, 0.00682811,
    #                           0.00882148, 0.01225806, 0.01819729, 0.02828646,
    #                           0.04390978, 0.06219219, 0.07132487])
    n_theta = 11
    theta = np.linspace(0, 90, num=n_theta)

    matplotlib.rcParams['mathtext.fontset'] = 'cm'
    plt.rc('text', usetex=True)
    matplotlib.rcParams.update({'font.size': 14})
    plt.figure(figsize=(4,3))
    plt.plot(theta, approxn_error, color="black")
    plt.ylim(0.0, 0.08)
    plt.yticks([0.0, 0.025, 0.05, 0.075])
    plt.xticks([0, 30, 60, 90])
    plt.ylabel(r'approximation error, $\epsilon$')
    # plt.ylabel(r'relative error in the mean', fontsize=16)
    plt.xlabel(r'$\theta^{\circ}$')
    plt.tight_layout()
    # plt.savefig("2DQuadratic_approxn_err_01_01.pdf", format="pdf")
    plt.savefig("2DQuadratic_approxn_err_02_01.pdf", format="pdf")

def surfaceplot_2DQuadratic():

    n_theta = 6
    n_samples = 50
    n_ratios = 16
    theta = np.linspace(0, 90, num=n_theta)
    std_dev_ratios = np.linspace(1, 20, num=n_ratios)
    m_theta, m_std_dev_ratios = np.meshgrid(theta, std_dev_ratios)
    print "m_theta.shape = ", m_theta.shape
    print "m_std_dev_ratios.shape = ", m_std_dev_ratios.shape

    # REad approximation error from file
    fname = "max_err_sigma_ratio.txt"
    approxn_error = np.loadtxt(fname)
    print "approxn_error.shape", approxn_error.shape, "\n"

    # Plot
    matplotlib.rcParams['mathtext.fontset'] = 'cm'
    plt.rc('text', usetex=True)
    matplotlib.rcParams.update({'font.size': 18})
    fig = plt.figure(figsize=(7,5))
    ax = fig.add_subplot(111, projection='3d')
    ax.view_init(30, 225)
    ax.plot_surface(m_theta, m_std_dev_ratios, approxn_error.T, cmap="coolwarm", edgecolors="k")
    plt.yticks([0, 5, 10, 15, 20])
    ax.set_zticks([0, 0.2, 0.4, 0.6, 0.8, 1])
    ax.set_zlim(0.0, 1.0)
    plt.xlabel(r'$\theta^{\circ}$', labelpad=10)
    plt.ylabel(r'$\frac{\sigma_{1}}{\sigma_{2}}$', labelpad=10)
    ax.set_zlabel(r'Approximation Error, $\epsilon$')
    plt.tight_layout()
    plt.savefig("2DQuadratic_std_dev_ratio.pdf", format="pdf")

# surfaceplot_2DQuadratic()
lineplot_2DQuadratic()
