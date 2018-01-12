# plot
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

fname = "max_err_02_01.dat"
# f = open(fname, "r")
approxn_error = np.array([0.00492413, 0.00510523, 0.00569139, 0.00682811,
                          0.00882148, 0.01225806, 0.01819729, 0.02828646,
                          0.04390978, 0.06219219, 0.07132487])
n_theta = 11
theta = np.linspace(0, 90, num=n_theta)

plt.figure(figsize=(5,5))
matplotlib.rcParams['mathtext.fontset'] = 'cm'
plt.plot(theta, approxn_error)
plt.rc('text', usetex=True)
plt.ylim(0.0, 0.08)
plt.ylabel('Approximation error')
plt.xlabel(r'$\theta^{\circ}$')
plt.savefig("2DQuadratic_approxn_err.pdf", format="pdf")
