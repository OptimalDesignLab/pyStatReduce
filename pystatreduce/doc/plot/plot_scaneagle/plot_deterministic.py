# Plot the following outputs from run_scaneagle.py
# prob['wing.geometry.twist_bsp.twist']
# prob['wing.struct_setup.nodes.nodes']

import numpy as np
from mpl_toolkits import mplot3d
import matplotlib
import matplotlib.pyplot as plt

dirname = '../../../src/openmdao/'
fname_twist = dirname + 'optim_twist_val.txt'
fname_span = dirname + 'span_val.txt'
read_twist_vals = np.loadtxt(fname_twist, delimiter=',')
read_span = np.loadtxt(fname_span)
span_opp = abs(read_span)

# Step 1: flip both the arrays
twist = np.append(read_twist_vals, np.flip(read_twist_vals))
span = np.append(read_span, np.flip(span_opp))

fname = "run_scaneagle_plot.pdf"
plt.rc('text', usetex=True)
matplotlib.rcParams['mathtext.fontset'] = 'cm'
fig = plt.figure("twist plot")
ax = plt.axes()
lp = ax.plot(span, twist)
ax.set_xlabel('Span')
ax.set_ylabel('Twist')
fig.savefig(fname, format="pdf")
