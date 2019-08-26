################################################################################
# plot_surrogate_subspace_angles.py
#
# This file plots the subspace angles created between a said number of
# eigenvectors of the Hessian of the Kriging surrogate and that of the complete
# model.
#
################################################################################

import numpy as np
from mpl_toolkits import mplot3d
import matplotlib
import matplotlib.pyplot as plt

iteration_array = [11, 15, 20, 25, 30, 35, 40, 46]
solution_dict = {11: np.array([22.58768443, 40.14984029, 45.77449285, 53.93839023, 60.81933739, 71.95296635, 75.21676673, 84.99440796, 85.95977056, 89.2749061 ]),
                 15: np.array([37.381385  , 38.7086864 , 39.48951855, 54.60781966, 62.1874709 , 70.88940688, 76.16215446, 82.30218943, 84.17992486, 89.06140591]),
                 20: np.array([36.42320838, 44.06796263, 47.67090685, 51.21224638, 60.57691783, 64.77522486, 75.05030471, 79.41531284, 82.60368997, 85.74611841]),
                 25: np.array([33.73482672, 43.35782435, 48.60873381, 53.77517933, 61.48233535, 62.53026329, 70.12255712, 77.75193158, 80.51015749, 85.93015959]),
                 30: np.array([36.35669815, 39.09838622, 48.67150623, 52.88427711, 61.94998387, 65.57447263, 71.90753567, 75.94736802, 82.13692482, 88.84181278]),
                 35: np.array([31.35350601, 38.43077776, 52.04789218, 57.99473094, 63.07924838, 66.55929435, 73.09415165, 73.55907362, 79.39252609, 88.87079526]),
                 40: np.array([36.57422202, 40.39846287, 51.33877396, 55.61298699, 59.99460451, 65.36285593, 69.5022202 , 75.11377689, 80.09107679, 87.90332549]),
                 46: np.array([36.46151106, 40.76188258, 47.51264469, 53.16838007, 60.06587121, 65.95610361, 72.64433204, 75.82369458, 79.97287075, 88.45803811])}

xvals = range(1, len(solution_dict[11])+1)

# Create a scatter plot
fname = "surrogate_subspace_angle_scatter.pdf"
fig = plt.figure('scatter', figsize=(7,6))
ax = plt.axes()
scatter_dict = {}
for i in solution_dict:
    scatter_dict[i] = ax.scatter(xvals, solution_dict[i], marker='o', label='{} Arnoldi iterations'.format(i))
ax.set_xlabel('subspace indices')
ax.set_ylabel('angles (degrees)')
ax.legend()
plt.xticks(xvals)
plt.tight_layout()
# plt.show()
plt.savefig(fname, format='pdf')
