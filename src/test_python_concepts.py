# Arbitrary runs:
# The purspose of this file is to test arbitrary python scripts/packages/and
# functions


import numpy as np
import chaospy as cp
from StochasticCollocation import StochasticCollocation


x = np.zeros(2)
theta = 0
sigma = np.array([0.2, 0.1])
tuple = (theta)
collocation = StochasticCollocation(3, "Normal")
print(collocation)
