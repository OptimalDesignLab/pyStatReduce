[![Build Status](https://travis-ci.com/OptimalDesignLab/pyStatReduce.svg?token=JPqRTsysF2SsyyyaMr8K&branch=master)](https://travis-ci.com/OptimalDesignLab/pyStatReduce)

# pyStatReduce

Python package for stochastic dimension reduction based on nonlinearity of a
quantity of interest. This package uses stochastic collocation to estimate
statistical moments of a quantity of interest (QoI).

## Dependencies
Besides `Numpy` and `SciPy`, this package depends on another python package
called `chaospy`, which can be installed using

```
pip install chaospy
```

The documentation of `chaospy` can be found at the following link
```
http://chaospy.readthedocs.io/en/master/
```

## Major Classes

Using `pyStatReduce` for approximating statistics involves working with objects
of the following major classes and their subclasses

* `QuantityOfInterest` : Base class for creating subclasses for a specific
  quantity of interest. examples of subclasses of `QuantityOfInterest` can be
  found in `src/examples` directory
* `StochasticCollocation` : Base class that handles uncertainty propagation
  using stochastic collocation. This is designed to handle multivariate uniform
  or normal distributions.
* `Dist` : Base class in `chaospy` upon different distributions have been defined.
  see [here](http://chaospy.readthedocs.io/en/master/distributions.html) for a
  complete list of available distributions.
* `DimensionReduction` : Base class that identifies the directions with highest
  nonlinearity in the QoI. This is done by computing the dominant eigenmodes of
  the Hessian of the QoI in the isoprobabilistic space.

## Sample script

A sample script for using the package has been provided below. However, the user
is recommended to look in the `test` to see the latest API for the package

```
systemsize = 4
eigen_decayrate = 2.0

# Create Hadmard Quadratic object
QoI = examples.HadamardQuadratic(systemsize, eigen_decayrate)

# Create stochastic collocation object
collocation = StochasticCollocation(3, "Normal")

# Create dimension reduction object
threshold_factor = 0.9
dominant_space = DimensionReduction(threshold_factor)

# Initialize chaospy distribution
std_dev = 0.2*np.ones(QoI.systemsize)
x = np.ones(QoI.systemsize)
jdist = cp.MvNormal(x, np.diag(std_dev))

# Get the eigenmodes of the Hessian product and the dominant indices
dominant_space.getDominantDirections(QoI, jdist)

mu_j = collocation.normalReduced(QoI, jdist, dominant_space)
```
