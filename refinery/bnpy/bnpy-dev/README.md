**bnpy** is Bayesian nonparametric unsupervised machine learning for python.

Contact:  Mike Hughes. mike AT michaelchughes.com 

# About
This python module provides code for training popular Bayesian nonparametric models on massive datasets. **bnpy** supports the latest online learning algorithms as well as standard offline methods. 

Supported probabilistic models include

* Gaussian mixture models
    * standard parametric
    * nonparametric (Dirichlet Process)

Supported learning algorithms include:

* EM: expectation-maximization (offline)
* VB: variational Bayes (offline)
* moVB: memoized online VB
* soVB: stochastic online VB

These are all variants of *variational inference*, a family of optimization algorithms that perform coordinate ascent to learn parameters. 

# Quick Start

**bnpy** provides an easy command-line interface for launching experiments.

Train 8-component Gaussian mixture model via EM.
```
python -m bnpy.Run AsteriskK8 MixModel ZMGauss EM --K 8
```

Train Dirichlet-process Gaussian mixture model (DP-GMM) via variational bayes.
```
python -m bnpy.Run AsteriskK8 DPMixModel Gauss VB --K 8
```

Train DP-GMM via memoized online VB, with birth and merge moves
```
python -m bnpy.Run AsteriskK8 DPMixModel Gauss moVB --moves birth,merge
```

### Quick help
```
# print help message for required arguments
python -m bnpy.Run --help 
# print help message for specific keyword options for Gaussian mixture models
python -m bnpy.Run AsteriskK8 MixModel Gauss EM --kwhelp
```

# Installation

Follow the [installation instructions](https://bitbucket.org/michaelchughes/bnpy/wiki/Installation.md) on our project wiki.

# Documentation

All documentation can be found on the  [project wiki](https://bitbucket.org/michaelchughes/bnpy/wiki/Home.md).

Especially check out the [quick start demos](https://bitbucket.org/michaelchughes/bnpy/wiki/QuickStart/QuickStart.md)

# Target Audience

Primarly, we intend bnpy to be a platform for researchers. By gathering many learning algorithms and popular models in one convenient, modular repository, we hope to make it easier to compare and contrast approaches.

# Repository Organization
  bnpy/ module-specific code

  demodata/ example dataset scripts

  tests/ unit-tests for assuring code correctness. using nose package.

