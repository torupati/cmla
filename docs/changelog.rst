Changelog
=========

Version 0.1.0 (2025-10-17)
---------------------------

Initial release of CMLA.

Features
~~~~~~~~

* **K-means Clustering**

  * EM algorithm implementation
  * Multiple covariance matrix types (FULL, DIAG, SPHERICAL)
  * Command-line interface
  * Visualization utilities

* **Gaussian Mixture Model (GMM)**

  * EM algorithm for parameter estimation
  * Soft clustering capabilities
  * Model selection tools (AIC, BIC)
  * Integration with scikit-learn interface

* **Hidden Markov Model (HMM)**

  * Forward algorithm for probability computation
  * Viterbi algorithm for state sequence decoding
  * Baum-Welch algorithm for parameter learning
  * Support for multiple observation sequences

* **Command-Line Tools**

  * ``kmeans_cli.py`` - K-means clustering tool
  * ``hmm_cli.py`` - HMM analysis tool
  * ``sampler_cli.py`` - MCMC sampling tool

* **Documentation**

  * Comprehensive Sphinx documentation
  * Algorithm explanations with mathematical details
  * Usage examples and tutorials
  * API reference

* **Testing**

  * Unit tests for all algorithms
  * Integration tests for CLI tools
  * Continuous integration setup

Dependencies
~~~~~~~~~~~~

* numpy >= 2.3.4
* scipy >= 1.11.0
* matplotlib >= 3.7.0
* tqdm >= 4.66.0

Development
~~~~~~~~~~~

* Pre-commit hooks for code quality
* Ruff for formatting and linting
* pytest for testing
* Sphinx for documentation
