Gaussian Mixture Model (GMM)
============================

Gaussian Mixture Model (GMM) is a probabilistic model that assumes the data comes from
a mixture of Gaussian distributions with unknown parameters. Unlike K-means, GMM provides
soft clustering where each data point has a probability of belonging to each cluster.

Theoretical Background
----------------------

Latent Variable Model
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In general, probability of observing a data point :math:`x` under unknown unobserved latent variables :math:`z` is given as marginal probability marginalizing over :math:`z`:

.. math::

   p(x) = \sum_z P(x, z)

which can be decomposed using conditional probability as:

.. math::

   p(x) = \sum_z p(x | z) p(z)

We call x as observed variables and z as latent variables, and complete-data is refers to the combination of both observed and latent variables, i.e., the pair (x, z).


Evidence Lower Bound (ELBO) and EM Algorithm
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The goal is to find the parameters that maximize the likelihood of the observed data. This is typically done using the Expectation-Maximization (EM) algorithm.
We can derive iterative update rules for the parameters by locally maximizing the expected complete-data log-likelihood.


.. math::
   \log p(x | \theta) = \mathbb{E}_{z \sim q(z)} \bigl[\frac{\log p(x, z | \theta)}{q(z)}\bigr] + KL_z \bigl(q(z) || p(z | x, \theta)\bigr)

The first term is called ELBO (Evidence Lower Bound) because it lower-bounds the log-likelihood.
We can confirm this equality by direct expansion of ELBO as follows:

.. math::
   \begin{aligned}
   \mathbb{E}_{z \sim q(z)} \bigl[\frac{\log p(x, z | \theta)}{q(z)}\bigr] &= \sum_z q(z) \frac{\log p(x, z | \theta)}{q(z)} \\
   &= \sum_z q(z) \log p(x | \theta) + \sum_z q(z) \log p(z | x, \theta) - \sum_z q(z) \log q(z) \\
   &= \log p(x | \theta) + \sum_z q(z) \log \frac{P(z|x, \theta)}{q(z)}
   \end{aligned}

The simple strategy to maximize the log-likelihood is to iteratively maximize the ELBO by alternating between the following two steps:

* Estimate :math:`q(z)` using current parameters :math:`\theta`
* Estimate :math:`\theta'` using given :math:`q(z)`

Since KL divergence is always non-negative and equals zero when :math:`q(z) = p(z | x, \theta)`, the first step above is equivalent to computing the posterior distribution over latent variables given the data and current parameters.

.. math::
   q(z) = p(z | x, \theta^{old}) = \frac{p(x, z | \theta^{old})}{p(x | \theta^{old})}

The second step is equivalent to maximizing the expected complete-data log-likelihood:

.. math::

   \theta' = \arg \max_{\theta}  \mathbb{E}_{z \sim q(z)} \bigl[\log p(x, z | \theta)\bigr]

This is the EM algorithm. Note that :math:`q(z)` is not expressed as closed form in general and cannot calculated directly. EM algorithm can be only applied when the posterior distribution :math:`p(z | x, \theta)` can be computed in closed form.

Traditionally, EM algorithm is derived by decomposing the log-likelihood into two terms and Jensen's inequality is used as follows:

.. math::

   \begin{aligned}
   \log p(x \mid \theta) &= \log \sum_z p(x, z \mid \theta) \\
   &= \log \sum_z p(x \mid z, \theta)\, p(z \mid \theta) \\
   &= \sum_z p(z \mid x, \theta)\, \log p(x \mid z, \theta) + \sum_z p(z \mid x, \theta)\, \log \frac{p(z \mid \theta)}{p(z \mid x, \theta)} \\
   &= \mathbb{E}_{z \mid x, \theta} \bigl[\log p(x \mid z, \theta)\bigr] + \mathbb{E}_{z \mid x, \theta} \Biggl[\log \frac{p(z \mid \theta)}{p(z \mid x, \theta)}\Biggr] \\
   &= \mathbb{E}_{z \mid x, \theta} \bigl[\log p(x \mid z, \theta)\bigr] + \mathbb{E}_{z \mid x, \theta} \bigl[\log p(z \mid \theta)\bigr] - \mathbb{E}_{z \mid x, \theta} \bigl[\log p(z \mid x, \theta)\bigr] \\
   &= Q(\theta \mid \theta^{old}) + H\bigl(p(z \mid x, \theta)\bigr)
   \end{aligned}

Where:

* :math:`Q(\theta | \theta^{old}) = \mathbb{E}_{z | x, \theta^{old}} [\log p(x, z | \theta)]` is the expected complete-data log-likelihood
* :math:`H(p(z | x, \theta))` is the entropy of the posterior distribution over latent variables
* :math:`\theta^{old}` are the parameters from the previous iteration
* :math:`\theta` are the parameters to be optimized
* :math:`p(z | x, \theta)` is the posterior distribution over latent variables given the data and current parameters
* :math:`p(x, z | \theta)` is the complete-data likelihood
* :math:`p(x | z, \theta)` is the likelihood of the data given the latent variables and parameters
* :math:`p(z | \theta)` is the prior distribution over latent variables given the
* parameters
* :math:`\mathbb{E}_{z | x, \theta}` denotes expectation with respect to the posterior distribution over latent variables given the data and current parameters
* :math:`\log` is the natural logarithm
* `:math:`\sum_z` denotes summation over all possible values of the latent variables
* :math:`p(x)` is the marginal likelihood of the observed data
* :math:`p(z)` is the prior distribution over latent variables
* :math:`p(x | z)` is the likelihood of the data given the latent variables
* :math:`p(z | x)` is the posterior distribution over latent variables given the data
* :math:`\theta` represents the model parameters



Model Definition of GMM
~~~~~~~~~~~~~~~~~~~~~~~


A Gaussian Mixture Model with :math:`K` components is defined as:

.. math::

   p(x) = \sum_{k=1}^{K} \pi_k \mathcal{N}(x | \mu_k, \Sigma_k)

Where:

* :math:`\pi_k` are the mixing coefficients (:math:`\sum_{k=1}^{K} \pi_k = 1`)
* :math:`\mathcal{N}(x | \mu_k, \Sigma_k)` is the :math:`k`-th Gaussian component
* :math:`\mu_k` is the mean of the :math:`k`-th component
* :math:`\Sigma_k` is the covariance matrix of the :math:`k`-th component

Log-Likelihood
~~~~~~~~~~~~~~

The algorithm maximizes the log-likelihood:

.. math::

   \mathcal{L} = \sum_{n=1}^{N} \log \left( \sum_{k=1}^{K} \pi_k \mathcal{N}(x_n | \mu_k, \Sigma_k) \right)

Expectation-Maximization Algorithm
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

GMM parameters are learned using the EM algorithm:

**E-step (Expectation)**: Compute posterior probabilities

.. math::

   \gamma_{nk} = \frac{\pi_k \mathcal{N}(x_n | \mu_k, \Sigma_k)}{\sum_{j=1}^{K} \pi_j \mathcal{N}(x_n | \mu_j, \Sigma_j)}

**M-step (Maximization)**: Update parameters

.. math::

   \pi_k^{new} &= \frac{1}{N} \sum_{n=1}^{N} \gamma_{nk} \\
   \mu_k^{new} &= \frac{\sum_{n=1}^{N} \gamma_{nk} x_n}{\sum_{n=1}^{N} \gamma_{nk}} \\
   \Sigma_k^{new} &= \frac{\sum_{n=1}^{N} \gamma_{nk} (x_n - \mu_k^{new})(x_n - \mu_k^{new})^T}{\sum_{n=1}^{N} \gamma_{nk}}

Implementation Features
-----------------------

Key Characteristics
~~~~~~~~~~~~~~~~~~~

* **Soft Clustering**: Provides probability distributions over clusters
* **Flexible Cluster Shapes**: Can model elliptical clusters
* **Model Selection**: Supports different numbers of components
* **Convergence Monitoring**: Tracks log-likelihood improvement

GMM vs K-means
~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1

   * - Aspect
     - K-means
     - GMM
   * - Clustering Type
     - Hard assignment
     - Soft assignment (probabilistic)
   * - Cluster Shape
     - Spherical (typically)
     - Elliptical (flexible)
   * - Output
     - Cluster labels
     - Probability distributions
   * - Computational Cost
     - Lower
     - Higher

Usage Examples
--------------

Basic Usage
~~~~~~~~~~~

.. code-block:: python

   import numpy as np
   from cmla.models.gmm import GMM

   # Generate sample data
   np.random.seed(42)
   data = np.random.randn(200, 2)
   data[:100] += [3, 3]  # First cluster
   data[100:] += [-2, -2]  # Second cluster

   # Fit GMM
   gmm = GMM(n_components=2)
   gmm.fit(data)

   # Get cluster assignments
   labels = gmm.predict(data)

   # Get probabilities
   probabilities = gmm.predict_proba(data)
   print(f"Sample probabilities: {probabilities[:5]}")

Advanced Features
~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Access model parameters
   print(f"Mixing coefficients: {gmm.weights_}")
   print(f"Means: {gmm.means_}")
   print(f"Covariances shape: {gmm.covariances_.shape}")

   # Compute log-likelihood
   log_likelihood = gmm.score(data)
   print(f"Log-likelihood: {log_likelihood}")

   # Sample from the model
   samples, component_labels = gmm.sample(100)

Model Selection
~~~~~~~~~~~~~~~

.. code-block:: python

   from sklearn.model_selection import cross_val_score
   import matplotlib.pyplot as plt

   # Test different numbers of components
   n_components_range = range(1, 10)
   aic_scores = []
   bic_scores = []

   for n_components in n_components_range:
       gmm = GMM(n_components=n_components)
       gmm.fit(data)
       aic_scores.append(gmm.aic(data))
       bic_scores.append(gmm.bic(data))

   # Plot information criteria
   plt.figure(figsize=(10, 6))
   plt.plot(n_components_range, aic_scores, label='AIC')
   plt.plot(n_components_range, bic_scores, label='BIC')
   plt.xlabel('Number of Components')
   plt.ylabel('Information Criterion')
   plt.legend()
   plt.title('Model Selection for GMM')
   plt.show()

Visualization
~~~~~~~~~~~~~

.. code-block:: python

   from cmla.plots.kmeans_plot import plot_data_with_centroid
   import matplotlib.pyplot as plt

   # Plot results
   plot_data_with_centroid(data, labels, gmm.means_)
   plt.title('GMM Clustering Results')

   # Add confidence ellipses
   from matplotlib.patches import Ellipse
   for i, (mean, cov) in enumerate(zip(gmm.means_, gmm.covariances_)):
       eigenvalues, eigenvectors = np.linalg.eigh(cov)
       angle = np.degrees(np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0]))
       width, height = 2 * np.sqrt(eigenvalues)
       ellipse = Ellipse(mean, width, height, angle=angle,
                        facecolor='none', edgecolor=f'C{i}', linewidth=2)
       plt.gca().add_patch(ellipse)

   plt.show()

Mathematical Properties
-----------------------

Convergence
~~~~~~~~~~~

The EM algorithm is guaranteed to converge to a local maximum of the likelihood function.
Key properties:

* **Monotonic**: Log-likelihood never decreases
* **Local Optima**: May converge to local rather than global maximum
* **Initialization Sensitive**: Results depend on initial parameter values

Model Complexity
~~~~~~~~~~~~~~~~~

The number of parameters in a GMM grows with:

* Number of components: :math:`K`
* Dimensionality: :math:`D`
* Total parameters: :math:`K \times (1 + D + \frac{D(D+1)}{2}) - 1`

Information Criteria
~~~~~~~~~~~~~~~~~~~~

For model selection:

**Akaike Information Criterion (AIC)**:

.. math::

   AIC = -2\mathcal{L} + 2p

**Bayesian Information Criterion (BIC)**:

.. math::

   BIC = -2\mathcal{L} + p \ln(N)

Where :math:`p` is the number of parameters and :math:`N` is the number of data points.

API Reference
-------------

.. autoclass:: cmla.models.gmm.GMM
   :members:
   :undoc-members:
   :show-inheritance:

Practical Considerations
------------------------

Initialization
~~~~~~~~~~~~~~

* **K-means++**: Initialize with K-means results
* **Random**: Random initialization (may need multiple runs)
* **User-specified**: Provide initial parameters

Convergence Criteria
~~~~~~~~~~~~~~~~~~~~

* **Log-likelihood tolerance**: Stop when improvement is below threshold
* **Maximum iterations**: Prevent infinite loops
* **Parameter tolerance**: Stop when parameters change minimally

Implementation Details
~~~~~~~~~~~~~~~~~~~~~~

Implementaion using matrix operations for efficiency. Utilizes NumPy for numerical computations and SciPy for statistical functions. Regularization techniques are applied to covariance matrices



Common Issues
~~~~~~~~~~~~~

* **Singular covariance matrices**: Use regularization
* **Empty clusters**: Reinitialize problematic components
* **Overfitting**: Use information criteria for model selection

See Also
--------

* :doc:`kmeans` - K-means clustering for comparison
* :doc:`../api/models` - Complete API reference
* :doc:`../tutorials/probabilistic_clustering` - Tutorial on probabilistic clustering
