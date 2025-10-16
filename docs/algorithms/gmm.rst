Gaussian Mixture Model (GMM)
============================

Gaussian Mixture Model (GMM) is a probabilistic model that assumes the data comes from
a mixture of Gaussian distributions with unknown parameters. Unlike K-means, GMM provides
soft clustering where each data point has a probability of belonging to each cluster.

Theoretical Background
----------------------

Model Definition
~~~~~~~~~~~~~~~~

A Gaussian Mixture Model with :math:`K` components is defined as:

.. math::

   p(x) = \sum_{k=1}^{K} \pi_k \mathcal{N}(x | \mu_k, \Sigma_k)

Where:

* :math:`\pi_k` are the mixing coefficients (:math:`\sum_{k=1}^{K} \pi_k = 1`)
* :math:`\mathcal{N}(x | \mu_k, \Sigma_k)` is the :math:`k`-th Gaussian component
* :math:`\mu_k` is the mean of the :math:`k`-th component
* :math:`\Sigma_k` is the covariance matrix of the :math:`k`-th component

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

Log-Likelihood
~~~~~~~~~~~~~~

The algorithm maximizes the log-likelihood:

.. math::

   \mathcal{L} = \sum_{n=1}^{N} \log \left( \sum_{k=1}^{K} \pi_k \mathcal{N}(x_n | \mu_k, \Sigma_k) \right)

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
