K-means Clustering
==================

K-means is a non-hierarchical clustering algorithm that partitions a dataset into k clusters.
The algorithm assigns each data point to the nearest cluster center (centroid) and aims to
minimize the within-cluster sum of squares.

Theoretical Background
----------------------

Objective Function
~~~~~~~~~~~~~~~~~~

K-means minimizes the following objective function:

.. math::

   J = \sum_{i=1}^{k} \sum_{x \in C_i} ||x - \mu_i||^2

Where:

* :math:`k` is the number of clusters
* :math:`C_i` is the :math:`i`-th cluster
* :math:`\mu_i` is the center of the :math:`i`-th cluster
* :math:`||x - \mu_i||^2` is the squared Euclidean distance

Algorithm
~~~~~~~~~

1. **Initialization**: Randomly place :math:`k` cluster centers
2. **Assignment Step**: Assign each data point to the nearest cluster center
3. **Update Step**: Update each cluster center to the mean of assigned data points
4. **Convergence**: Repeat steps 2-3 until cluster centers stop changing significantly

Mathematical Details
~~~~~~~~~~~~~~~~~~~~

The cluster center update formula:

.. math::

   \mu_i = \frac{1}{|C_i|} \sum_{x \in C_i} x

Data point cluster assignment:

.. math::

   c(x) = \arg\min_i ||x - \mu_i||^2

Implementation Features
-----------------------

Class Structure
~~~~~~~~~~~~~~~

The CMLA implementation provides the ``KmeansCluster`` class with the following features:

* EM (Expectation-Maximization) algorithm implementation
* Multiple covariance matrix types support
* Built-in visualization capabilities

Covariance Matrix Types
~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1

   * - Type
     - Description
     - Characteristics
   * - FULL
     - Full covariance matrix
     - Supports elliptical clusters
   * - DIAG
     - Diagonal covariance matrix
     - Axis-aligned elliptical clusters
   * - SPHERICAL
     - Spherical covariance matrix
     - Circular clusters (standard K-means)

Usage Examples
--------------

Basic Usage
~~~~~~~~~~~

.. code-block:: python

   import numpy as np
   from cmla.models.kmeans import KmeansCluster, kmeans_clustering

   # Generate sample data
   np.random.seed(42)
   data = np.random.randn(150, 2)
   data[:50] += [2, 2]   # Cluster 1
   data[50:100] += [-2, 2]  # Cluster 2
   data[100:] += [0, -2]    # Cluster 3

   # Method 1: Simple function interface
   centroids, labels = kmeans_clustering(data, k=3)
   print(f"Centroids: {centroids}")

   # Method 2: Class interface
   kmeans = KmeansCluster(num_clusters=3, feature_dimensionality=2)
   kmeans.fit(data)
   labels = kmeans.predict(data)

Advanced Usage
~~~~~~~~~~~~~~

.. code-block:: python

   # Using different covariance types
   kmeans_full = KmeansCluster(
       num_clusters=3,
       feature_dimensionality=2,
       cov_mode="FULL"
   )
   kmeans_full.fit(data)

   # With visualization
   from cmla.plots.kmeans_plot import plot_data_with_centroid
   import matplotlib.pyplot as plt

   plot_data_with_centroid(data, labels, centroids)
   plt.title('K-means Clustering Results')
   plt.show()

Command-Line Interface
----------------------

The K-means CLI tool provides easy access to clustering functionality:

.. code-block:: bash

   # Clustering with random data
   uv run python scripts/kmeans_cli.py --random-data --clusters 3 --samples 200

   # Clustering with CSV file
   uv run python scripts/kmeans_cli.py --data-file data.csv --clusters 4 --output results.json

   # Show help
   uv run python scripts/kmeans_cli.py --help

CLI Options
~~~~~~~~~~~

.. list-table::
   :header-rows: 1

   * - Option
     - Description
   * - ``--clusters, -k``
     - Number of clusters (default: 3)
   * - ``--data-file, -f``
     - Input data file (CSV format)
   * - ``--random-data``
     - Generate random data for demonstration
   * - ``--samples``
     - Number of random data samples (default: 100)
   * - ``--output, -o``
     - Output file for results

Parameters Reference
--------------------

KmeansCluster Class Parameters
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1

   * - Parameter
     - Type
     - Default
     - Description
   * - ``num_clusters``
     - int
     - Required
     - Number of clusters
   * - ``feature_dimensionality``
     - int
     - Required
     - Number of features/dimensions
   * - ``cov_mode``
     - str
     - "SPHERICAL"
     - Covariance matrix type
   * - ``max_iterations``
     - int
     - 100
     - Maximum number of iterations

API Reference
-------------

.. autoclass:: cmla.models.kmeans.KmeansCluster
   :members:
   :undoc-members:
   :show-inheritance:

.. autofunction:: cmla.models.kmeans.kmeans_clustering

See Also
--------

* :doc:`gmm` - Gaussian Mixture Models for probabilistic clustering
* :doc:`../api/models` - Complete API reference for all models
* :doc:`../tutorials/clustering_comparison` - Comparison of clustering algorithms
