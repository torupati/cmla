.. CMLA documentation master file, created by
   sphinx-quickstart on Fri Oct 17 00:09:42 2025.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

CMLA - Classical Machine Learning Algorithms
====================================================

.. image:: https://img.shields.io/badge/version-v0.1.0--alpha-orange
   :target: https://github.com/torupati/cmla/releases/tag/v0.1.0-alpha
   :alt: Version

.. image:: https://img.shields.io/badge/license-MIT-blue.svg
   :target: https://github.com/torupati/cmla/blob/main/LICENSE
   :alt: License

**CMLA** is a Python package for mathematical algorithms and data analysis,
providing implementations of fundamental machine learning and statistical algorithms
for educational and research purposes.

.. note::
   **Current Version: v0.1.0-alpha** (Initial Alpha Release)

   This is an early alpha release containing core implementations of K-means, GMM,
   and HMM algorithms with comprehensive documentation and CLI tools.

Features
--------

* **K-means Clustering**: Non-hierarchical clustering algorithm with multiple covariance matrix types
* **Gaussian Mixture Model (GMM)**: Probabilistic clustering using EM algorithm
* **Hidden Markov Model (HMM)**: Time series modeling with Forward-Backward, Viterbi, and Baum-Welch algorithms; discrete or Gaussian mixture (GMM-HMM) emissions
* **Command-Line Tools**: Easy-to-use CLI interfaces for all algorithms
* **Visualization**: Built-in plotting capabilities for results analysis

Quick Start
-----------

Installation
~~~~~~~~~~~~

.. code-block:: bash

   # Clone the repository
   git clone https://github.com/torupati/cmla.git
   cd cmla

   # Install using uv
   uv sync
   uv pip install -e .

Basic Usage
~~~~~~~~~~~

.. code-block:: python

   # K-means clustering
   from cmla.models.kmeans import kmeans_clustering
   import numpy as np

   data = np.random.randn(100, 2)
   centroids, labels = kmeans_clustering(data, k=3)

   # Gaussian Mixture Model
   from cmla.models.gmm import GMM
   gmm = GMM(n_components=3)
   gmm.fit(data)

   # Hidden Markov Model
   from cmla.models.hmm import HMM
   hmm = HMM(2, 2)  # 2 states, 2 symbols (random emission probabilities)
   observations = [0, 1, 0, 1, 1]
   path, log_prob = hmm.viterbi_search(observations)

   # HMM with Gaussian mixture emission (2-D observations)
   gmm_hmm = HMM(3, 2, observation_type="gmm", num_mixtures=2)

Command-Line Tools
~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   # K-means clustering with random data
   uv run python scripts/kmeans_cli.py --random-data --clusters 3

   # HMM Viterbi algorithm
   uv run python -m cmla.scripts.hmm_cli viterbi --model model.json --observations "0 1 0 1"

   # MCMC sampling
   uv run python scripts/sampler_cli.py --method metropolis --samples 1000

.. toctree::
   :maxdepth: 2
   :caption: User Guide:

   installation
   quickstart
   tutorials/index

.. toctree::
   :maxdepth: 2
   :caption: Algorithms:

   algorithms/kmeans
   algorithms/gmm
   algorithms/hmm

.. toctree::
   :maxdepth: 2
   :caption: API Reference:

   api/models
   api/plots
   api/scripts

.. toctree::
   :maxdepth: 1
   :caption: Development:

   contributing
   changelog
   design/gmm_hmm

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
