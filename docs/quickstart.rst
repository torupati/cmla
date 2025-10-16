Quick Start Guide
=================

This guide will help you get started with CMLA quickly.

First Steps
-----------

1. **Installation**

   .. code-block:: bash

      git clone https://github.com/torupati/cmla.git
      cd cmla
      uv sync
      uv pip install -e .

2. **Basic Import**

   .. code-block:: python

      import numpy as np
      import matplotlib.pyplot as plt
      from cmla.models import kmeans, gmm, hmm

Quick Examples
--------------

K-means Clustering
~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from cmla.models.kmeans import kmeans_clustering
   import numpy as np

   # Generate sample data
   np.random.seed(42)
   data = np.random.randn(150, 2)
   data[:50] += [2, 2]     # Cluster 1
   data[50:100] += [-2, 2] # Cluster 2
   data[100:] += [0, -2]   # Cluster 3

   # Perform clustering
   centroids, labels = kmeans_clustering(data, k=3)

   # Visualize results
   import matplotlib.pyplot as plt
   colors = ['red', 'blue', 'green']
   for i in range(3):
       cluster_data = data[labels == i]
       plt.scatter(cluster_data[:, 0], cluster_data[:, 1],
                  c=colors[i], label=f'Cluster {i}')
       plt.scatter(centroids[i, 0], centroids[i, 1],
                  c='black', marker='x', s=100)
   plt.legend()
   plt.title('K-means Clustering Results')
   plt.show()

Gaussian Mixture Model
~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from cmla.models.gmm import GMM
   import numpy as np

   # Generate sample data
   np.random.seed(123)
   data = np.vstack([
       np.random.multivariate_normal([0, 0], [[1, 0], [0, 1]], 100),
       np.random.multivariate_normal([3, 3], [[1, 0], [0, 1]], 100)
   ])

   # Fit GMM
   gmm = GMM(n_components=2)
   gmm.fit(data)

   # Get soft assignments
   probabilities = gmm.predict_proba(data)
   labels = gmm.predict(data)

   print(f"Mixing coefficients: {gmm.weights_}")
   print(f"Log-likelihood: {gmm.score(data)}")

Hidden Markov Model
~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from cmla.models.hmm import HMM
   import numpy as np

   # Create HMM
   hmm = HMM(num_states=2, num_observations=2)

   # Set up a simple model (Fair/Biased coin example)
   hmm.transition_matrix = np.array([
       [0.95, 0.05],  # Fair -> Fair, Fair -> Biased
       [0.10, 0.90]   # Biased -> Fair, Biased -> Biased
   ])

   hmm.observation_matrix = np.array([
       [0.5, 0.5],    # Fair coin: equal probability
       [0.1, 0.9]     # Biased coin: mostly heads
   ])

   hmm.initial_state_probability = np.array([0.5, 0.5])

   # Observation sequence (0=Tails, 1=Heads)
   observations = [1, 1, 0, 1, 1, 1, 0, 1, 1, 1]

   # Find most likely state sequence
   path, prob = hmm.viterbi(observations)
   print(f"Observations: {observations}")
   print(f"Most likely states: {path}")
   print(f"Probability: {prob}")

Command Line Tools
------------------

CMLA includes command-line tools for quick analysis:

K-means CLI
~~~~~~~~~~~

.. code-block:: bash

   # Cluster random data
   uv run python scripts/kmeans_cli.py --random-data --clusters 3 --samples 200

   # Cluster data from CSV file
   uv run python scripts/kmeans_cli.py --data-file mydata.csv --clusters 4

HMM CLI
~~~~~~~

.. code-block:: bash

   # Run Viterbi algorithm
   uv run python scripts/hmm_cli.py --viterbi --observations "0 1 1 0 1"

   # Train HMM model
   uv run python scripts/hmm_cli.py --train --data-file observations.txt

Common Workflows
----------------

Data Analysis Pipeline
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import numpy as np
   import matplotlib.pyplot as plt
   from cmla.models.kmeans import kmeans_clustering
   from cmla.models.gmm import GMM

   # 1. Load or generate data
   data = np.random.randn(200, 2)
   data[:100] += [2, 2]
   data[100:] += [-2, -2]

   # 2. Try different clustering approaches

   # K-means
   k_centroids, k_labels = kmeans_clustering(data, k=2)

   # GMM
   gmm = GMM(n_components=2)
   gmm.fit(data)
   g_labels = gmm.predict(data)

   # 3. Compare results
   fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

   # K-means results
   ax1.scatter(data[:, 0], data[:, 1], c=k_labels, cmap='viridis')
   ax1.scatter(k_centroids[:, 0], k_centroids[:, 1],
              c='red', marker='x', s=100)
   ax1.set_title('K-means')

   # GMM results
   ax2.scatter(data[:, 0], data[:, 1], c=g_labels, cmap='viridis')
   ax2.scatter(gmm.means_[:, 0], gmm.means_[:, 1],
              c='red', marker='x', s=100)
   ax2.set_title('GMM')

   plt.tight_layout()
   plt.show()

Time Series Analysis
~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from cmla.models.hmm import HMM
   import numpy as np

   # Generate synthetic time series
   np.random.seed(42)
   true_states = []
   observations = []

   current_state = 0
   for t in range(100):
       # State transition
       if current_state == 0:
           current_state = np.random.choice([0, 1], p=[0.9, 0.1])
       else:
           current_state = np.random.choice([0, 1], p=[0.2, 0.8])

       true_states.append(current_state)

       # Observation emission
       if current_state == 0:
           obs = np.random.choice([0, 1], p=[0.8, 0.2])
       else:
           obs = np.random.choice([0, 1], p=[0.3, 0.7])

       observations.append(obs)

   # Train HMM
   hmm = HMM(num_states=2, num_observations=2)
   log_likelihoods = hmm.train_baum_welch([observations], max_iterations=50)

   # Decode states
   predicted_states, _ = hmm.viterbi(observations)

   # Evaluate accuracy
   accuracy = np.mean(np.array(true_states) == np.array(predicted_states))
   print(f"State prediction accuracy: {accuracy:.2f}")

Next Steps
----------

* :doc:`algorithms/kmeans` - Detailed K-means documentation
* :doc:`algorithms/gmm` - Gaussian Mixture Model guide
* :doc:`algorithms/hmm` - Hidden Markov Model tutorial
* :doc:`api/models` - Complete API reference

Tips for Beginners
-------------------

1. **Start Simple**: Begin with K-means for clustering problems
2. **Visualize**: Always plot your data and results
3. **Parameter Tuning**: Try different numbers of clusters/components
4. **Model Selection**: Use information criteria for model comparison
5. **Validation**: Split your data for proper evaluation

Common Pitfalls
---------------

* **Scaling**: Normalize features for distance-based algorithms
* **Initialization**: Run algorithms multiple times with different initializations
* **Convergence**: Check if algorithms have converged properly
* **Overfitting**: Don't use too many components for small datasets
