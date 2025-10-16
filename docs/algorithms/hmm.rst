Hidden Markov Model (HMM)
=========================

Hidden Markov Models (HMMs) are statistical models for sequential data where the system
being modeled is assumed to be a Markov process with unobserved (hidden) states. HMMs are
widely used in speech recognition, bioinformatics, and time series analysis.

Theoretical Background
----------------------

Model Components
~~~~~~~~~~~~~~~~

An HMM is characterized by:

* **States**: :math:`S = \{s_1, s_2, \ldots, s_N\}` (hidden states)
* **Observations**: :math:`O = \{o_1, o_2, \ldots, o_M\}` (observable symbols)
* **Transition Probabilities**: :math:`A = \{a_{ij}\}` where :math:`a_{ij} = P(q_{t+1} = s_j | q_t = s_i)`
* **Emission Probabilities**: :math:`B = \{b_j(k)\}` where :math:`b_j(k) = P(o_t = v_k | q_t = s_j)`
* **Initial Probabilities**: :math:`\pi = \{\pi_i\}` where :math:`\pi_i = P(q_1 = s_i)`

Fundamental Assumptions
~~~~~~~~~~~~~~~~~~~~~~~

1. **Markov Property**: :math:`P(q_t | q_{t-1}, q_{t-2}, \ldots, q_1) = P(q_t | q_{t-1})`
2. **Output Independence**: :math:`P(o_t | q_t, q_{t-1}, \ldots, o_{t-1}, \ldots) = P(o_t | q_t)`

Three Fundamental Problems
--------------------------

1. Evaluation Problem
~~~~~~~~~~~~~~~~~~~~~

**Given**: Model parameters :math:`\lambda = (A, B, \pi)` and observation sequence :math:`O = o_1, o_2, \ldots, o_T`

**Find**: :math:`P(O | \lambda)` - the probability of the observation sequence

**Solution**: Forward Algorithm

.. math::

   \alpha_t(i) = P(o_1, o_2, \ldots, o_t, q_t = s_i | \lambda)

Recursion:

.. math::

   \alpha_1(i) &= \pi_i b_i(o_1) \\
   \alpha_{t+1}(j) &= \left[ \sum_{i=1}^{N} \alpha_t(i) a_{ij} \right] b_j(o_{t+1})

Final probability:

.. math::

   P(O | \lambda) = \sum_{i=1}^{N} \alpha_T(i)

2. Decoding Problem
~~~~~~~~~~~~~~~~~~~

**Given**: Model :math:`\lambda` and observation sequence :math:`O`

**Find**: Most likely state sequence :math:`Q^* = q_1^*, q_2^*, \ldots, q_T^*`

**Solution**: Viterbi Algorithm

.. math::

   \delta_t(i) = \max_{q_1, \ldots, q_{t-1}} P(q_1, \ldots, q_{t-1}, q_t = s_i, o_1, \ldots, o_t | \lambda)

Recursion:

.. math::

   \delta_1(i) &= \pi_i b_i(o_1) \\
   \delta_{t+1}(j) &= \max_i [\delta_t(i) a_{ij}] b_j(o_{t+1})

Backtracking:

.. math::

   \psi_{t+1}(j) = \arg\max_i [\delta_t(i) a_{ij}]

3. Learning Problem
~~~~~~~~~~~~~~~~~~~

**Given**: Observation sequence :math:`O` (and possibly multiple sequences)

**Find**: Model parameters :math:`\lambda^* = (A^*, B^*, \pi^*)` that maximize :math:`P(O | \lambda)`

**Solution**: Baum-Welch Algorithm (EM for HMMs)

Forward-Backward Variables
~~~~~~~~~~~~~~~~~~~~~~~~~~

**Forward variable**: :math:`\alpha_t(i)` (as defined above)

**Backward variable**:

.. math::

   \beta_t(i) = P(o_{t+1}, o_{t+2}, \ldots, o_T | q_t = s_i, \lambda)

Recursion:

.. math::

   \beta_T(i) &= 1 \\
   \beta_t(i) &= \sum_{j=1}^{N} a_{ij} b_j(o_{t+1}) \beta_{t+1}(j)

Baum-Welch Re-estimation
~~~~~~~~~~~~~~~~~~~~~~~~

**E-step**: Compute posterior probabilities

.. math::

   \gamma_t(i) &= P(q_t = s_i | O, \lambda) = \frac{\alpha_t(i) \beta_t(i)}{\sum_{j=1}^{N} \alpha_t(j) \beta_t(j)} \\
   \xi_t(i,j) &= P(q_t = s_i, q_{t+1} = s_j | O, \lambda) = \frac{\alpha_t(i) a_{ij} b_j(o_{t+1}) \beta_{t+1}(j)}{P(O | \lambda)}

**M-step**: Update parameters

.. math::

   \pi_i^{new} &= \gamma_1(i) \\
   a_{ij}^{new} &= \frac{\sum_{t=1}^{T-1} \xi_t(i,j)}{\sum_{t=1}^{T-1} \gamma_t(i)} \\
   b_j^{new}(k) &= \frac{\sum_{t=1, o_t=v_k}^{T} \gamma_t(j)}{\sum_{t=1}^{T} \gamma_t(j)}

Implementation Features
-----------------------

CMLA HMM Implementation
~~~~~~~~~~~~~~~~~~~~~~~

The CMLA HMM class provides:

* **Discrete observations**: Integer-valued observation sequences
* **All three algorithms**: Forward, Viterbi, and Baum-Welch
* **Multiple sequence training**: Can train on multiple observation sequences
* **Numerical stability**: Log-space computations to prevent underflow

Key Methods
~~~~~~~~~~~

.. list-table::
   :header-rows: 1

   * - Method
     - Purpose
     - Algorithm
   * - ``forward()``
     - Compute observation probability
     - Forward Algorithm
   * - ``viterbi()``
     - Find most likely state sequence
     - Viterbi Algorithm
   * - ``train_baum_welch()``
     - Learn model parameters
     - Baum-Welch Algorithm
   * - ``backward()``
     - Compute backward probabilities
     - Backward Algorithm

Usage Examples
--------------

Basic HMM Usage
~~~~~~~~~~~~~~~

.. code-block:: python

   import numpy as np
   from cmla.models.hmm import HMM

   # Create HMM with 2 states and 2 observation symbols
   hmm = HMM(num_states=2, num_observations=2)

   # Define model parameters manually
   hmm.transition_matrix = np.array([
       [0.7, 0.3],
       [0.4, 0.6]
   ])

   hmm.observation_matrix = np.array([
       [0.9, 0.1],  # State 0: likely to emit symbol 0
       [0.2, 0.8]   # State 1: likely to emit symbol 1
   ])

   hmm.initial_state_probability = np.array([0.6, 0.4])

   # Observation sequence
   observations = [0, 1, 0, 1, 1, 0]

Forward Algorithm
~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Compute probability of observation sequence
   alpha, prob = hmm.forward(observations)
   print(f"P(observations|model) = {prob}")
   print(f"Alpha matrix shape: {alpha.shape}")

Viterbi Algorithm
~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Find most likely state sequence
   path, prob = hmm.viterbi(observations)
   print(f"Most likely path: {path}")
   print(f"Path probability: {prob}")

Baum-Welch Training
~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Generate training data
   training_sequences = [
       [0, 1, 0, 1, 1],
       [1, 1, 0, 0, 1],
       [0, 0, 1, 1, 0],
       [1, 0, 1, 0, 1]
   ]

   # Initialize HMM with random parameters
   hmm_train = HMM(num_states=2, num_observations=2)

   # Train the model
   log_likelihoods = hmm_train.train_baum_welch(
       training_sequences,
       max_iterations=100,
       tolerance=1e-6
   )

   print("Final model parameters:")
   print(f"Transition matrix:\n{hmm_train.transition_matrix}")
   print(f"Observation matrix:\n{hmm_train.observation_matrix}")
   print(f"Initial probabilities: {hmm_train.initial_state_probability}")

Command-Line Interface
----------------------

The HMM CLI provides access to all three fundamental algorithms:

.. code-block:: bash

   # Train HMM model
   uv run python scripts/hmm_cli.py --train --data-file observations.txt

   # Run Viterbi algorithm
   uv run python scripts/hmm_cli.py --viterbi --observations "0 1 0 1 1"

   # Run forward algorithm
   uv run python scripts/hmm_cli.py --forward --observations "0 1 0 1"

CLI Options
~~~~~~~~~~~

.. list-table::
   :header-rows: 1

   * - Option
     - Description
   * - ``--train``
     - Train HMM using Baum-Welch algorithm
   * - ``--viterbi``
     - Run Viterbi decoding
   * - ``--forward``
     - Run forward algorithm
   * - ``--model-file, -m``
     - Load/save HMM model (JSON format)
   * - ``--observations``
     - Observation sequence (space-separated)
   * - ``--data-file, -f``
     - Input data file
   * - ``--states, -s``
     - Number of hidden states (default: 2)

Advanced Examples
-----------------

Multiple Sequence Training
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Multiple observation sequences for better training
   sequences = []
   for _ in range(10):
       seq_length = np.random.randint(5, 15)
       sequence = np.random.choice([0, 1], size=seq_length).tolist()
       sequences.append(sequence)

   hmm = HMM(num_states=3, num_observations=2)
   log_likelihoods = hmm.train_baum_welch(sequences, max_iterations=50)

   # Plot training progress
   import matplotlib.pyplot as plt
   plt.plot(log_likelihoods)
   plt.xlabel('Iteration')
   plt.ylabel('Log-likelihood')
   plt.title('HMM Training Progress')
   plt.show()

Model Comparison
~~~~~~~~~~~~~~~~

.. code-block:: python

   # Compare models with different numbers of states
   state_counts = [2, 3, 4, 5]
   final_likelihoods = []

   for n_states in state_counts:
       hmm = HMM(num_states=n_states, num_observations=2)
       log_likes = hmm.train_baum_welch(sequences, max_iterations=30)
       final_likelihoods.append(log_likes[-1])

   # Plot model comparison
   plt.bar(state_counts, final_likelihoods)
   plt.xlabel('Number of States')
   plt.ylabel('Final Log-likelihood')
   plt.title('HMM Model Comparison')
   plt.show()

State Analysis
~~~~~~~~~~~~~~

.. code-block:: python

   # Analyze state transitions
   transition_matrix = hmm.transition_matrix
   print("State transition analysis:")
   for i in range(hmm.num_states):
       most_likely_next = np.argmax(transition_matrix[i])
       prob = transition_matrix[i, most_likely_next]
       print(f"State {i} -> State {most_likely_next} (prob: {prob:.3f})")

   # Analyze observation patterns
   observation_matrix = hmm.observation_matrix
   print("\nObservation pattern analysis:")
   for i in range(hmm.num_states):
       most_likely_obs = np.argmax(observation_matrix[i])
       prob = observation_matrix[i, most_likely_obs]
       print(f"State {i} most likely emits {most_likely_obs} (prob: {prob:.3f})")

Applications
------------

Common Use Cases
~~~~~~~~~~~~~~~~

* **Speech Recognition**: Phoneme modeling
* **Bioinformatics**: Gene finding, protein structure prediction
* **Finance**: Regime detection in financial time series
* **Natural Language Processing**: Part-of-speech tagging
* **Weather Modeling**: Weather state prediction

Model Selection
~~~~~~~~~~~~~~~

For choosing the number of states:

* **Cross-validation**: Split data into train/validation sets
* **Information criteria**: AIC, BIC for model complexity
* **Domain knowledge**: Use problem-specific insights

API Reference
-------------

.. autoclass:: cmla.models.hmm.HMM
   :members:
   :undoc-members:
   :show-inheritance:

See Also
--------

* :doc:`../api/models` - Complete API reference
* :doc:`../tutorials/sequence_modeling` - Tutorial on sequence modeling
* :doc:`kmeans` - Clustering algorithms for comparison
