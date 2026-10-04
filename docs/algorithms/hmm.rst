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

Continuous Observations: GMM-HMM
--------------------------------

For real-valued observation vectors :math:`\mathbf{x}_t \in \mathbb{R}^D`, each state
emits from a Gaussian mixture with diagonal covariance:

.. math::

   b_j(\mathbf{x}) = \sum_{k=1}^{K} c_{jk}\,
   \mathcal{N}\!\left(\mathbf{x};\boldsymbol\mu_{jk},\operatorname{diag}(\boldsymbol\sigma^2_{jk})\right)

Forward, backward and Viterbi are unchanged: they only need :math:`b_j(\mathbf{x}_t)`.
The M-step needs the posterior of each mixture component,

.. math::

   r_t(j,k) = \gamma_t(j)\,
   \frac{c_{jk}\,\mathcal{N}(\mathbf{x}_t;\boldsymbol\mu_{jk},\boldsymbol\sigma^2_{jk})}{b_j(\mathbf{x}_t)}

and then updates

.. math::

   c_{jk} = \frac{\sum_t r_t(j,k)}{\sum_t \gamma_t(j)}, \qquad
   \boldsymbol\mu_{jk} = \frac{\sum_t r_t(j,k)\,\mathbf{x}_t}{\sum_t r_t(j,k)}, \qquad
   \boldsymbol\sigma^2_{jk} = \max\!\left(\frac{\sum_t r_t(j,k)\,\mathbf{x}_t^2}{\sum_t r_t(j,k)}
   - \boldsymbol\mu_{jk}^2,\ \sigma^2_{\text{floor}}\right)

Gaussian densities can be far below the smallest double for every state at once
(e.g. an outlier frame). The forward-backward algorithm therefore scales
:math:`b_j(\mathbf{x}_t)` by :math:`1/\max_j b_j(\mathbf{x}_t)` at each time step; the scale
cancels in :math:`\gamma` and :math:`\xi` and is added back to :math:`\log P(O|\lambda)`.

EM finds only a local optimum, so initialization matters. ``init_gmm_hmm`` assigns
frames to states (uniform segmentation for left-to-right models, k-means for ergodic
models) and runs k-means within each state to place the mixture means.

Implementation
--------------

Design
~~~~~~

``HMM`` keeps the initial state probabilities ``init_state`` (M,) and transition
probabilities ``state_tran`` (M, M). Everything that depends on the observation type is
in an ``Emission`` object, ``hmm.emission``:

.. list-table::
   :header-rows: 1

   * - Class
     - Observation
     - Parameters
   * - ``DiscreteEmission``
     - integer symbols, sequence shape (T,)
     - ``probs`` (M, K); also accessible as ``hmm.obs_prob``
   * - ``GMMEmission``
     - real vectors, sequence shape (T, D)
     - ``weights`` (M, K), ``means`` (M, K, D), ``covs`` (M, K, D) diagonal variances

See :doc:`../design/gmm_hmm` for the design document.

Key Functions and Methods
~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1

   * - Function / method
     - Purpose
   * - ``hmm.log_likelihood(x)``
     - :math:`\log P(O|\lambda)` by the forward algorithm
   * - ``hmm.viterbi_search(x)``
     - Most likely state sequence and its log probability
   * - ``hmm.forward_backward_algorithm_linear(x)``
     - State posteriors :math:`\gamma`, :math:`\xi` and :math:`\log P(O|\lambda)`
   * - ``hmm_baum_welch(hmm, seqs, itr_limit)``
     - Baum-Welch (EM) training
   * - ``hmm_viterbi_training(hmm, seqs, itr_limit)``
     - Viterbi (hard EM) training
   * - ``init_gmm_hmm(hmm, seqs, method)``
     - Initialize a GMM-HMM from data
   * - ``hmm.save(file)`` / ``HMM.load(file)``
     - Save / load model parameters (JSON or pickle)
   * - ``sampling_from_hmm(lengths, hmm)``
     - Generate state and observation sequences

Usage Examples
--------------

Discrete HMM
~~~~~~~~~~~~

.. code-block:: python

   import numpy as np
   from cmla.models.hmm import HMM, hmm_baum_welch

   # 2 states, 2 observation symbols
   hmm = HMM(2, 2)
   hmm.init_state = np.array([0.6, 0.4])
   hmm.state_tran = np.array([[0.7, 0.3], [0.4, 0.6]])
   hmm.obs_prob = np.array([[0.9, 0.1],   # state 0: likely to emit symbol 0
                            [0.2, 0.8]])  # state 1: likely to emit symbol 1

   observations = [0, 1, 0, 1, 1, 0]
   print(hmm.log_likelihood(observations))          # log P(O | model)
   path, log_prob = hmm.viterbi_search(observations)

   # Baum-Welch training on several sequences
   sequences = [[0, 1, 0, 1, 1], [1, 1, 0, 0, 1], [0, 0, 1, 1, 0]]
   model = HMM(2, 2)                                 # random emission probabilities
   history = hmm_baum_welch(model, sequences, itr_limit=50, checkpoint_dir=None)
   print(history["log_likelihood"][-1])

GMM-HMM
~~~~~~~

.. code-block:: python

   from cmla.models.hmm import HMM, hmm_baum_welch
   from cmla.models.hmm_init import init_gmm_hmm
   from cmla.models.sampler import generate_gmm_hmm_parameter, sampling_from_hmm

   # sample from a random 3-state, 2-mixture, 2-D model
   true_hmm = generate_gmm_hmm_parameter(num_states=3, num_mixtures=2, feature_dim=2)
   states, sequences = sampling_from_hmm([50] * 30, true_hmm)

   hmm = HMM(3, 2, observation_type="gmm", num_mixtures=2)
   init_gmm_hmm(hmm, sequences, method="kmeans")    # "uniform_segment" for left-to-right
   hmm_baum_welch(hmm, sequences, itr_limit=30, checkpoint_dir=None)

   print(hmm.emission.means)                         # (M, K, D)
   hmm.save("gmm_hmm.json")

Plotting
~~~~~~~~

.. code-block:: python

   import numpy as np
   from cmla.plots.hmm_plot import plot_emission

   # GMM: ellipses of each Gaussian over the samples, colored by true state
   fig = plot_emission(hmm, x=np.concatenate(sequences), states=np.concatenate(states))
   fig.savefig("gmm_hmm.png")

Utility Functions
~~~~~~~~~~~~~~~~~

``cmla.models.utils`` randomizes the parameters of a discrete HMM in place:

* ``randomize_state_transition_probabilities(hmm)`` - initial state and transition probabilities
* ``randomize_observation_probabilities(hmm)`` - observation probability matrix
* ``randomize_all_probabilities(hmm)`` - all of the above

Command-Line Interface
----------------------

.. code-block:: bash

   # generate GMM-HMM samples: model + data (JSON) and sequences (CSV)
   uv run python -m cmla.scripts.sampler_cli 30 gmm_hmm.json --csv HMM-GMM \
       --states 3 --mixtures 2 --dimension 2

   # train a GMM-HMM (Baum-Welch; --algorithm viterbi for Viterbi training)
   uv run python -m cmla.scripts.hmm_cli train --type gmm --states 3 --mixtures 2 \
       --data-file gmm_hmm.csv --iterations 30 --output model.json

   # log-likelihood and Viterbi path of each sequence
   uv run python -m cmla.scripts.hmm_cli forward --model model.json --data-file gmm_hmm.csv
   uv run python -m cmla.scripts.hmm_cli viterbi --model model.json --data-file gmm_hmm.csv

   # discrete HMM
   uv run python -m cmla.scripts.hmm_cli train --data-file discrete.json --states 2 -o d.json
   uv run python -m cmla.scripts.hmm_cli viterbi --model d.json --observations "0 1 0 1 1"

Data files are either the JSON/pickle output of ``sampler_cli`` or a text file with one
comma-separated frame per line and a blank line between sequences.

.. list-table:: ``train`` options
   :header-rows: 1

   * - Option
     - Description
   * - ``--data-file, -f`` / ``--observations``
     - Training data / one discrete sequence (space-separated)
   * - ``--type``
     - ``discrete`` (default) or ``gmm``
   * - ``--states, -s``
     - Number of hidden states (default: 2)
   * - ``--symbols``
     - Number of symbols of a discrete HMM (default: max symbol + 1)
   * - ``--mixtures``
     - Gaussians per state of a GMM-HMM (default: 2)
   * - ``--init``
     - GMM-HMM initialization, ``kmeans`` (default) or ``uniform_segment``
   * - ``--algorithm``
     - ``baum-welch`` (default) or ``viterbi``
   * - ``--iterations, -n``
     - Training iterations (default: 20)
   * - ``--verbose, -v`` / ``--quiet, -q``
     - Show all log messages / hide the log-likelihood of each iteration
   * - ``--log-file``
     - Append all log messages with timestamps to this file
       (default: ``hmm_cli.log``; ``""`` disables). Also accepted by ``viterbi`` and ``forward``.
   * - ``--model, -m``
     - Start from this model instead of a new one
   * - ``--checkpoint-dir``
     - Save Baum-Welch checkpoints (default: none)
   * - ``--output, -o``
     - Output model file, ``.json`` or ``.pkl``

Model Comparison
~~~~~~~~~~~~~~~~

.. code-block:: python

   # Compare models with different numbers of states on held-out data
   for n_states in [2, 3, 4, 5]:
       hmm = HMM(n_states, 2, observation_type="gmm", num_mixtures=2)
       init_gmm_hmm(hmm, train_seqs, method="kmeans")
       hmm_baum_welch(hmm, train_seqs, itr_limit=30, checkpoint_dir=None)
       print(n_states, sum(hmm.log_likelihood(x) for x in valid_seqs))

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
   :no-index:
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: cmla.models.emission.GMMEmission
   :no-index:
   :members:
   :show-inheritance:

See Also
--------

* :doc:`../api/models` - Complete API reference
* :doc:`../tutorials/sequence_modeling` - Tutorial on sequence modeling
* :doc:`kmeans` - Clustering algorithms for comparison
