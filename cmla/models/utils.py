"""
Utility functions for CMLA models.

This module provides standalone utility functions that can be used with model instances
to perform common operations like parameter initialization and randomization.
"""

import numpy as np
from logging import getLogger

logger = getLogger(__name__)


def randomize_state_transition_probabilities(hmm):
    """Randomize state transition probabilities for an HMM instance.

    Args:
        hmm: HMM instance to modify

    Returns:
        None (modifies hmm in-place)
    """
    _vals = np.random.uniform(size=hmm.num_hidden_states)
    hmm.init_state = _vals / sum(_vals)  # _vals > 0 is guaranteed.

    hmm.state_tran = np.random.uniform(
        size=(hmm.num_hidden_states, hmm.num_hidden_states)
    )
    for m in range(hmm.num_hidden_states):
        hmm.state_tran[m, :] = hmm.state_tran[m, :] / sum(hmm.state_tran[m, :])

    # Assertions to verify correctness
    assert np.allclose(hmm.state_tran.sum(axis=1), 1.0)
    assert np.all(hmm.state_tran >= 0.0)
    assert np.all(hmm.state_tran <= 1.0)
    assert np.allclose(hmm.init_state.sum(), 1.0)
    assert np.all(hmm.init_state >= 0.0)
    assert np.all(hmm.init_state <= 1.0)

    logger.debug(f"randomized state_tran=\n{hmm.state_tran}")
    logger.debug(f"randomized init_state=\n{hmm.init_state}")


def randomize_observation_probabilities(hmm):
    """Randomize observation probabilities for an HMM instance.

    Args:
        hmm: HMM instance to modify

    Returns:
        None (modifies hmm in-place)
    """
    hmm.obs_prob = np.random.uniform(
        size=(hmm.num_hidden_states, hmm.obs_prob.shape[1])
    )
    for m in range(hmm.num_hidden_states):
        hmm.obs_prob[m, :] = hmm.obs_prob[m, :] / sum(hmm.obs_prob[m, :])

    # Assertions to verify correctness
    assert np.allclose(hmm.obs_prob.sum(axis=1), 1.0)
    assert np.all(hmm.obs_prob >= 0.0)
    assert np.all(hmm.obs_prob <= 1.0)

    logger.debug(f"randomized obs_prob=\n{hmm.obs_prob}")


def randomize_all_probabilities(hmm):
    """Randomize both state transition and observation probabilities.

    Args:
        hmm: HMM instance to modify

    Returns:
        None (modifies hmm in-place)
    """
    randomize_state_transition_probabilities(hmm)
    randomize_observation_probabilities(hmm)
