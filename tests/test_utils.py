import numpy as np
import pytest
from cmla.models.hmm import HMM
from cmla.models.utils import (
    randomize_state_transition_probabilities,
    randomize_observation_probabilities,
    randomize_all_probabilities,
)


@pytest.mark.parametrize("n_states,n_observations", [(3, 4), (10, 10)])
def test_randomize_state_transition_probabilities(n_states, n_observations):
    hmm = HMM(n_states, n_observations)
    randomize_state_transition_probabilities(hmm)
    assert np.allclose(hmm.state_tran.sum(axis=1), 1.0)
    assert np.all(hmm.state_tran >= 0.0)
    assert np.all(hmm.state_tran <= 1.0)
    assert np.allclose(hmm.init_state.sum(), 1.0)
    assert np.all(hmm.init_state >= 0.0)
    assert np.all(hmm.init_state <= 1.0)


@pytest.mark.parametrize("n_states,n_observations", [(3, 4), (10, 10)])
def test_randomize_observation_probabilities(n_states, n_observations):
    hmm = HMM(n_states, n_observations)
    randomize_observation_probabilities(hmm)
    assert np.allclose(hmm.obs_prob.sum(axis=1), 1.0)
    assert np.all(hmm.obs_prob >= 0.0)
    assert np.all(hmm.obs_prob <= 1.0)


@pytest.mark.parametrize("n_states,n_observations", [(3, 4), (10, 10)])
def test_randomize_all_probabilities(n_states, n_observations):
    hmm = HMM(n_states, n_observations)
    randomize_all_probabilities(hmm)
    assert np.allclose(hmm.state_tran.sum(axis=1), 1.0)
    assert np.all(hmm.state_tran >= 0.0)
    assert np.all(hmm.state_tran <= 1.0)
    assert np.allclose(hmm.init_state.sum(), 1.0)
    assert np.all(hmm.init_state >= 0.0)
    assert np.all(hmm.init_state <= 1.0)
    assert np.allclose(hmm.obs_prob.sum(axis=1), 1.0)
    assert np.all(hmm.obs_prob >= 0.0)
    assert np.all(hmm.obs_prob <= 1.0)
