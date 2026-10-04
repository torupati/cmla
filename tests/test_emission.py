import numpy as np
import pytest

from cmla.models.emission import DiscreteEmission
from cmla.models.hmm import HMM


def _discrete_hmm():
    hmm = HMM(2, 3)
    hmm.init_state = np.array([0.6, 0.4])
    hmm.state_tran = np.array([[0.7, 0.3], [0.2, 0.8]])
    hmm.obs_prob = np.array([[0.5, 0.5, 0.0], [0.1, 0.3, 0.6]])
    return hmm


def test_discrete_log_prob():
    emission = DiscreteEmission(2, 3)
    emission.probs = np.array([[0.5, 0.5, 0.0], [0.1, 0.3, 0.6]])
    logb = emission.log_prob([0, 2, 1])
    assert logb.shape == (3, 2)
    with np.errstate(divide="ignore"):
        expected = np.log([[0.5, 0.1], [0.0, 0.6], [0.5, 0.3]])
    np.testing.assert_array_equal(logb, expected)  # includes -inf


def test_discrete_accumulate_and_update():
    emission = DiscreteEmission(2, 3)
    obss = [0, 2, 2, 1]
    gamma = np.array([[1.0, 0.0], [0.25, 0.75], [0.0, 1.0], [0.5, 0.5]])
    emission.accumulate(obss, gamma)
    emission.update()
    count = np.array([[1.0, 0.5, 0.25], [0.0, 0.5, 1.75]])
    np.testing.assert_allclose(emission.probs, count / count.sum(axis=1, keepdims=True))
    np.testing.assert_array_equal(emission._count, 0.0)


def test_discrete_update_keeps_unvisited_state():
    emission = DiscreteEmission(2, 3)
    before = emission.probs[1].copy()
    emission.accumulate([0, 1], np.array([[1.0, 0.0], [1.0, 0.0]]))
    emission.update()
    np.testing.assert_array_equal(emission.probs[1], before)


def test_discrete_dict_roundtrip_does_not_consume_rng():
    emission = DiscreteEmission(2, 3)
    np.random.seed(0)
    restored = DiscreteEmission.from_dict(emission.to_dict())
    after = np.random.uniform()
    np.random.seed(0)
    assert np.random.uniform() == after
    np.testing.assert_array_equal(restored.probs, emission.probs)


def test_obs_prob_is_emission_probs():
    hmm = _discrete_hmm()
    assert hmm.obs_prob is hmm.emission.probs
    assert isinstance(hmm.emission, DiscreteEmission)


def test_hmm_with_explicit_emission():
    emission = DiscreteEmission(2, 3)
    hmm = HMM(2, emission=emission)
    assert hmm.emission is emission
    with pytest.raises(ValueError):
        HMM(3, emission=emission)
    with pytest.raises(ValueError):
        HMM(2)  # neither feature_dim nor emission


def test_viterbi_does_not_modify_obs_prob():
    hmm = _discrete_hmm()
    before = hmm.obs_prob.copy()
    hmm.viterbi_search([0, 1, 2, 2])
    np.testing.assert_array_equal(hmm.obs_prob, before)


class _ShiftedEmission(DiscreteEmission):
    """log b_j(x) shifted by a constant: far below float64 range in linear scale."""

    shift = -1.0e4

    def log_prob(self, obss):
        return super().log_prob(obss) + self.shift


def test_forward_backward_survives_underflow():
    hmm = _discrete_hmm()
    obss = [0, 1, 2, 2, 0]
    g1, g2, log_prob = hmm.forward_backward_algorithm_linear(obss)

    shifted = _ShiftedEmission.from_dict(hmm.emission.to_dict())
    hmm_shifted = HMM(2, emission=shifted)
    hmm_shifted.init_state = hmm.init_state
    hmm_shifted.state_tran = hmm.state_tran
    s1, s2, log_prob_shifted = hmm_shifted.forward_backward_algorithm_linear(obss)

    np.testing.assert_allclose(s1, g1)
    np.testing.assert_allclose(s2, g2)
    assert np.isclose(log_prob_shifted, log_prob + len(obss) * _ShiftedEmission.shift)


def test_discrete_rejects_out_of_range_symbols():
    emission = DiscreteEmission(2, 3)
    with pytest.raises(ValueError):
        emission.log_prob([0, 3])
    with pytest.raises(ValueError):
        emission.log_prob([-1])
