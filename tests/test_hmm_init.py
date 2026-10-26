import numpy as np
import pytest

from cmla.models.emission import GMMEmission
from cmla.models.hmm import HMM, hmm_baum_welch
from cmla.models.hmm_init import init_gmm_hmm, uniform_segmentation
from cmla.models.sampler import sampling_from_hmm

# 2 states x 2 mixtures in 2-D. States are separated along x, mixtures along y.
TRUE_WEIGHTS = np.array([[0.3, 0.7], [0.5, 0.5]])
TRUE_MEANS = np.array([[[-6.0, -2.0], [-6.0, 2.0]], [[6.0, -2.0], [6.0, 2.0]]])
TRUE_COVS = np.array([[[0.5, 0.3], [0.4, 0.6]], [[0.6, 0.5], [0.3, 0.4]]])


def _true_emission():
    emission = GMMEmission(2, 2, 2)
    emission.weights = TRUE_WEIGHTS.copy()
    emission.means = TRUE_MEANS.copy()
    emission.covs = TRUE_COVS.copy()
    return emission


def _sort_mixtures(means):
    """Order mixtures of each state by y so that runs can be compared."""
    order = np.argsort(means[:, :, 1], axis=1)
    return np.take_along_axis(means, order[:, :, np.newaxis], axis=1), order


def test_uniform_segmentation():
    np.testing.assert_array_equal(
        uniform_segmentation(10, 3), [0, 0, 0, 0, 1, 1, 1, 2, 2, 2]
    )
    np.testing.assert_array_equal(uniform_segmentation(4, 4), [0, 1, 2, 3])
    seg = uniform_segmentation(7, 2)
    assert seg[0] == 0 and seg[-1] == 1 and np.all(np.diff(seg) >= 0)


def test_uniform_segment_left_to_right():
    """First half of every sequence is state 0, second half state 1."""
    np.random.seed(0)
    rng = np.random.default_rng(0)
    emission = _true_emission()
    seqs = []
    for T in (30, 40, 50) * 10:
        st = uniform_segmentation(T, 2)
        seqs.append(np.array([emission.sample(s, rng) for s in st]))

    hmm = HMM(2, 2, observation_type="gmm", num_mixtures=2)
    init_gmm_hmm(hmm, seqs, method="uniform_segment")

    means, order = _sort_mixtures(hmm.emission.means)
    np.testing.assert_allclose(means, TRUE_MEANS, atol=0.2)
    weights = np.take_along_axis(hmm.emission.weights, order, axis=1)
    np.testing.assert_allclose(weights, TRUE_WEIGHTS, atol=0.08)
    np.testing.assert_allclose(hmm.emission.weights.sum(axis=1), 1.0)
    assert np.all(hmm.emission.covs >= hmm.emission.var_floor)


def test_kmeans_then_baum_welch_ergodic(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)  # hmm_baum_welch writes checkpoints to the cwd
    np.random.seed(1)
    true_hmm = HMM(2, emission=_true_emission())
    true_hmm.init_state = np.array([0.5, 0.5])
    true_hmm.state_tran = np.array([[0.9, 0.1], [0.2, 0.8]])
    seqs = []
    for _ in range(20):
        _, x = sampling_from_hmm([40], true_hmm)
        seqs.append(np.array(x[0]))

    hmm = HMM(2, 2, observation_type="gmm", num_mixtures=2)
    init_gmm_hmm(hmm, seqs, method="kmeans")
    hmm_baum_welch(hmm, seqs, itr_limit=10)

    # state order is arbitrary: sort states by x of their first mixture
    state_order = np.argsort(hmm.emission.means[:, 0, 0])
    means, _ = _sort_mixtures(hmm.emission.means[state_order])
    np.testing.assert_allclose(means, TRUE_MEANS, atol=0.3)
    np.testing.assert_allclose(
        hmm.state_tran[np.ix_(state_order, state_order)],
        true_hmm.state_tran,
        atol=0.1,
    )


def test_single_mixture():
    np.random.seed(2)
    seqs = [np.random.normal(size=(10, 3)) for _ in range(4)]
    hmm = HMM(2, 3, observation_type="gmm", num_mixtures=1)
    init_gmm_hmm(hmm, seqs)
    frames = np.concatenate([x[:5] for x in seqs])  # state 0 = first half
    np.testing.assert_allclose(hmm.emission.means[0, 0], frames.mean(axis=0))
    np.testing.assert_allclose(hmm.emission.covs[0, 0], frames.var(axis=0))
    np.testing.assert_array_equal(hmm.emission.weights, 1.0)


def test_does_not_modify_input():
    np.random.seed(3)
    seqs = [np.random.normal(size=(20, 2)) for _ in range(3)]
    before = [x.copy() for x in seqs]
    init_gmm_hmm(
        HMM(2, 2, observation_type="gmm", num_mixtures=3), seqs, method="kmeans"
    )
    for x, y in zip(seqs, before):
        np.testing.assert_array_equal(x, y)


def test_errors():
    seqs = [np.zeros((4, 2))]
    with pytest.raises(TypeError):
        init_gmm_hmm(HMM(2, 3), seqs)  # discrete HMM
    with pytest.raises(ValueError):
        init_gmm_hmm(HMM(2, 2, observation_type="gmm", num_mixtures=3), seqs)
    with pytest.raises(ValueError):
        init_gmm_hmm(
            HMM(2, 2, observation_type="gmm", num_mixtures=1), seqs, method="x"
        )
