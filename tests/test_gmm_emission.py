import json

import numpy as np
import pytest
from scipy.special import logsumexp
from scipy.stats import multivariate_normal

from cmla.models.emission import GMMEmission, emission_from_dict
from cmla.models.gaussian import diag_gaussian_log_pdf
from cmla.models.hmm import HMM, hmm_baum_welch, hmm_viterbi_training
from cmla.models.sampler import sampling_from_hmm


def _true_gmm_hmm():
    """2 states x 2 mixtures in 2-D. States are separated along x, mixtures along y."""
    emission = GMMEmission(2, 2, 2)
    emission.weights = np.array([[0.3, 0.7], [0.5, 0.5]])
    emission.means = np.array([[[-4.0, -2.0], [-4.0, 2.0]], [[4.0, -2.0], [4.0, 2.0]]])
    emission.covs = np.array([[[0.5, 0.3], [0.4, 0.6]], [[0.6, 0.5], [0.3, 0.4]]])
    hmm = HMM(2, emission=emission)
    hmm.init_state = np.array([0.5, 0.5])
    hmm.state_tran = np.array([[0.9, 0.1], [0.2, 0.8]])
    return hmm


def _sample(hmm, num_seqs, length):
    seqs, states = [], []
    for _ in range(num_seqs):
        st, x = sampling_from_hmm([length], hmm)
        seqs.append(np.array(x[0]))
        states.append(np.array(st[0]))
    return seqs, states


def test_diag_gaussian_log_pdf_matches_scipy():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(5, 3))
    means = rng.normal(size=(2, 4, 3))
    variances = rng.uniform(0.1, 2.0, size=(2, 4, 3))
    out = diag_gaussian_log_pdf(x, means, variances)
    assert out.shape == (5, 2, 4)
    for m in range(2):
        for k in range(4):
            expected = multivariate_normal(
                means[m, k], np.diag(variances[m, k])
            ).logpdf(x)
            np.testing.assert_allclose(out[:, m, k], expected)


def test_gmm_log_prob_matches_scipy():
    hmm = _true_gmm_hmm()
    emission = hmm.emission
    x = np.random.default_rng(1).normal(scale=3.0, size=(6, 2))
    logb = emission.log_prob(x)
    assert logb.shape == (6, 2)
    for j in range(2):
        comp = [
            np.log(emission.weights[j, k])
            + multivariate_normal(
                emission.means[j, k], np.diag(emission.covs[j, k])
            ).logpdf(x)
            for k in range(2)
        ]
        np.testing.assert_allclose(logb[:, j], logsumexp(comp, axis=0))


def test_single_mixture_update_is_weighted_mean_and_variance():
    rng = np.random.default_rng(2)
    x = rng.normal(loc=[1.0, -2.0], scale=[0.5, 2.0], size=(50, 2))
    gamma = rng.uniform(size=(50, 2))
    gamma = gamma / gamma.sum(axis=1, keepdims=True)
    emission = GMMEmission(2, 1, 2, var_floor=1.0e-6)
    emission.accumulate(x, gamma)
    emission.update()
    for j in range(2):
        w = gamma[:, j]
        mean = (w[:, None] * x).sum(axis=0) / w.sum()
        var = (w[:, None] * (x - mean) ** 2).sum(axis=0) / w.sum()
        np.testing.assert_allclose(emission.means[j, 0], mean)
        np.testing.assert_allclose(emission.covs[j, 0], var)
        np.testing.assert_allclose(emission.weights[j], [1.0])


def test_variance_floor():
    emission = GMMEmission(1, 1, 2, var_floor=np.array([0.1, 0.2]))
    x = np.array([[1.0, 2.0]] * 10)  # zero variance
    emission.accumulate(x, np.ones((10, 1)))
    emission.update()
    np.testing.assert_allclose(emission.covs[0, 0], [0.1, 0.2])


def test_dead_component_keeps_parameters():
    emission = GMMEmission(1, 2, 1, min_occupancy=1.0)
    emission.means = np.array([[[0.0], [1.0e3]]])
    emission.covs = np.ones((1, 2, 1))
    x = np.random.default_rng(3).normal(size=(20, 1))
    emission.accumulate(x, np.ones((20, 1)))
    emission.update()
    np.testing.assert_array_equal(emission.means[0, 1], [1.0e3])
    np.testing.assert_array_equal(emission.covs[0, 1], [1.0])
    assert emission.weights[0, 1] < 1.0e-6


def test_unvisited_state_keeps_parameters():
    emission = GMMEmission(2, 1, 1)
    before = emission.means[1].copy()
    emission.accumulate(np.zeros((3, 1)), np.array([[1.0, 0.0]] * 3))
    emission.update()
    np.testing.assert_array_equal(emission.means[1], before)


def test_only_diag_supported():
    with pytest.raises(NotImplementedError):
        GMMEmission(2, 2, 2, cov_type="full")


def test_hmm_constructor_gmm():
    hmm = HMM(3, 2, observation_type="gmm", num_mixtures=4)
    assert isinstance(hmm.emission, GMMEmission)
    assert hmm.emission.means.shape == (3, 4, 2)
    with pytest.raises(AttributeError):
        hmm.obs_prob


def test_gmm_dict_roundtrip():
    hmm = _true_gmm_hmm()
    restored = HMM.from_dict(json.loads(json.dumps(hmm.to_dict())))
    assert isinstance(restored.emission, GMMEmission)
    for name in ("weights", "means", "covs"):
        np.testing.assert_array_equal(
            getattr(restored.emission, name), getattr(hmm.emission, name)
        )
    restored_emission = emission_from_dict(hmm.emission.to_dict())
    assert restored_emission.var_floor == hmm.emission.var_floor


def test_gmm_save_load(tmp_path):
    from cmla.models.hmm import load_hmm_and_data

    np.random.seed(0)
    hmm = _true_gmm_hmm()
    st, x = sampling_from_hmm([4, 6], hmm)
    out_file = tmp_path / "gmm_hmm.json"
    hmm.save_hmm_and_data(str(out_file), x, st)
    hmm2, x2, _ = load_hmm_and_data(str(out_file))
    np.testing.assert_array_equal(hmm2.emission.means, hmm.emission.means)
    assert [np.asarray(s).shape for s in x2] == [(4, 2), (6, 2)]


def test_sample_shape():
    emission = _true_gmm_hmm().emission
    x = emission.sample(1, rng=np.random.default_rng(4))
    assert x.shape == (2,)
    assert x[0] > 0  # state 1 is on the positive side


def test_forward_backward_with_outlier_frame():
    hmm = _true_gmm_hmm()
    x = np.array([[-4.0, 2.0], [1.0e4, -1.0e4], [4.0, 2.0]])
    g1, g2, log_prob = hmm.forward_backward_algorithm_linear(x)
    assert np.all(np.isfinite(g1)) and np.all(np.isfinite(g2))
    np.testing.assert_allclose(g1.sum(axis=1), 1.0)
    assert np.isfinite(log_prob)


def test_viterbi_decodes_separated_states():
    np.random.seed(5)
    hmm = _true_gmm_hmm()
    seqs, states = _sample(hmm, 5, 40)
    for x, st in zip(seqs, states):
        path, log_prob = hmm.viterbi_search(x)
        assert np.mean(np.array(path) == st) > 0.95
        assert np.isfinite(log_prob)


def _perturbed_model(true_hmm):
    np.random.seed(7)
    hmm = HMM(2, 2, observation_type="gmm", num_mixtures=2)
    hmm.emission.means = true_hmm.emission.means + np.random.normal(
        scale=0.7, size=(2, 2, 2)
    )
    hmm.emission.covs = np.full((2, 2, 2), 2.0)
    return hmm


def test_baum_welch_recovers_parameters(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)  # hmm_baum_welch writes checkpoints to the cwd
    np.random.seed(6)
    true_hmm = _true_gmm_hmm()
    seqs, _ = _sample(true_hmm, 40, 50)
    hmm = _perturbed_model(true_hmm)
    history = hmm_baum_welch(hmm, seqs, itr_limit=30)

    ll = np.array(history["log_likelihood"])
    assert np.all(np.diff(ll) >= -1.0e-9 * np.abs(ll[:-1]))
    np.testing.assert_allclose(hmm.emission.means, true_hmm.emission.means, atol=0.2)
    np.testing.assert_allclose(
        hmm.emission.weights, true_hmm.emission.weights, atol=0.08
    )
    np.testing.assert_allclose(hmm.emission.covs, true_hmm.emission.covs, rtol=0.3)
    np.testing.assert_allclose(hmm.state_tran, true_hmm.state_tran, atol=0.05)


def test_viterbi_training_gmm():
    np.random.seed(8)
    true_hmm = _true_gmm_hmm()
    seqs, _ = _sample(true_hmm, 20, 50)
    hmm = _perturbed_model(true_hmm)
    history = hmm_viterbi_training(hmm, seqs, itr_limit=5)
    assert np.all(np.isfinite(history["log_likelihood"]))
    np.testing.assert_allclose(hmm.emission.means, true_hmm.emission.means, atol=0.3)
