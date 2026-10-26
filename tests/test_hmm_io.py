import json
import pickle

import numpy as np
import pytest

from cmla.models.emission import DiscreteEmission
from cmla.models.hmm import HMM, load_hmm_and_data


def _discrete_hmm():
    hmm = HMM(2, 3)
    hmm.init_state = np.array([0.6, 0.4])
    hmm.state_tran = np.array([[0.7, 0.3], [0.2, 0.8]])
    hmm.obs_prob = np.array([[0.5, 0.5, 0.0], [0.1, 0.3, 0.6]])
    return hmm


def _assert_same_hmm(a: HMM, b: HMM):
    np.testing.assert_array_equal(a.init_state, b.init_state)
    np.testing.assert_array_equal(a.state_tran, b.state_tran)
    assert type(a.emission) is type(b.emission)
    np.testing.assert_array_equal(a.obs_prob, b.obs_prob)


def test_to_dict_is_json_serializable():
    d = _discrete_hmm().to_dict()
    assert d["version"] == 2
    assert d["emission"]["type"] == "discrete"
    json.dumps(d)


def test_dict_roundtrip():
    hmm = _discrete_hmm()
    _assert_same_hmm(HMM.from_dict(hmm.to_dict()), hmm)


@pytest.mark.parametrize("ext", [".json", ".pkl", ".pickle"])
def test_save_load_roundtrip(tmp_path, ext):
    hmm = _discrete_hmm()
    x = [[0, 1, 2], [2, 2, 1, 0, 0]]  # sequences of different lengths
    st = np.array([0, 1, 1, 0, 0])
    out_file = tmp_path / f"hmm{ext}"
    hmm.save_hmm_and_data(str(out_file), x, st)

    hmm2, x2, st2 = load_hmm_and_data(str(out_file))
    _assert_same_hmm(hmm2, hmm)
    assert [list(seq) for seq in x2] == x
    np.testing.assert_array_equal(st2, st)


def test_save_json_with_numpy_sequences(tmp_path):
    hmm = _discrete_hmm()
    x = [np.array([0, 1, 2]), np.array([2, 1, 0])]
    out_file = tmp_path / "hmm.json"
    hmm.save_hmm_and_data(str(out_file), x, np.array([0, 1, 0]))
    _, x2, _ = load_hmm_and_data(str(out_file))
    np.testing.assert_array_equal(x2, np.array(x))


def test_load_version1_json(tmp_path):
    """File written by the previous save_hmm_and_data."""
    data = {
        "model_param": {
            "init_state": [0.6, 0.4],
            "state_tran": [[0.7, 0.3], [0.2, 0.8]],
            "obs_prob": [[0.5, 0.5, 0.0], [0.1, 0.3, 0.6]],
            "n_state": 2,
        },
        "sample": [[0, 1, 2], [2, 0]],
        "latent": [1, 0],
        "model_type": "HMM",
    }
    in_file = tmp_path / "v1.json"
    in_file.write_text(json.dumps(data))
    hmm, x, st = load_hmm_and_data(str(in_file))
    _assert_same_hmm(hmm, _discrete_hmm())
    assert [list(seq) for seq in x] == [[0, 1, 2], [2, 0]]


def test_load_version1_checkpoint_dict(tmp_path):
    """Checkpoint written by the previous hmm_baum_welch (numpy arrays, n_obs)."""
    ref = _discrete_hmm()
    ckpt = {
        "model": {
            "init_state": ref.init_state.copy(),
            "state_tran": ref.state_tran.copy(),
            "obs_prob": ref.obs_prob.copy(),
            "n_state": 2,
            "n_obs": 3,
        },
        "model_type": "HMM",
    }
    ckpt_file = tmp_path / "hmm.ckpt"
    ckpt_file.write_bytes(pickle.dumps(ckpt))
    loaded = pickle.loads(ckpt_file.read_bytes())
    _assert_same_hmm(HMM.from_dict(loaded["model"]), ref)


def test_from_dict_errors():
    d = _discrete_hmm().to_dict()
    with pytest.raises(ValueError):
        HMM.from_dict({**d, "emission": {"type": "unknown"}})
    with pytest.raises(ValueError):
        HMM.from_dict({**d, "state_tran": [[1.0]]})
    with pytest.raises(ValueError):
        HMM.from_dict({"init_state": [1.0], "state_tran": [[1.0]]})


def test_from_dict_does_not_consume_rng():
    d = _discrete_hmm().to_dict()
    np.random.seed(0)
    HMM.from_dict(d)
    after = np.random.uniform()
    np.random.seed(0)
    assert np.random.uniform() == after


def test_emission_object_roundtrip():
    emission = DiscreteEmission(3, 4)
    hmm = HMM(3, emission=emission)
    restored = HMM.from_dict(json.loads(json.dumps(hmm.to_dict())))
    np.testing.assert_allclose(restored.emission.probs, emission.probs)


@pytest.mark.parametrize("ext", [".json", ".pkl"])
def test_model_save_load(tmp_path, ext):
    hmm = _discrete_hmm()
    out_file = tmp_path / f"model{ext}"
    hmm.save(str(out_file))
    _assert_same_hmm(HMM.load(str(out_file)), hmm)


def test_load_accepts_data_file(tmp_path):
    hmm = _discrete_hmm()
    out_file = tmp_path / "data.json"
    hmm.save_hmm_and_data(str(out_file), [[0, 1]], [0, 1])
    _assert_same_hmm(HMM.load(str(out_file)), hmm)


def test_log_likelihood_matches_forward_backward():
    hmm = _discrete_hmm()
    obss = [0, 1, 2, 2, 1]
    _, _, log_prob = hmm.forward_backward_algorithm_linear(obss)
    assert np.isclose(hmm.log_likelihood(obss), log_prob)
    # brute force: sum over all state sequences
    import itertools

    total = 0.0
    for path in itertools.product(range(2), repeat=len(obss)):
        p = hmm.init_state[path[0]] * hmm.obs_prob[path[0], obss[0]]
        for t in range(1, len(obss)):
            p *= hmm.state_tran[path[t - 1], path[t]] * hmm.obs_prob[path[t], obss[t]]
        total += p
    assert np.isclose(hmm.log_likelihood(obss), np.log(total))
