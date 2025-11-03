# how to run:
# PYTHONPATH=. uv run misc/demo_hmm.py

import numpy as np

from cmla.models.hmm import HMM, hmm_baum_welch, hmm_viterbi_training
from cmla.models.sampler import sampling_from_hmm

from cmla.models.utils import (
    randomize_state_transition_probabilities,
    randomize_observation_probabilities,
)


def test_viterbi_search():
    """
    test viterbi_search
    """
    # Assume there are 3 observation states, 2 hidden states, and 5 observations
    M = 2
    D = 4
    hmm = HMM(M, D)
    hmm.init_state = np.array([0.5, 0.5])
    hmm._log_init_state = np.log(hmm.init_state)
    hmm.state_tran = np.array([[0.1, 0.9], [0.5, 0.5]])
    hmm.obs_prob = np.array([[0.5, 0.5, 0.0, 0.0], [0.00, 0.00, 0.5, 0.5]])
    print(f"hmm={hmm}")

    # one sequence of length 10 is generated.
    st_orig, obs = sampling_from_hmm([10], hmm)
    print("observations({})={}".format(len(obs[0]), obs[0]))
    st, ll = hmm.viterbi_search(obs[0])
    print("st (", len(st), ")= ", st)
    for i, (_x, _s) in enumerate(zip(obs[0], st)):
        print(f"t={i} s={_s} s_true={st_orig[i]} x={_x}")


def test_viterbi_search2():
    """
    test viterbi_search
    """
    # Assume there are 3 observation states, 2 hidden states, and 5 observations
    M = 3
    D = 4
    hmm = HMM(M, D)
    hmm.init_state = np.array([0.5, 0.3, 0.2])
    hmm.state_tran = np.array([[0.7, 0.2, 0.1], [0.2, 0.0, 0.8], [0.3, 0.6, 0.1]])
    hmm.obs_prob = np.array(
        [[0.5, 0.3, 0.1, 0.1], [0.10, 0.10, 0.30, 0.50], [0.30, 0.30, 0.30, 0.10]]
    )
    print(f"hmm={hmm}")

    counts = 0
    total_counts = 0
    for samp_idx in range(1):
        st_orig, obs = sampling_from_hmm([200], hmm)
        st, ll = hmm.viterbi_search(obs[0])
        print("idx   org  est")
        for _i, (s0, s1, o) in enumerate(zip(st_orig, st, obs[0])):
            print(f"i={_i:03d} {s0}    {s1}  o={o}")
            if s0 == s1:
                counts += 1
            total_counts += 1
        print(counts, total_counts)


def test_viterbi_search3():
    M = 3
    D = 4
    hmm = HMM(M, D)
    # pylint: disable=duplicate-code
    hmm.init_state = np.array([0.50, 0.50, 0.00])
    hmm.state_tran = np.array(
        [[0.60, 0.35, 0.05], [0.01, 0.60, 0.39], [0.30, 0.00, 0.70]]
    )
    hmm.obs_prob = np.array(
        [[0.70, 0.10, 0.10, 0.10], [0.01, 0.09, 0.80, 0.10], [0.1, 0.45, 0.00, 0.45]]
    )
    # pylint: enable=duplicate-code
    obs = [0, 1, 1]
    st, ll = hmm.viterbi_search(obs)
    print("Optimal state sequence: ")
    print(st)


def test_viterbi_training():
    training_data = [
        [0, 1, 0, 0, 1, 2, 1, 1, 2, 2, 2, 3],
        [0, 0, 0, 1, 1, 2, 2, 2, 3, 3],
        [1, 0, 1, 1, 2, 4, 3, 0, 1],
        [2, 0, 1, 1, 1, 2, 3, 2, 1, 1],
    ]
    # print("N={}".format(len(training_data)))
    M = 2
    D = 5
    hmm = HMM(M, D)
    hmm.init_state = np.array([0.5, 0.5])
    hmm.state_tran = np.array([[0.5, 0.5], [0.5, 0.5]])
    hmm.obs_prob = np.array([[0.5, 0.2, 0.2, 0.1, 0.0], [0.00, 0.1, 0.4, 0.4, 0.1]])

    randomize_state_transition_probabilities(hmm)
    randomize_observation_probabilities(hmm)
    print(f"hmm={hmm}")
    hist = hmm_viterbi_training(hmm, training_data)
    print(f"hmm={hmm}")
    for i in range(len(hist["step"])):
        print(f"itr={hist['step'][i]} {hist['log_likelihood'][i]}")


def test_baum_welch():
    #   np.random.seed(3)
    training_data = [
        [0, 1, 0, 0, 1, 2, 1, 1, 2, 2, 2, 3],
        [0, 0, 0, 1, 1, 2, 2, 2, 3, 3],
        [1, 0, 1, 1, 2, 4, 3, 0, 1],
        [2, 0, 1, 1, 1, 2, 3, 2, 1, 1],
    ]
    # print("N={}".format(len(training_data)))
    M = 2
    D = 5
    hmm = HMM(M, D)
    hmm.init_state = np.array([0.5, 0.5])
    hmm.state_tran = np.array([[0.9, 0.1], [0.5, 0.5]])
    hmm.obs_prob = np.array([[0.5, 0.2, 0.2, 0.1, 0.0], [0.0, 0.1, 0.4, 0.4, 0.1]])

    randomize_state_transition_probabilities(hmm)
    randomize_observation_probabilities(hmm)
    print(f"hmm={hmm}")
    hmm_baum_welch(hmm, training_data)
    print(f"hmm={hmm}")
