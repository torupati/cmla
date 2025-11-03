import copy

import matplotlib.pyplot as plt
import numpy as np

from cmla.models.hmm import HMM, hmm_baum_welch
from cmla.models.sampler import sampling_from_hmm


def test_forward_backward():
    """ """
    # Assume there are 3 observation states, 2 hidden states, and 5 observations
    M = 3
    D = 4
    hmm = HMM(M, D)
    hmm.init_state = np.array([0.34, 0.33, 0.33])
    # hmm._log_init_state = np.log(hmm.init_state)
    hmm.state_tran = np.array(
        [[0.60, 0.35, 0.05], [0.01, 0.60, 0.39], [0.30, 0.00, 0.70]]
    )
    hmm.obs_prob = np.array(
        [[0.70, 0.10, 0.10, 0.10], [0.01, 0.09, 0.80, 0.10], [0.1, 0.45, 0.00, 0.45]]
    )
    hmm_orig = copy.deepcopy(hmm)
    print(f"defined hmm={hmm}")

    N = 200
    T = 30
    sample_lengths = [T] * N
    states_orig, obs_seqs = sampling_from_hmm(sample_lengths, hmm)

    hmm.state_tran = np.array([[1 / 3.0] * 3, [1 / 3.0] * 3, [1 / 3.0] * 3])
    hmm.state_tran = np.array(
        [[0.60, 0.30, 0.10], [0.10, 0.60, 0.30], [0.30, 0.10, 0.60]]
    )
    hmm.obs_prob = np.array(
        [[0.4, 0.2, 0.20, 0.20], [0.20, 0.20, 0.4, 0.20], [0.25, 0.25, 0.25, 0.25]]
    )
    hmm_init = copy.deepcopy(hmm)
    ll_history = hmm_baum_welch(hmm, obs_seqs, itr_limit=100)
    previous_loglikelihood = -np.inf
    for i, loglikelihood in enumerate(ll_history["log_likelihood"]):
        assert loglikelihood >= previous_loglikelihood, (
            f"Log-likelihood did not increase! {i} th iteration: {loglikelihood} < {previous_loglikelihood}"
        )
        previous_loglikelihood = loglikelihood
    print(hmm.state_tran)
    print("")
    import json

    with open("out_test_hmm_baumwelch.json", "w") as f:
        json.dump(ll_history, f, indent=2)
    fig, ax = plt.subplots(1, 1, figsize=(6, 3))
    ave_ll_history = [
        ll / n
        for ll, n in zip(ll_history["log_likelihood"], ll_history["total_seq_num"])
    ]
    ax.plot(ll_history["step"], ave_ll_history)
    ax.set_ylabel(r"$E[\log P(X)]$")
    ax.set_xlabel("iteration steps")
    ax.grid(True)

    ax.set_xlim((-2, len(ll_history["step"])))
    # fig.set_tight_layout(True)
    fig.set_layout_engine("tight")
    out_figname = "out_hmm_training_loglikelihood.png"

    fig.savefig(out_figname)

    #
    state_name = ["A dominant", "B dominant", "Transient"]
    # state_labels = ["A boom", "B boom", "Trans."]
    names = ["A", "B", "C", "D"]
    fig, axs = fig, axs = plt.subplots(1, M, figsize=(9, 3), sharey=True)
    for b, b0, b_t, st_name, ax in zip(
        hmm.obs_prob, hmm_init.obs_prob, hmm_orig.obs_prob, state_name, axs
    ):
        print(b)
        ax.bar(names, b, alpha=0.75, label="est")
        ax.scatter(np.arange(0, len(names)), b0, label="init")
        ax.scatter(np.arange(0, len(names)), b_t, alpha=0.75, label="true")
        ax.set_title(st_name)
        ax.set_ylim([0, 1.0])
        ax.legend()
    fig.savefig("out_hmm_training_outprob_dist.png")
