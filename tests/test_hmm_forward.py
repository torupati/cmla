import matplotlib.pyplot as plt
import numpy as np

from cmla.models.hmm import HMM
from cmla.plots.hmm_plot import plot_gamma


def test_forward_backward():
    """
    test viterbi_search
    """
    # Assume there are 3 observation states, 2 hidden states, and 5 observations
    M = 3
    D = 4
    hmm = HMM(M, D)
    hmm.init_state = np.array([0.34, 0.33, 0.33])
    hmm._log_init_state = np.log(hmm.init_state)
    hmm.state_tran = np.array(
        [[0.60, 0.35, 0.05], [0.01, 0.60, 0.39], [0.30, 0.00, 0.70]]
    )
    hmm.obs_prob = np.array(
        [[0.70, 0.10, 0.10, 0.10], [0.01, 0.09, 0.80, 0.10], [0.1, 0.45, 0.00, 0.45]]
    )
    print(f"hmm={hmm}")

    state_name = ["A dominant", "C dominant", "Transient"]
    state_labels = ["A boom", "C boom", "Trans."]
    names = ["A", "B", "C", "D"]
    fig, axs = fig, axs = plt.subplots(1, M, figsize=(9, 3), sharey=True)
    for b, st_name, ax in zip(hmm.obs_prob, state_name, axs):
        print(b)
        ax.bar(names, b, alpha=0.75)
        ax.set_title(st_name)
        ax.set_ylim([0, 1.0])
        ax.grid(True)
    fig.savefig("hmm_outprob_dist.png")

    obs = [
        0,
        1,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        2,
        3,
        2,
        2,
        2,
        3,
        2,
        2,
        1,
        3,
        3,
        1,
        1,
        3,
        0,
        0,
        0,
        0,
    ]
    # np.random.seed(0)
    # st_orig, obs = sampling_from_hmm([10], hmm)
    print("observations({})={}".format(len(obs), obs))

    fig, axes = plt.subplots(3, 1)

    obsprob = hmm.calculate_prob(obs)
    _fwd, _fwd_scale = hmm.forward_algorithm(obsprob)
    print("fwd", _fwd)
    print("fwd_scale.shape=", _fwd_scale.shape)
    plot_gamma(axes[0], _fwd, state_labels)
    axes[0].set_title("Forward")

    _gamma, _xi, _logprob = hmm.forward_backward_algorithm_linear(obs)
    print("gamma", _gamma)
    print("xi", _xi)
    print(_logprob)
    # assert -11.524219471588987
    plot_gamma(axes[1], _gamma, state_labels)
    axes[1].set_title("Forward-Backward")

    _gamma, _xi, _logprob = hmm.forward_viterbi(obs)
    print("gamma", _gamma)
    print("xi", _xi)
    print(_logprob)
    # assert -11.524219471588987
    axes[2].set_title("Viterbi search")
    plot_gamma(axes[2], _gamma, state_labels)

    fig.savefig("test_gamma.png")
