import pickle

import matplotlib.pyplot as plt


def plot_gamma(ax, _gamma, state_labels: list = []):
    """Plot the state occupation probabilities (gamma) over time.

    Args:
        ax (_type_): Matplotlib axis to plot on.
        _gamma (_type_): State occupation probabilities.
        state_labels (list, optional): Labels for the states. Defaults to [].

    Returns:
        ax (plt.Axes): The axis with the plot.
    """

    ax.imshow(_gamma.transpose(), cmap="Reds", vmin=0, vmax=1)
    ax.set_xlabel("time index")

    T, M = _gamma.shape
    if len(state_labels) == M:
        ax.set_yticks([0, 1, 2], labels=state_labels)
    else:
        ax.set_ylabel("state index")
    ax.invert_yaxis()  # labels read top-to-bottom
    # ax.set_ylabel('state index')
    return ax


def plot_likelihood(ax, steps: list, log_likelihood: list, ylabel: str = "log P(X)"):
    ax.plot(steps, log_likelihood)
    ax.grid(True)
    ax.set_xlabel("iteration steps")
    ax.set_ylabel(ylabel)


def plot_checkpoint_dir(ckpt_file):
    """not implmeneted yet.

    Args:
        ckpt_file (_type_): _description_
    """

    state_name = ["A dominant", "B dominant", "Transient"]
    names = ["A", "B", "C", "D"]
    with open(ckpt_file, "rb") as f:
        model = pickle.load(f)
        hmm = model.get("model", None)
        # model_type = model.get("model_type", "")
        #                     'total_likelihood': total_likelihood,
        #                     'total_sequence_num': len(obss_seqs),
        #                     'total_obs_num': total_obs_num,
        #                     'iteration': itr_count},
    M = len(state_name)
    fig, axs = plt.subplots(1, M, figsize=(9, 3), sharey=True)
    for b, st_name, ax in zip(hmm.obs_prob, state_name, axs):
        ax.bar(names, b, alpha=0.75)
        ax.set_title(st_name)
        ax.set_ylim([0, 1.0])
    fig.savefig("hmm_outprob_dist.png")
