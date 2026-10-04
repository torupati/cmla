import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Ellipse

from cmla.models.emission import DiscreteEmission, GMMEmission
from cmla.models.hmm import HMM


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
        ax.set_yticks(range(M), labels=state_labels)
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


def plot_discrete_emission(
    axs, emission: DiscreteEmission, state_names=None, symbol_names=None
):
    """Bar chart of P(x | s=j), one axis per state.

    Args:
        axs: sequence of M matplotlib axes
        emission (DiscreteEmission): emission to plot
        state_names (list[str], optional): titles. Defaults to "state j".
        symbol_names (list[str], optional): x tick labels. Defaults to symbol index.
    """
    M, K = emission.probs.shape
    state_names = state_names or [f"state {j}" for j in range(M)]
    symbol_names = symbol_names or [str(k) for k in range(K)]
    for b, name, ax in zip(emission.probs, state_names, axs):
        ax.bar(symbol_names, b, alpha=0.75)
        ax.set_title(name)
        ax.set_ylim([0, 1.0])
    return axs


def plot_gmm_emission(
    ax,
    emission: GMMEmission,
    dims: tuple[int, int] = (0, 1),
    x=None,
    states=None,
    n_std: float = 2.0,
    state_names=None,
):
    """Gaussians of each state as ellipses (n_std standard deviations) in 2 dimensions.

    Line width and opacity of each ellipse follow the mixture weight.

    Args:
        ax: matplotlib axis
        emission (GMMEmission): emission to plot
        dims (tuple[int, int]): feature dimensions to plot
        x (np.ndarray, optional): (N, D) samples drawn as points
        states (np.ndarray, optional): (N,) state of each sample, used for color
        n_std (float): ellipse radius in standard deviations
        state_names (list[str], optional): legend labels. Defaults to "state j".
    """
    d0, d1 = dims
    M, K, _ = emission.means.shape
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    state_names = state_names or [f"state {j}" for j in range(M)]
    if x is not None:
        x = np.asarray(x)
        if states is None:
            ax.scatter(x[:, d0], x[:, d1], s=4, color="gray", alpha=0.4)
        else:
            states = np.asarray(states)
            for j in range(M):
                sel = states == j
                ax.scatter(
                    x[sel, d0],
                    x[sel, d1],
                    s=4,
                    color=colors[j % len(colors)],
                    alpha=0.4,
                )
    for j in range(M):
        color = colors[j % len(colors)]
        for k in range(K):
            mean = emission.means[j, k]
            std = np.sqrt(emission.covs[j, k])
            w = emission.weights[j, k]
            ax.add_patch(
                Ellipse(
                    (mean[d0], mean[d1]),
                    width=2 * n_std * std[d0],
                    height=2 * n_std * std[d1],
                    fill=False,
                    edgecolor=color,
                    linewidth=0.5 + 2.5 * w,
                    alpha=0.3 + 0.7 * w,
                )
            )
        ax.scatter(
            emission.means[j, :, d0],
            emission.means[j, :, d1],
            marker="x",
            color=color,
            label=state_names[j],
        )
    ax.autoscale_view()
    ax.set_xlabel(f"x[{d0}]")
    ax.set_ylabel(f"x[{d1}]")
    ax.legend()
    return ax


def plot_emission(hmm: HMM, **kwargs):
    """Plot the emission distribution of an HMM in a new figure.

    Discrete: one bar chart per state. GMM: ellipses of the first two dimensions.
    kwargs are passed to plot_discrete_emission or plot_gmm_emission.

    Returns:
        matplotlib.figure.Figure: the figure
    """
    emission = hmm.emission
    if isinstance(emission, DiscreteEmission):
        M = emission.num_states
        fig, axs = plt.subplots(1, M, figsize=(3 * M, 3), sharey=True, squeeze=False)
        plot_discrete_emission(axs[0], emission, **kwargs)
    elif isinstance(emission, GMMEmission):
        fig, ax = plt.subplots(1, 1, figsize=(6, 6))
        if emission.feature_dim == 1:
            raise ValueError("plot of 1-dimensional GMM emission is not supported")
        plot_gmm_emission(ax, emission, **kwargs)
    else:
        raise TypeError(f"Unsupported emission: {type(emission).__name__}")
    fig.set_layout_engine("tight")
    return fig


def plot_checkpoint_dir(ckpt_file, out_file: str = "hmm_outprob_dist.png"):
    """Plot the emission distribution of a model saved by hmm_baum_welch.

    Args:
        ckpt_file (str): checkpoint file (or any file HMM.load accepts)
        out_file (str): output image file
    """
    fig = plot_emission(HMM.load(str(ckpt_file)))
    fig.savefig(out_file)
    plt.close(fig)
