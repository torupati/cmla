# Matplotlib plotting functions for K-means clustering
import matplotlib.pyplot as plt


def plot_training_samples(ax, x, **kwargs) -> plt.Axes:
    """Plot training samples in 2D

    Args:
        ax (plt.Axes): _description_
        x (np.ndarray): _description_
    """
    idx0: int = kwargs.get("idx0", 0)
    idx1: int = kwargs.get("idx1", 1)
    x_range_x1: list = [min(x[:, idx0]), max(x[:, idx0])]
    x_range_x2: list = [min(x[:, idx1]), max(x[:, idx1])]
    ax.plot(
        x[:, 0],
        x[:, 1],
        marker=".",
        # markerfacecolor=X_col[k],
        markeredgecolor="k",
        markersize=6,
        alpha=0.5,
        linestyle="none",
    )
    ax.set_xlim(x_range_x1)
    ax.set_ylim(x_range_x2)
    ax.grid("True")
    ax.set_xlabel("$x_1$")
    ax.set_ylabel("$x_2$")
    ax.set_aspect("equal")
    return ax


def plot_distortion_history(cost_history: list) -> tuple:
    """Plot distortion history
    Args:
        cost_history (list): _description_
    Returns:
        tuple: (fig, ax)
    """
    fig, ax = plt.subplots(1, 1, figsize=(8, 4))
    ax.plot(
        range(0, len(cost_history)),
        cost_history,
        color="black",
        linestyle="-",
        marker="o",
    )
    ax.set_yscale("log")
    ax.set_xlim([0, len(cost_history)])
    ax.grid(True)
    ax.set_xlabel("Iteration Step")
    ax.set_ylabel("Loss")
    return fig, ax


def plot_data_with_centroid(ax, x, r, mu, kmeans_param_ref: dict = {}, **kwargs):
    # X_col = ['cornflowerblue', "orange", 'black', 'white', 'red']

    # plot training samples
    ax.plot(
        x[:, 0],
        x[:, 1],
        marker=".",
        markeredgecolor="k",
        markersize=6,
        alpha=0.5,
        linestyle="none",
    )

    K = mu.shape[0]
    for k in range(K):
        ax.plot(
            x[r[:, k] == 1, 0],
            x[r[:, k] == 1, 1],
            marker=".",
            # markerfacecolor=X_col[k],
            markeredgecolor="k",
            markersize=6,
            alpha=0.5,
            linestyle="none",
        )
    # plot ground truth if given.
    mu_ref = kmeans_param_ref.get("Mu")
    if mu_ref is not None:
        for k in range(K):
            ax.plot(
                mu_ref[k, 0],
                mu_ref[k, 1],
                marker="*",
                # markerfacecolor=X_col[k],
                markersize=8,
                markeredgecolor="k",
                markeredgewidth=2,
            )
    for k in range(K):
        ax.plot(
            mu[k, 0],
            mu[k, 1],
            marker="o",
            # markerfacecolor=X_col[k],
            markersize=10,
            markeredgecolor="k",
            markeredgewidth=1,
        )
    X_range_x1 = [min(x[:, 0]), max(x[:, 0])]
    X_range_x2 = [min(x[:, 1]), max(x[:, 1])]
    ax.set_xlim(X_range_x1)
    ax.set_ylim(X_range_x2)
    ax.grid(True)
    ax.set_aspect("equal")
