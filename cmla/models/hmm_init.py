# Initialization of GMM-HMM emission parameters from training data.
# EM (Baum-Welch) only finds a local optimum, so the starting point matters.

from logging import getLogger

import numpy as np

from .emission import GMMEmission
from .hmm import HMM
from .kmeans import kmeans_clustering

logger = getLogger(__name__)


def uniform_segmentation(length: int, num_states: int) -> np.ndarray:
    """Split time steps 0..length-1 into num_states contiguous segments of equal size.

    Args:
        length (int): sequence length T
        num_states (int): number of states M

    Returns:
        np.ndarray: (T,) state index of each time step, non-decreasing from 0 to M-1
    """
    return np.arange(length) * num_states // length


def init_gmm_hmm(
    hmm: HMM,
    sequences: list,
    method: str = "uniform_segment",
    kmeans_max_it: int = 20,
) -> HMM:
    """Initialize weights, means and variances of a GMM-HMM from data (flat start).

    1. Assign every frame to a state.
       - "uniform_segment": split each sequence into M equal segments, segment j
         to state j. Suits left-to-right models.
       - "kmeans": k-means with M clusters over all frames, cluster j to state j.
         Suits ergodic models.
    2. For each state, k-means with K clusters over its frames gives the means.
       Weights and variances are computed from the k-means assignment.

    Initial state and transition probabilities are not changed. Run
    hmm_viterbi_training or hmm_baum_welch afterwards.

    Args:
        hmm (HMM): model with GMMEmission. Modified in place.
        sequences (list[np.ndarray]): observation sequences, each (T, D)
        method (str): "uniform_segment" or "kmeans"
        kmeans_max_it (int): maximum iterations of each k-means

    Returns:
        HMM: the initialized model (same object as hmm)
    """
    emission = hmm.emission
    if not isinstance(emission, GMMEmission):
        raise TypeError(
            f"init_gmm_hmm requires GMMEmission. got {type(emission).__name__}"
        )
    M, K, D = emission.means.shape
    seqs = [np.asarray(x, dtype=float).reshape(-1, D) for x in sequences]
    frames = np.concatenate(seqs)

    if method == "uniform_segment":
        state_of_frame = np.concatenate([uniform_segmentation(len(x), M) for x in seqs])
    elif method == "kmeans":
        state_of_frame = _kmeans_labels(frames, M, kmeans_max_it)
    else:
        raise ValueError(f"Unknown method: {method}")

    for j in range(M):
        x_j = frames[state_of_frame == j]
        if len(x_j) < K:
            raise ValueError(
                f"state {j} has {len(x_j)} frames, fewer than {K} mixtures"
            )
        labels = _kmeans_labels(x_j, K, kmeans_max_it)
        state_var = np.maximum(x_j.var(axis=0), emission.var_floor)
        for k in range(K):
            x_jk = x_j[labels == k]
            # an empty cluster keeps a small weight so that EM can still use it
            emission.weights[j, k] = max(len(x_jk), 1)
            if len(x_jk) == 0:
                emission.means[j, k] = x_j.mean(axis=0)
                emission.covs[j, k] = state_var
                continue
            emission.means[j, k] = x_jk.mean(axis=0)
            if len(x_jk) < 2:
                emission.covs[j, k] = state_var
            else:
                emission.covs[j, k] = np.maximum(x_jk.var(axis=0), emission.var_floor)
        emission.weights[j] /= emission.weights[j].sum()
        logger.info(
            "state %d: %d frames, mixture sizes %s",
            j,
            len(x_j),
            np.bincount(labels, minlength=K).tolist(),
        )
    emission.reset_stats()
    return hmm


def _kmeans_labels(x: np.ndarray, num_clusters: int, max_it: int) -> np.ndarray:
    """Cluster x (N, D) by k-means and return the nearest-centroid index of each sample."""
    if num_clusters == 1:
        return np.zeros(len(x), dtype=int)
    model, _ = kmeans_clustering(x, _kmeans_plus_plus(x, num_clusters), max_it=max_it)
    dist = ((x[:, np.newaxis, :] - model.Mu[np.newaxis, :, :]) ** 2).sum(axis=2)
    return dist.argmin(axis=1)


def _kmeans_plus_plus(x: np.ndarray, num_clusters: int) -> np.ndarray:
    """k-means++ seeding: each new centroid is a sample drawn with probability
    proportional to its squared distance from the nearest chosen centroid.

    Returns:
        np.ndarray: (num_clusters, D) initial centroids (copies of samples)
    """
    centroids = [x[np.random.randint(len(x))]]
    min_dist = ((x - centroids[0]) ** 2).sum(axis=1)
    for _ in range(1, num_clusters):
        total = min_dist.sum()
        if total > 0.0:
            idx = np.random.choice(len(x), p=min_dist / total)
        else:  # all samples coincide with chosen centroids
            idx = np.random.randint(len(x))
        centroids.append(x[idx])
        min_dist = np.minimum(min_dist, ((x - x[idx]) ** 2).sum(axis=1))
    return np.array(centroids, dtype=float)
