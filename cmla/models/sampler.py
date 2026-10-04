import logging
import numpy as np

from cmla.models.hmm import HMM

logger = logging.getLogger(__name__)


def generate_sample_parameter(
    num_components: int = 4, feature_dimension: int = 2, **kwargs
) -> tuple:
    """Generate random GMM parameters.
    Args:
        num_components (int, optional): number of clusters. Defaults to 4.
        feature_dimension (int, optional): dimension of samples. Defaults to 2.  In this case,

    Returns:
        tuple: A tuple containing initial probabilities, mean vectors, and covariance matrices.

    Note: The default covariance matrices are identity matrices.
    """
    if kwargs.get("PRESET", False):
        init_prob = np.array([[3.0, 3.0], [0.0, 2.0], [2.0, -3.5], [-3.0, 0.0]])

        covariances = np.array(
            [
                [[1.0, 0.0], [0.0, 1.0]],
                [[0.3, 0.1], [0.1, 0.1]],
                [[0.6, -0.3], [-0.3, 0.5]],
                [[1.0, 0.8], [0.8, 0.8]],
            ]
        )
        return init_prob, covariances

    init_prob = np.array([1 / num_components] * num_components)
    mean_vectors = np.random.randn(num_components, feature_dimension)
    covariances = np.zeros((num_components, feature_dimension, feature_dimension))
    for k in range(num_components):
        covariances[k, :, :] = np.eye(feature_dimension)
    return init_prob, mean_vectors, covariances


def generate_gmm_samples(
    n_sample: int, weights: np.ndarray, centroids: np.ndarray, covariances: np.ndarray
) -> np.ndarray:
    """Generate samples from a GMM parameter.

    Args:
        n_sample (int): number of samples to be generated.
        weights (np.ndarray): weights of each cluster, shape (K,)
        centroids (np.ndarray): centroids of each cluster, shape (K,D)
        covariances (np.ndarray): covariances of each cluster, shape (K,D,D) or (K,D) for diagonal covariance.

    Returns:
        np.ndarray: generated samples.
    """
    num_cluster = len(weights)
    if num_cluster < 1:
        raise ValueError("num_cluster must be > 0")
    if not np.isclose(np.sum(weights), 1.0):
        raise ValueError("weights must sum to 1.0")
    if np.any(weights < 0.0):
        raise ValueError("weights must be non-negative")
    feature_dim = covariances.shape[1]
    if len(covariances.shape) == 2:
        if covariances.shape[0] != feature_dim:
            raise ValueError("covariances shape mismatch")
    elif len(covariances.shape) == 3:
        if (
            covariances.shape[0] != num_cluster
            or covariances.shape[1] != feature_dim
            or covariances.shape[2] != feature_dim
        ):
            raise ValueError("covariances shape mismatch")
    else:
        raise ValueError("covariances must be 2D or 3D array")
    logger.info(
        f"Generating {n_sample} samples from GMM: K={num_cluster}, D={feature_dim}"
    )

    # Allocate space for samples
    sample_data = np.zeros((n_sample, feature_dim))

    # Determine number of samples for each cluster
    counts = np.random.multinomial(n_sample, weights)  # samples count for each cluster
    logger.info(f"counts={counts}")
    labels = []
    for k in range(num_cluster):
        labels += [k] * counts[k]
    # np.random.shuffle(labels) # shuffle labels

    # Prepare covariance matrices
    if len(covariances.shape) == 2:
        logger.info("Using diagonal covariance")
        covariances = np.array(
            [np.diag(covariances[_k, :]) for _k in range(num_cluster)]
        )
    print(covariances.shape)
    # Prepare matrix L such as L*L = S
    L = [np.linalg.cholesky(covariances[_k, :, :]) for _k in range(num_cluster)]
    i = 0
    for k in range(num_cluster):
        # generate samples for cluster k
        # sample from N(0,I)
        for j in range(counts[k]):
            sample_data[i + j, :] = centroids[k, :] + np.dot(
                L[k], np.random.randn(feature_dim)
            )
        i += counts[k]
    return sample_data, labels


def sample_markov_process(length: int, init_prob, tran_prob):
    """Sample a Markov process with given model parameters.

    Args:
        length (int): length of state sequence
        init_prob (np.ndarray): initial state probability, pi[i] = P(s[t=0]=i)
        tran_prob (np.ndarray): state transition probability, a[i][j] = P(s[t]=j|s[t-1]=i)
    """
    # print(f'init_prob.shape={init_prob.shape} tran_prob.shape={tran_prob.shape}')
    if length < 1:
        raise ValueError(f"Length must be larger than 1. length={length}")
    if not np.isclose(np.sum(init_prob), 1.0):
        raise ValueError("init_prob must sum to 1.0")
    if np.any(init_prob < 0.0):
        raise ValueError("init_prob must be non-negative")
    if np.any(tran_prob < 0.0):
        raise ValueError("tran_prob must be non-negative")
    if not np.all(np.isclose(np.sum(tran_prob, axis=1), 1.0)):
        raise ValueError("Each row of tran_prob must sum to 1.0")
    # Check dimensions
    n_states = len(init_prob)
    if tran_prob.shape != (n_states, n_states):
        raise ValueError("tran_prob shape mismatch")
    # Sample
    s = [np.nan] * length
    s[0] = np.random.choice(n_states, p=init_prob)
    for t in range(1, length):
        s[t] = np.random.choice(n_states, p=tran_prob[s[t - 1], :])
    return s


def sample_lengths(ave_len: int, num: int):
    """Determin lengths of sequences by possion distribution.

    Args:
        ave_len (int): average of sample length
        num (int): number of sequence to be generated
    """
    lengths = np.random.poisson(ave_len, num)
    if len(lengths[lengths == 0]) > 0:
        for i in np.where(lengths == 0):
            while True:
                v = np.random.poisson(ave_len, 1)[0]
                if v > 0:
                    lengths[i] = v
                    break
    return lengths


def sample_multiple_markov_process(num: int, init_prob, tran_prob):
    """
    Sampling multiple Markov processes with given model paremeters.

    Args:
        num (int): number of sequences to be generated
        init_prob (np.ndarray): initial state probability, pi[i] = P(s[t=0]=i)
        tran_prob (np.ndarray): state transition probability, a[i][j] = P(s[t=j] | s[t-1]=i)
    """
    assert num > 0
    lengths = sample_lengths(10, num)
    print("lengths=", lengths)
    x = []
    for _l in lengths:
        x1 = sample_markov_process(_l, init_prob, tran_prob)
        x.append(x1)
    return x


def sampling_from_hmm(sequence_lengths, hmm: HMM):
    """sample HMM output and its hidden states from given parameters

    Args:
        n_sequence (int): number of sequences to be generated
        hmm (HMM): Hidden Markov Model instance
    """
    out = []
    for _l in sequence_lengths:
        states = sample_markov_process(_l, hmm.init_state, hmm.state_tran)
        obs = []
        for s_t in states:
            # sample x from p(x|s[t])
            x = np.random.choice(hmm.obs_prob.shape[1], p=hmm.obs_prob[s_t, :])
            obs.append(x)
        out.append(obs)
    return states, out


def save_sequences_with_blank(filename: str, sequences: list[np.ndarray]):
    """
    sequences: list of 2D numpy arrays (each sequence: shape=(T, D) or (T,))
    """
    with open(filename, "w") as f:
        for seq in sequences:
            np.savetxt(f, seq, delimiter=",", fmt="%.6f")
            f.write("\n")


def load_sequences_with_blank(filename: str):
    """
    Return a list of sequences from a file. blank line separates sequences.
    """
    sequences = []
    current = []
    with open(filename, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line == "":
                if current:
                    sequences.append(np.array(current, dtype=float))
                    current = []
            else:
                current.append([float(x) for x in line.split(",")])
        if current:
            sequences.append(np.array(current, dtype=float))
    return sequences
