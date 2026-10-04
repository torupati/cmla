# Emission (observation) distributions of Hidden Markov Models.
# HMM delegates everything that depends on the observation type to an Emission:
# log b_j(x[t]), sufficient statistics given P(s[t]=j|X), M-step, sampling and serialization.

from abc import ABC, abstractmethod
from logging import getLogger

import numpy as np
from scipy.special import logsumexp

from .gaussian import diag_gaussian_log_pdf

logger = getLogger(__name__)


class Emission(ABC):
    """Base class of HMM emission distributions b_j(x) = P(x | s=j)."""

    @property
    @abstractmethod
    def num_states(self) -> int:
        """Number of hidden states."""

    @abstractmethod
    def log_prob(self, obss) -> np.ndarray:
        """Calculate log emission probabilities.

        Args:
            obss: observation sequence of length T

        Returns:
            np.ndarray: (T, M)-shape array, log b_j(x[t]). May contain -inf.
        """

    @abstractmethod
    def reset_stats(self) -> None:
        """Clear sufficient statistics."""

    @abstractmethod
    def accumulate(self, obss, gamma: np.ndarray) -> None:
        """Accumulate sufficient statistics.

        Args:
            obss: observation sequence of length T
            gamma (np.ndarray): (T, M)-shape array, P(s[t]=j | X)
        """

    @abstractmethod
    def update(self) -> None:
        """Update parameters from sufficient statistics, then reset them."""

    @abstractmethod
    def sample(self, state: int, rng=None):
        """Draw one observation from b_state(x)."""

    @abstractmethod
    def to_dict(self) -> dict:
        """Parameters as a JSON-serializable dict."""

    @classmethod
    @abstractmethod
    def from_dict(cls, d: dict) -> "Emission":
        """Create an instance from the output of to_dict()."""


class DiscreteEmission(Emission):
    """Categorical emission. probs[j, k] = P(x=k | s=j)."""

    def __init__(self, num_states: int, num_symbols: int):
        """Initialize each state's distribution uniformly at random.

        Args:
            num_states (int): number of hidden states
            num_symbols (int): number of observation symbols (categories)
        """
        if num_states < 1:
            raise ValueError(f"num_states must be > 0. got {num_states}")
        if num_symbols < 1:
            raise ValueError(f"num_symbols must be > 0. got {num_symbols}")
        self.probs = np.zeros((num_states, num_symbols))
        for m in range(num_states):
            self.probs[m, :] = np.random.uniform(0, 1, num_symbols)
            self.probs[m, :] = self.probs[m, :] / self.probs[m, :].sum()
        self.reset_stats()

    def __repr__(self) -> str:
        return f"DiscreteEmission(M={self.num_states}, K={self.num_symbols})"

    @property
    def num_states(self) -> int:
        return self.probs.shape[0]

    @property
    def num_symbols(self) -> int:
        """Number of observation symbols. Read-only."""
        return self.probs.shape[1]

    def log_prob(self, obss) -> np.ndarray:
        x = self._symbols(obss)
        with np.errstate(divide="ignore"):
            return np.log(self.probs[:, x].T)

    def _symbols(self, obss) -> np.ndarray:
        """Observation as int array, checking 0 <= x < num_symbols."""
        x = np.asarray(obss, dtype=int)
        if x.size and (x.min() < 0 or x.max() >= self.num_symbols):
            raise ValueError(
                f"observation symbols must be in [0, {self.num_symbols - 1}]. "
                f"got [{x.min()}, {x.max()}]"
            )
        return x

    def reset_stats(self) -> None:
        self._count = np.zeros(self.probs.shape)

    def accumulate(self, obss, gamma: np.ndarray) -> None:
        # count[j, x[t]] += gamma[t, j]
        np.add.at(self._count.T, self._symbols(obss), gamma)

    def update(self) -> None:
        for m in range(self.num_states):
            # a state never visited keeps its previous distribution
            if self._count[m, :].sum() > 0.0:
                self.probs[m, :] = self._count[m, :] / self._count[m, :].sum()
        self.reset_stats()

    def sample(self, state: int, rng=None) -> int:
        choice = np.random.choice if rng is None else rng.choice
        return int(choice(self.num_symbols, p=self.probs[state, :]))

    def to_dict(self) -> dict:
        return {"type": "discrete", "probs": self.probs.tolist()}

    @classmethod
    def from_dict(cls, d: dict) -> "DiscreteEmission":
        emission = cls.__new__(cls)  # skip random initialization
        emission.probs = np.asarray(d["probs"], dtype=float)
        emission.reset_stats()
        return emission


class GMMEmission(Emission):
    """Gaussian mixture emission with diagonal covariance.

    b_j(x) = sum_k weights[j, k] N(x; means[j, k], diag(covs[j, k]))
    """

    def __init__(
        self,
        num_states: int,
        num_mixtures: int,
        feature_dim: int,
        cov_type: str = "diag",
        var_floor: float | np.ndarray = 1.0e-3,
        min_occupancy: float = 1.0e-3,
    ):
        """Initialize equal weights, standard normal random means and unit variances.

        Args:
            num_states (int): number of hidden states (M)
            num_mixtures (int): number of Gaussians per state (K)
            feature_dim (int): dimension of observation vector (D)
            cov_type (str): covariance type. Only "diag" is supported.
            var_floor (float | np.ndarray): lower bound of variance, scalar or (D,)
            min_occupancy (float): a component whose occupancy sum_t r[t,j,k] is
                below this value keeps its previous mean and variance in update().
        """
        if num_states < 1:
            raise ValueError(f"num_states must be > 0. got {num_states}")
        if num_mixtures < 1:
            raise ValueError(f"num_mixtures must be > 0. got {num_mixtures}")
        if feature_dim < 1:
            raise ValueError(f"feature_dim must be > 0. got {feature_dim}")
        if cov_type != "diag":
            raise NotImplementedError(f"cov_type '{cov_type}' is not supported yet")
        self.cov_type = cov_type
        self.var_floor = var_floor
        self.min_occupancy = min_occupancy
        self.weights = np.ones((num_states, num_mixtures)) / num_mixtures  # (M, K)
        self.means = np.random.randn(num_states, num_mixtures, feature_dim)  # (M, K, D)
        self.covs = np.ones((num_states, num_mixtures, feature_dim))  # (M, K, D)
        self.reset_stats()

    def __repr__(self) -> str:
        return (
            f"GMMEmission(M={self.num_states}, K={self.num_mixtures}, "
            f"D={self.feature_dim}, cov_type={self.cov_type})"
        )

    @property
    def num_states(self) -> int:
        return self.means.shape[0]

    @property
    def num_mixtures(self) -> int:
        """Number of Gaussians per state. Read-only."""
        return self.means.shape[1]

    @property
    def feature_dim(self) -> int:
        """Dimension of observation vector. Read-only."""
        return self.means.shape[2]

    def _log_component_prob(self, obss) -> np.ndarray:
        """log weights[j,k] + log N(x[t]; means[j,k], covs[j,k]), shape (T, M, K)."""
        x = np.asarray(obss, dtype=float).reshape(-1, self.feature_dim)
        with np.errstate(divide="ignore"):
            log_weights = np.log(self.weights)
        return log_weights + diag_gaussian_log_pdf(x, self.means, self.covs)

    def log_prob(self, obss) -> np.ndarray:
        return logsumexp(self._log_component_prob(obss), axis=2)

    def reset_stats(self) -> None:
        self._s0 = np.zeros(self.weights.shape)  # sum_t r[t,j,k]
        self._s1 = np.zeros(self.means.shape)  # sum_t r[t,j,k] x[t]
        self._s2 = np.zeros(self.means.shape)  # sum_t r[t,j,k] x[t]^2

    def accumulate(self, obss, gamma: np.ndarray) -> None:
        if self._s1.shape != self.means.shape:  # parameters were replaced
            self.reset_stats()
        x = np.asarray(obss, dtype=float).reshape(-1, self.feature_dim)
        log_comp = self._log_component_prob(x)
        # P(k | x[t], s[t]=j)
        log_post = log_comp - logsumexp(log_comp, axis=2, keepdims=True)
        r = gamma[:, :, np.newaxis] * np.exp(log_post)  # (T, M, K)
        self._s0 += r.sum(axis=0)
        self._s1 += np.einsum("tmk,td->mkd", r, x)
        self._s2 += np.einsum("tmk,td->mkd", r, x**2)

    def update(self) -> None:
        for m in range(self.num_states):
            occupancy = self._s0[m].sum()
            if occupancy <= 0.0:  # state never visited
                continue
            self.weights[m] = self._s0[m] / occupancy
            for k in range(self.num_mixtures):
                n = self._s0[m, k]
                if n < self.min_occupancy:
                    logger.warning(
                        "component (state=%d, mixture=%d) occupancy %g < %g. "
                        "mean and variance are not updated.",
                        m,
                        k,
                        n,
                        self.min_occupancy,
                    )
                    continue
                mean = self._s1[m, k] / n
                var = self._s2[m, k] / n - mean**2
                self.means[m, k] = mean
                self.covs[m, k] = np.maximum(var, self.var_floor)
        self.reset_stats()

    def sample(self, state: int, rng=None) -> np.ndarray:
        rng = np.random if rng is None else rng
        k = rng.choice(self.num_mixtures, p=self.weights[state])
        return self.means[state, k] + np.sqrt(
            self.covs[state, k]
        ) * rng.standard_normal(self.feature_dim)

    def to_dict(self) -> dict:
        return {
            "type": "gmm",
            "cov_type": self.cov_type,
            "weights": self.weights.tolist(),
            "means": self.means.tolist(),
            "covs": self.covs.tolist(),
            "var_floor": np.asarray(self.var_floor).tolist(),
            "min_occupancy": self.min_occupancy,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "GMMEmission":
        if d.get("cov_type", "diag") != "diag":
            raise NotImplementedError(
                f"cov_type '{d['cov_type']}' is not supported yet"
            )
        emission = cls.__new__(cls)  # skip random initialization
        emission.cov_type = "diag"
        emission.weights = np.asarray(d["weights"], dtype=float)
        emission.means = np.asarray(d["means"], dtype=float)
        emission.covs = np.asarray(d["covs"], dtype=float)
        var_floor = d.get("var_floor", 1.0e-3)
        emission.var_floor = (
            var_floor if np.isscalar(var_floor) else np.asarray(var_floor)
        )
        emission.min_occupancy = d.get("min_occupancy", 1.0e-3)
        M, K, D = emission.means.shape
        if emission.weights.shape != (M, K) or emission.covs.shape != (M, K, D):
            raise ValueError(
                f"inconsistent shapes: weights {emission.weights.shape}, "
                f"means {emission.means.shape}, covs {emission.covs.shape}"
            )
        emission.reset_stats()
        return emission


_EMISSION_TYPES: dict[str, type[Emission]] = {
    "discrete": DiscreteEmission,
    "gmm": GMMEmission,
}


def emission_from_dict(d: dict) -> Emission:
    """Create an emission from the output of Emission.to_dict().

    Args:
        d (dict): serialized emission. d["type"] selects the class.

    Returns:
        Emission: restored emission
    """
    emission_type = d.get("type")
    if emission_type not in _EMISSION_TYPES:
        raise ValueError(
            f"Unknown emission type: {emission_type}. "
            f"Expected one of {sorted(_EMISSION_TYPES)}"
        )
    return _EMISSION_TYPES[emission_type].from_dict(d)
