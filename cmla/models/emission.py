# Emission (observation) distributions of Hidden Markov Models.
# HMM delegates everything that depends on the observation type to an Emission:
# log b_j(x[t]), sufficient statistics given P(s[t]=j|X), M-step, sampling and serialization.

from abc import ABC, abstractmethod

import numpy as np


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
        with np.errstate(divide="ignore"):
            return np.log(self.probs[:, np.asarray(obss, dtype=int)].T)

    def reset_stats(self) -> None:
        self._count = np.zeros(self.probs.shape)

    def accumulate(self, obss, gamma: np.ndarray) -> None:
        # count[j, x[t]] += gamma[t, j]
        np.add.at(self._count.T, np.asarray(obss, dtype=int), gamma)

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


_EMISSION_TYPES: dict[str, type[Emission]] = {
    "discrete": DiscreteEmission,
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
