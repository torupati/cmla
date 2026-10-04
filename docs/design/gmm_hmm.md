# Design: HMM with GMM Emissions (GMM-HMM)

Status: **Proposed** (2026-10-04)

## Goal

Extend `cmla.models.hmm.HMM` from discrete emission probabilities to Gaussian
mixture emissions:

$$
b_j(\mathbf{x}) = \sum_{k=1}^{K} c_{jk}\,\mathcal{N}(\mathbf{x};\boldsymbol\mu_{jk},\boldsymbol\Sigma_{jk})
$$

while keeping the discrete HMM working unchanged. This follows the idea noted in
`MEMO.md`: unify continuous and discrete observations at the point where the
emission probabilities are handed to the HMM as a matrix.

## 1. Current state

The forward–backward core is already almost emission-agnostic:

| Part | Depends on discrete emission? |
|---|---|
| `forward_algorithm(obsprob)` | **No**: takes a `(T, M)` matrix |
| backward pass / γ, ξ in `forward_backward_algorithm_linear` | No, except the `calculate_prob(obss)` call |
| `viterbi_search` | **Yes**: builds one-hot log-probs inline |
| `calc_logobss`, `calculate_prob` | Yes |
| `push_sufficient_statistics` (`_obs_count`), `obs_prob` part of `update_parameters` | Yes |
| `__init__`, save/load, checkpoint, `sampler.sampling_from_hmm`, `utils.randomize_observation_probabilities`, `hmm_plot` | Yes (`obs_prob`) |

The emission-specific parts are: **computing $\log b_j(x_t)$**,
**accumulating statistics given γ**, **the M-step**, **sampling**, and
**serialization**. These move behind one interface. Transition and
initial-state logic stays untouched.

## 2. Architecture: emission strategy

```
cmla/models/
  emission.py      NEW: Emission (ABC), DiscreteEmission, GMMEmission
  gaussian.py      NEW: vectorised log N(x; mu, Sigma), later shared with gmm.py
  hmm.py           HMM holds self.emission; transition/initial logic unchanged
```

```python
class Emission(ABC):
    num_states: int

    # --- evaluation
    @abstractmethod
    def log_prob(self, X) -> np.ndarray: ...          # (T, M): log b_j(x_t)

    # --- training (sufficient-statistics pattern, same as the HMM now)
    @abstractmethod
    def reset_stats(self) -> None: ...
    @abstractmethod
    def accumulate(self, X, gamma: np.ndarray) -> None: ...  # gamma (T, M) = P(s_t=j | X)
    @abstractmethod
    def update(self) -> None: ...                     # M-step, then reset_stats()

    # --- generation / IO
    @abstractmethod
    def sample(self, state: int, rng) -> np.ndarray | int: ...
    @abstractmethod
    def to_dict(self) -> dict: ...                    # JSON-safe
    @classmethod
    @abstractmethod
    def from_dict(cls, d: dict) -> "Emission": ...
```

`accumulate` receives only the state posterior γ. The GMM recomputes its
per-component posteriors internally from its own densities, so the HMM never
needs to know about mixtures, and Viterbi training (hard γ) and Baum–Welch
(soft γ) share one code path.

### DiscreteEmission (moves the current code here)

- Parameters: `probs` with shape `(M, K)`, which is today's `obs_prob`.
- `log_prob(X) = log(probs[:, X].T + eps)`. One vectorised line replaces the
  per-t/per-s loops.
- Statistics: $\text{count}[j,k] \mathrel{+}= \sum_t \gamma_{tj}\,[x_t = k]$
  (computed with `np.add.at`).

### GMMEmission

Parameters, with $K$ mixtures per state, feature dimension $D$, and
`cov_type in {"full", "diag"}`:

| Name | Shape |
|---|---|
| `weights` $c_{jk}$ | `(M, K)` |
| `means` $\mu_{jk}$ | `(M, K, D)` |
| `covs` $\Sigma_{jk}$ | `(M, K, D, D)` for full, `(M, K, D)` for diag |
| `var_floor` | scalar or `(D,)` |

**Evaluation**

$$
\log b_j(\mathbf{x}_t) = \operatorname{logsumexp}_k \left[\log c_{jk} + \log \mathcal{N}(\mathbf{x}_t;\boldsymbol\mu_{jk},\boldsymbol\Sigma_{jk})\right]
$$

with the component log-densities computed as a `(T, M, K)` array.

**Statistics** (for a given $\gamma_{tj}$):

$$
r_{tjk} = \gamma_{tj}\,\exp\!\left(\log c_{jk} + \log\mathcal{N}_{jk}(\mathbf{x}_t) - \log b_j(\mathbf{x}_t)\right)
$$

$$
S_0[j,k] \mathrel{+}= \sum_t r_{tjk},\quad
S_1[j,k] \mathrel{+}= \sum_t r_{tjk}\,\mathbf{x}_t,\quad
S_2[j,k] \mathrel{+}= \sum_t r_{tjk}\,\mathbf{x}_t\mathbf{x}_t^\top
$$

(for diag, $S_2$ keeps only $\mathbf{x}_t^2$). Use `np.einsum` and avoid the
Python loop over samples used in `GaussianMixtureModel.update_m_step`.

**M-step**

$$
c_{jk} = \frac{S_0[j,k]}{\sum_{k'} S_0[j,k']},\quad
\boldsymbol\mu_{jk} = \frac{S_1[j,k]}{S_0[j,k]},\quad
\boldsymbol\Sigma_{jk} = \frac{S_2[j,k]}{S_0[j,k]} - \boldsymbol\mu_{jk}\boldsymbol\mu_{jk}^\top + \text{var\_floor}\cdot I
$$

(diag: $\sigma^2 \leftarrow \max(\sigma^2, \text{var\_floor})$). If
$S_0[j,k] <$ `min_occupancy`, keep the old parameters for that component and
log a warning (dead-component guard).

## 3. Changes in the `HMM` class

```python
class HMM:
    def __init__(self, num_hidden_states, feature_dim=None,
                 observation_type="discrete", *, emission: Emission | None = None,
                 num_mixtures=1, cov_type="diag"):
        # keep the old positional signature: HMM(M, D) -> DiscreteEmission(M, D)
        # "gmm" -> GMMEmission(M, num_mixtures, feature_dim, cov_type)
        # an explicit emission= overrides observation_type

    @property
    def obs_prob(self):           # compatibility shim, discrete only
        return self.emission.probs

    @obs_prob.setter
    def obs_prob(self, v):
        self.emission.probs = np.asarray(v)
```

The `obs_prob` property lets the existing tests, `utils.py`, `sampler.py`,
`hmm_plot.py` and `sampler_cli.py` run unchanged during migration. On a
GMM-HMM it raises `AttributeError` with a clear message.

| Method | Change |
|---|---|
| `viterbi_search(obss)` | `logB = self.emission.log_prob(obss)`. Delete the one-hot block. The trellis loop stays. |
| `forward_backward_algorithm_linear(obss)` | See the numerics section below. |
| `push_sufficient_statistics(obss, g1, g2)` | Keep the transition statistics. Replace the `_obs_count` loop with `self.emission.accumulate(obss, g1)`. |
| `update_parameters()` | Keep the π/A normalisation, then call `self.emission.update()`. |
| `calc_logobss` / `calculate_prob` | Thin wrappers around `emission.log_prob`, marked deprecated. |
| `save_hmm_and_data` / `load_hmm_and_data` / checkpoint | Use one `to_dict()`/`from_dict()`. |

### Numerics (required for continuous emissions)

Discrete probabilities are ≤ 1 and rarely all zero. Gaussian densities in $D$
dimensions can underflow to 0 for **every** state at once (an outlier frame,
or large $D$), which makes `_alpha_scale[t] = 0` and produces NaN. Fix this
inside the existing scaled-linear forward–backward by normalising per frame:

```python
logB = self.emission.log_prob(obss)          # (T, M)
m_t  = logB.max(axis=1, keepdims=True)
B    = np.exp(logB - m_t)                    # each row's max is 1
alpha, c = self.forward_algorithm(B)
log_prob = np.log(c).sum() + m_t.sum()       # add the offset back
```

γ and ξ don't change, because a per-frame constant cancels when they are
normalised. `forward_algorithm` keeps its signature, so
`tests/test_hmm_forward.py` still passes.

## 4. Data format and I/O

- **Observations:** discrete is `list[np.ndarray[int]]` with shape `(T,)`.
  GMM is `list[np.ndarray[float]]` with shape `(T, D)`.
  `sampler.load_sequences_with_blank` already produces the GMM format.
- **Saved model (v2):**

  ```text
  {"model_type": "HMM", "version": 2,
   "init_state": [...], "state_tran": [[...]],
   "emission": {"type": "gmm", "cov_type": "diag",
                "weights": ..., "means": ..., "covs": ..., "var_floor": 1e-3}}
  ```

  The loader still accepts the v1 format (top-level `obs_prob`) and turns it
  into a `DiscreteEmission`.
- **Sampler:** `sampling_from_hmm` calls `hmm.emission.sample(s_t, rng)`. The
  GMM version picks a component $k \sim c_j$, then draws
  $\mathbf{x} \sim \mathcal{N}(\boldsymbol\mu_{jk}, \boldsymbol\Sigma_{jk})$.
  Add an `HMM-GMM` subcommand to `sampler_cli`.

## 5. Initialisation

EM for a GMM-HMM is very sensitive to the starting point. Provide
`init_gmm_hmm(hmm, sequences, method="uniform_segment")`:

1. **Uniform segmentation (flat start):** split each sequence into $M$ equal
   segments and assign segment $j$ to state $j$. This is the standard method
   for left-to-right models.
2. **For each state,** run `kmeans_clustering` (`cmla/models/kmeans.py`) on its
   frames to get $K$ means. Set the covariances to the cluster variances plus
   the floor and the weights to the cluster proportions.
3. **Optional:** run a few Viterbi-training iterations
   (`hmm_viterbi_training`) before Baum–Welch.

Alternative for ergodic models: run global k-means with $M \cdot K$ clusters
and assign clusters to states.

## 6. Tests to add

| Test | What it checks |
|---|---|
| `test_emission_discrete_equivalence` | `DiscreteEmission.log_prob` equals the old `calc_logobss` exactly; Baum–Welch gives identical parameters before and after the refactor (fixed seed). |
| `test_gmm_emission_logprob` | Matches `scipy.stats.multivariate_normal.logpdf` mixed by hand. |
| `test_gmm_hmm_K1_equals_gaussian_hmm` | With K=1 the result is a plain Gaussian HMM, easy to check analytically. |
| `test_gmm_hmm_monotonic_ll` | The Baum–Welch log-likelihood never decreases. |
| `test_gmm_hmm_recovery` | Sample from a known 2–3 state, 2-mixture, 2-D model, train, and recover the means up to permutation. |
| `test_underflow_frame` | A frame 1e3σ away from every mean must not produce NaN. |
| `test_save_load_roundtrip` | JSON and pickle round-trips for both emission types, plus loading a v1 file. |

## 7. Implementation order (one PR each)

1. **Refactor only:** add `Emission`/`DiscreteEmission` and the `obs_prob`
   shim, route the HMM through them, and add the log-domain offset. All
   existing tests must still pass.
2. **Unify save/load:** `to_dict`/`from_dict`, the v2 format, and the v1 loader.
3. **Add the GMM:** `gaussian.py` and `GMMEmission` (diag covariance first,
   then full), plus the tests above.
4. **Add initialisation:** `init_gmm_hmm` using k-means.
5. **Tools and docs:** the sampler `HMM-GMM` subcommand, the HMM CLI,
   `hmm_plot` for continuous emissions (state-coloured scatter plots with
   ellipses), and a GMM-HMM section in `docs/algorithms/hmm.rst`.

## 8. Existing problems to fix along the way

- **`GaussianMixtureModel.update_m_step` (`cmla/models/gmm.py`) never resets
  `self.Sigma` before adding to it,** so the new covariance contains the old
  one. Don't reuse it for `GMMEmission`; the new `gaussian.py` helper can
  replace it later.
- **`cmla/scripts/hmm_cli.py` calls `HMM(num_states=..., num_observations=...)`
  and sets `transition_matrix`/`observation_matrix`.** None of those names exist
  on `HMM`, so `cmla-hmm` crashes on every path. Fix this in step 5.
- **`hmm_baum_welch` always writes checkpoints to `models/checkpoints/` and
  uses `print`.** Make the directory a parameter and use `logger`.
- **`hmm_plot.plot_checkpoint_dir` hard-codes 3 states and 4 symbol names,**
  so it fails on any other model size. Derive the layout from the model.
- ~~**The training loops `assert` that the log-likelihood never decreases.**~~
  Fixed in step 3: relative tolerance 1e-9. Converged GMM-HMM training showed
  decreases of about 1e-16 relative.

## Decisions

- **Covariance (2026-10-04):** the first version of `GMMEmission` supports
  **diagonal** covariance only. Full covariance can be added later as a
  `cov_type` switch in the same class without changing the interface.

## Progress

- [x] Step 1: `Emission` / `DiscreteEmission` refactor (`cmla/models/emission.py`)
- [x] Step 2: unified save/load (`HMM.to_dict`/`from_dict`, v2 format, v1 loader)
- [x] Step 3: `GMMEmission` (diagonal) and `gaussian.py`
- [ ] Step 4: initialisation
- [ ] Step 5: tools and docs
