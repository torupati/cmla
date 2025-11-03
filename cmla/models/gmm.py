# EM algorithm implementation to train Gaussian Mixture Models.
# Reference: "Pattern Recognition and Machine Learning" by C. M. Bishop, Chapter 9

import logging

import numpy as np
from numpy import array, dot, ndarray, sum
from scipy.stats import multivariate_normal
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

from cmla.plots.kmeans_plot import plot_data_with_centroid

logger = logging.getLogger(__name__)


class GaussianMixtureModel:
    """Definition of Gaussian Mixture Model (GMM)

    Update with given training data by EM algorithm is implemented.
    """

    def __init__(self, num_components: int, feature_dimension: int):
        """
        Each Gaussian is initialized as zero means, unit covariance.
        Weight is initailized as equal probability.
        Args:
            num_components (int): number of mixtures
            feature_dimension (int): dimension of smaples
        """
        if num_components < 1:
            raise ValueError(f"num_components must be > 0. got {num_components}")
        if feature_dimension < 1:
            raise ValueError(f"feature_dimension must be > 0. got {feature_dimension}")
        self.Mu = np.random.randn(num_components, feature_dimension)
        self.Sigma = np.zeros((num_components, feature_dimension, feature_dimension))
        for m in range(self.Mu.shape[0]):
            self.Sigma[m, :, :] = np.eye(feature_dimension)
        self.Pi = (
            np.ones(num_components) / num_components
        )  # equal probability for initial condition

    @property
    def num_components(self) -> int:
        """Number of Gaussians. Read-only.

        Returns:
            int: number of Gaussians
        """
        return self.Mu.shape[0]

    @property
    def feature_dimension(self) -> int:
        """Dimension of input data. Read-only.

        Returns:
            int: dimension of input data
        """
        return self.Mu.shape[1]

    def __repr__(self):
        return "{n} (M={k} D={d})".format(
            n=self.__class__.__name__, k=self.num_components, d=self.feature_dimension
        )

    def probability(self, x: ndarray) -> ndarray:
        """Calculate probability of this GMM at given sample points

        Args:
            x (ndarray): random variable of GMM

        Returns:
            ndarray: probability density of GMM. Note this is NOT in log scale.
        """
        return sum(
            self.Pi[k] * multivariate_normal(self.Mu[k, :], self.Sigma[k, :, :]).pdf(x)
            for k in range(self.num_components)
        )

    def log_likelihood(self, x: np.ndarray) -> float:
        """Calculate log-likelihood of vectors.

        Args:
            x (np.ndarray): D-dimesional vectors. Its number is N. (N, D)

        Returns:
            float: Total log-likelihood
        """
        num_obs = x.shape[0]
        y = np.zeros(
            (num_obs, self.num_components)
        )  # keep all Gaussian pdf (not smart)
        for k in range(self.num_components):
            y[:, k] = multivariate_normal(self.Mu[k, :], self.Sigma[k, :, :]).pdf(
                x
            )  # (K,N)
        lh = 0
        for n in range(num_obs):
            wk = 0
            for k in range(self.num_components):
                wk = wk + self.Pi[k] * y[n, k]
            lh = lh + np.log(wk)
        return lh

    def update_e_step(self, x: ndarray) -> tuple[np.ndarray, float]:
        """Calculate gamma of GMM (Expectation step of GMM training)

        Args:
            x(N,D), trainig samples (n-th sample, d dimensional vector)

        Returns:
            gam(np.ndarray): probability of x(n) in k-th Gaussian, P(k|x[n]), shape=(N,K)
            llh(float): log-likelihood
        """
        num_obs = x.shape[0]
        # caluclate P(x[n], k) = P(k)P(x[n]|k) and hold all as array (n,m)
        try:
            gam = array(
                [
                    self.Pi[m]
                    * multivariate_normal(
                        self.Mu[m, :], self.Sigma[m, :, :], allow_singular=True
                    ).pdf(x)
                    for m in range(self.num_components)
                ]
            ).transpose()  # (N,M)
        except np.linalg.LinAlgError as e:
            print(self.Pi)
            print(self.Sigma)
            raise e
        llh = 0.0  # log likelihood of all training data
        for n in range(num_obs):
            _s = sum(gam[n, :])  # P(x[n]) = sum_k P(k,x[n])
            gam[n, :] = gam[n, :] / _s
            llh += np.log(_s)
        return gam, llh

    def update_m_step(self, x: ndarray, gamma: ndarray) -> bool:
        """Update parameter of GMM with given allocation.

        Args:
            X(N,D), training samples. (N is number of samples, D is dimension)
        """
        num_obs = x.shape[0]
        self.Pi = sum(gamma, axis=0) / sum(gamma)
        self.Mu = dot(gamma.transpose(), x)
        for m in range(self.num_components):
            if sum(gamma[:, m]) < 1.0e-10:
                continue
            self.Mu[m, :] = self.Mu[m, :] / sum(gamma[:, m])
            # self.Sigma = np.zeros((M, D, D))
            # for m in range(self._M):
            #    if sum(gamma[:,m]) < 1.0E-5:
            #        continue
            for n in range(num_obs):
                wk = x - self.Mu[m, :]  # distance between x and m-th Gaussian's mean
                wk = wk[n, :, np.newaxis]
                self.Sigma[m, :, :] = self.Sigma[m, :, :] + gamma[n, m] * np.dot(
                    wk, wk.T
                )
            # print(self.Sigma[m,:,:])
            tmp_sig = self.Sigma[m, :, :] / np.sum(gamma[:, m])
            if not (tmp_sig > 1e4).any():
                self.Sigma[m, :, :] = tmp_sig
            else:
                print("too large: self.Sigma=", self.Sigma)
                print("gamma = ", np.sum(gamma[:, m]), " covariance is not updated.")
                # input()
        return True


def train_gmm(gmm: GaussianMixtureModel, X: np.ndarray, max_it: int = 12, **kwargs):
    """Train GMM

    Args:
        gmm (GaussianMixtureModel): initial GMM
        X (np.ndarray): training data
        max_it (int, optional): number of iteration. Defaults to 12.

    Returns:
        _type_: _description_
    """
    N = X.shape[0]
    loglikelihood_history = []  # distortion measure
    with logging_redirect_tqdm(loggers=[logger]):
        pbar = tqdm(
            range(max_it),
            desc=f"gmm-train(M={gmm.num_components})",
            postfix="postfix",
            ncols=80,
        )
        for it in pbar:
            # for it in range(0, max_it):
            #        _ll = gmm.log_likelihood(X)
            _gamma, _ll = gmm.update_e_step(X)
            loglikelihood_history.append(_ll)
            gmm.update_m_step(X, _gamma)
            # pbar.write('GMM EM training: step={_i} E[log(P(X)]={_l}'.format(_i=len(loglikelihood_history), _l=_ll/N))
            logger.info(
                "GMM EM training: step={_i} E[log(P(X)]={_l}".format(
                    _i=len(loglikelihood_history), _l=_ll / N
                )
            )
            if kwargs.get("plot_ckpt", False) and gmm.feature_dimension >= 2:
                import matplotlib.pyplot as plt

                fig, ax = plt.subplots(1, 1, figsize=(6, 6))
                plot_data_with_centroid(ax, X, _gamma, gmm.Mu)
                out_pngfile = "gmm-step{it:03d}.png".format(
                    it=len(loglikelihood_history)
                )
                fig.suptitle(
                    "GMM EM training: step={_i} E[log(P(X)]={_l:.2f}".format(
                        _i=len(loglikelihood_history), _l=_ll / N
                    )
                )
                fig.savefig(out_pngfile)
                logger.info("save PNG file: %s", out_pngfile)
    return gmm, loglikelihood_history


def gmm_em_training_mixutre_scan(X, max_mixnum: int = 10):
    N, D = X.shape[0], X.shape[1]

    mixnum_likelihood = {}
    L = np.linalg.cholesky(np.cov(X.T))
    for mix_num in range(2, max_mixnum):
        np.random.seed(0)
        gmm = GaussianMixtureModel(mix_num, D)
        gmm.Sigma = np.zeros([mix_num, D, D])
        gmm.Mu = np.zeros([mix_num, D])
        for m in range(mix_num):
            gmm.Mu[m, :] = X.mean() + np.dot(L, np.random.randn(D))
            gmm.Sigma[m, :, :] = np.eye(D)
        gmm, loglikelihood_history = train_gmm(gmm, X, max_it=100)
        logger.info(
            "GMM EM training: mixture={i} E[log(P(X)]={_l}".format(
                i=mix_num, _l=loglikelihood_history[-1] / N
            )
        )
        mixnum_likelihood[mix_num] = loglikelihood_history[-1]
    print(mixnum_likelihood)
    return mixnum_likelihood


# ---


def plot_loglikelihood_history(ax, loglikelihood_history):
    ax.plot(
        range(0, len(loglikelihood_history)),
        loglikelihood_history,
        color="k",
        linestyle="-",
        marker="o",
    )
    ax.set_xlim([0, len(loglikelihood_history)])
    ax.set_ylabel("log liklihood")
    ax.set_xlabel("iteration step")
    # plt.ylim([40, 80])
    ax.grid(True)
    return ax
