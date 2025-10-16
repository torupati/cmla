# K-means clustering in-house implementation.
# coding by Python and Numpy.

import logging
from pathlib import Path

import numpy as np

from cmla.plots.kmeans_plot import plot_data_with_centroid

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class KmeansCluster:
    """
    Definition of K-means clustering model.
    Note that number of clusters is fixed after instance creation.
    Distance metric can be selected from linear scale, log scale and
    KL divergence.
    Covariance can be set to none, diag or full. If covariance is none,
    only mean vectors are updated.
    Training variables can be set to outside or inside. If inside is
    selected, training variables are created inside the instance and
    updated by PushSample() method. If outside is selected, training
    variables are not created and PushSample() method is not available.
    In this case, get_alignment() method is used to get sample alignment
    to clusters.
    """

    COV_NONE = 0
    COV_FULL = 1
    COV_DIAG = 2

    DISTANCE_LINEAR_SCALE = 0
    DISTANCE_LOG_SCALE = 1
    DISTANCE_KL_DIVERGENCE = 2

    def __init__(
        self,
        num_clusters: int,
        D: int,
        trainable: bool = True,
        covariance_mode: str = "diag",
        distance_mode: str = "linear",
    ):
        """Initialize instance.

        Args:
            num_clusters (int): number of clusters. Fixed.
            D (int): dimension of smaples. Fixed.
            trainable (bool): if True, training variables are created inside the instance and
                updated by PushSample() method. If False, training variables are not created
                and PushSample() method is not available.
            covariance_mode(str): "none"(default), "diag" or "full" (optional)
            distance_mode (str): distance mode. "linear"(default), "log", "kldiv"

        Raises:
            ValueError: wrong distance mode input
            ValueError: wrong covariance mode input
        Note:
        If you want to change the number of clusters, new instance with desired
        cluster numbers should be created.
        """
        if num_clusters <= 0:
            raise ValueError(f"num_clusters must be > 0. got {num_clusters}")
        if D <= 0:
            raise ValueError(f"D must be > 0. got {D}")
        self.Mu = np.random.randn(num_clusters, D)  # centroid
        if covariance_mode == "diag":
            self._cov_mode = KmeansCluster.COV_DIAG
            self.Sigma = np.ones((num_clusters, D))
        elif covariance_mode == "full":
            self._cov_mode = KmeansCluster.COV_FULL
            self.Sigma = np.ones((num_clusters, D, D))
        elif covariance_mode == "none":
            self._cov_mode = KmeansCluster.COV_NONE
            self.Sigma = None
        else:
            raise ValueError(f"covariance mode is wrong. got {covariance_mode}")

        # define training variables
        self._trainable = trainable
        if self._trainable:
            self._loss = 0.0
            self._X0 = np.zeros([num_clusters], dtype=np.uint32)
            self._X1 = np.zeros([num_clusters, D])
            if self._cov_mode == KmeansCluster.COV_NONE:
                self._X2 = None
            elif self._cov_mode == KmeansCluster.COV_FULL:
                self._X2 = np.zeros([num_clusters, D, D])
            elif self._cov_mode == KmeansCluster.COV_DIAG:
                self._X2 = np.zeros([num_clusters, D])
        else:
            self._X0 = None
            self._X1 = None
            self._X2 = None
            self._loss = None

        if distance_mode == "linear":
            self._dist_mode = KmeansCluster.DISTANCE_LINEAR_SCALE
        elif distance_mode == "log":
            self._dist_mode = KmeansCluster.DISTANCE_LOG_SCALE
        elif distance_mode == "kldiv":
            self._dist_mode = KmeansCluster.DISTANCE_KL_DIVERGENCE
        else:
            raise ValueError(f"distance mode is wrong. got {distance_mode}")

    @property
    def num_clusters(self) -> int:
        """Number of clusters

        Returns:
            int: cluster count
        """
        return self.Mu.shape[0]

    @property
    def feature_dimensionality(self) -> int:
        """Dimension of input data

        Returns:
            int: dimension of input data
        """
        return self.Mu.shape[1]

    @property
    def distance_mode(self) -> str:
        name_list = {
            self.DISTANCE_LINEAR_SCALE: "linear",
            self.DISTANCE_LOG_SCALE: "log",
            self.DISTANCE_KL_DIVERGENCE: "kldiv",
        }
        return name_list[self._dist_mode]

    def __repr__(self):
        return (
            "KmeansClustering {n} num_cluster:{k} feature dimensionality:{d}".format(
                n=self.__class__.__name__,
                k=self.num_clusters,
                d=self.feature_dimensionality,
            )
            + " distance={d2}".format(d2=self.distance_mode)
            + " covariance={c}".format(c=self.covariance_mode)
            + " trainable={tr}".format(tr=self.trainable)
        )

    @property
    def covariance_mode(self):
        name_list = {
            self.COV_DIAG: "diag",
            self.COV_FULL: "full",
            self.COV_NONE: "none",
        }
        return name_list[self._cov_mode]

    @property
    def trainable(self) -> bool:
        return self._trainable

    def distortion_measure(self, x: np.ndarray, r: np.ndarray) -> float:
        """Calculate distortion measure.

        Args:
            x (ndarray): input samples (N,D)
            r (ndarray): sample alignment to clusters (N,num_clusters)

        Returns:
            float: disotrion (average per sample). If N = 0, 0.0 is returned.
        """
        J = 0.0
        n_sample = x.shape[0]
        if n_sample == 0:
            return 0.0
        for n in range(n_sample):
            dist = None
            if self._dist_mode == KmeansCluster.DISTANCE_LINEAR_SCALE:
                dist = [
                    sum(v * v for v in x[n, :] - self.Mu[k, :])
                    for k in range(self.num_clusters)
                ]
            elif self._dist_mode == KmeansCluster.DISTANCE_LOG_SCALE:
                dist = [
                    sum(v * v for v in np.log(x[n, :]) - np.log(self.Mu[k, :]))
                    for k in range(self.num_clusters)
                ]
            elif self._dist_mode == KmeansCluster.DISTANCE_KL_DIVERGENCE:
                dist = [
                    KmeansCluster.KL_divergence(x[n, :], self.Mu[k, :])
                    for k in range(self.num_clusters)
                ]
            if dist is not None:
                J = J + np.dot(r[n, :], dist)
        return J / n_sample

    def get_alignment(self, x: np.ndarray) -> np.ndarray:
        """
        Hard allocation of each sample to a cluster (or Gaussians)

        Args:
            x (ndarray): input samples (N,D)
        Returns:
            r (ndarray): sample alignment to clusters (N,K)
        """
        if len(x.shape) != 2:
            raise ValueError(f"input shape is wrong {x.shape=}")
        N = x.shape[0]
        r = np.zeros((N, self.num_clusters), dtype=np.uint16)
        for n in range(N):
            costs = None
            if self._dist_mode == KmeansCluster.DISTANCE_LINEAR_SCALE:
                costs = [
                    sum([v * v for v in (x[n, :] - self.Mu[k, :])])
                    for k in range(self.num_clusters)
                ]
            elif self._dist_mode == KmeansCluster.DISTANCE_LOG_SCALE:
                costs = [
                    sum([v * v for v in (np.log(x[n, :]) - np.log(self.Mu[k, :]))])
                    for k in range(self.num_clusters)
                ]
            elif self._dist_mode == KmeansCluster.DISTANCE_KL_DIVERGENCE:
                costs = [
                    KmeansCluster.KL_divergence(x[n, :], self.Mu[k, :])
                    for k in range(self.num_clusters)
                ]
            if costs is not None:
                r[n, np.argmin(costs)] = 1
            # r[n, np.argmin([ sum(v*v for v in x[n, :] - self.Mu[k, :]) for k in range(self.num_clusters)])] = 1
            r[n, :] = r[n, :] / r[n, :].sum()
            # wk = [(x[n, 0] - mu[k, 0])**2 + (x[n, 1] - mu[k, 1])**2 for k in range(K)]
            # r[n, argmin(wk)] = 1
        return r

    def predict(self, x: np.ndarray) -> np.ndarray:
        """
        Predict the cluster index for each sample.

        Args:
            x (ndarray): input samples (N,D)
        Returns:
            r (ndarray): sample alignment to clusters (N,K)
        """
        r = self.get_alignment(x)
        labels = np.argmax(r, axis=1)
        return labels

    def PushSample(self, x: np.ndarray) -> tuple[int, float]:
        """Push one training sample to inner training variables

        Args:
            x (ndarray): traiing sample (D,)

        Returns:
            int: aligned cluster's index (between 0 and K-1)
            float: loss of this sample
        """
        if self._trainable is False:
            raise RuntimeError("model is not set to training mode.")
        if self._dist_mode == KmeansCluster.DISTANCE_LINEAR_SCALE:
            costs = [
                sum([v * v for v in (x - self.Mu[k, :])])
                for k in range(self.num_clusters)
            ]
        elif self._dist_mode == KmeansCluster.DISTANCE_LOG_SCALE:
            costs = [
                sum([v * v for v in (np.log(x) - np.log(self.Mu[k, :]))])
                for k in range(self.num_clusters)
            ]
        elif self._dist_mode == KmeansCluster.DISTANCE_KL_DIVERGENCE:
            costs = [
                KmeansCluster.KL_divergence(x, self.Mu[k, :])
                for k in range(self.num_clusters)
            ]
        else:
            raise ValueError("wrong distance model")
        if np.isinf(costs).any():
            if self._dist_mode == KmeansCluster.DISTANCE_LOG_SCALE:
                raise ValueError(
                    f"log(x)={np.log(x)}" + f" log(mu)={np.log(self.Mu)} costs={costs}"
                )
            raise RuntimeError(
                f"wrong input in distance computation x={x} mu={self.Mu}"
                + f"costs={costs}"
            )

        # convert to float (python scaler) from numpy array of numpy scalar
        costs = costs.item() if isinstance(costs, np.ndarray) else costs
        if not isinstance(costs, list):
            raise ValueError(
                f"wrong input in distance computation x={x} mu={self.Mu}"
                + f"costs={costs}"
            )
        k_min = np.argmin(costs)
        self._loss += costs[k_min].item()
        self._X0[k_min] += 1
        self._X1[k_min, :] += x
        if self._cov_mode == KmeansCluster.COV_DIAG:
            self._X2[k_min, :] = x * x
        elif self._cov_mode == KmeansCluster.COV_FULL:
            # NOT checked.
            self._X2[k_min, :, :] = x.reshape(
                self.feature_dimensionality, 1
            ) * x.reshape(1, self.feature_dimensionality)
        return k_min, costs[k_min]

    def ClearTrainingVariables(self):
        """Reset inside statistics for training

        Raises:
            RuntimeError: invalid training setting
        """
        if self._trainable is False:
            raise RuntimeError("model is not set to training mode.")
        self._loss = 0.0
        self._X0 = np.zeros([self.num_clusters])
        self._X1 = np.zeros([self.num_clusters, self.feature_dimensionality])
        if self._cov_mode == KmeansCluster.COV_NONE:
            self._X2 = None
        elif self._cov_mode == KmeansCluster.COV_FULL:
            self._X2 = np.zeros(
                [
                    self.num_clusters,
                    self.feature_dimensionality,
                    self.feature_dimensionality,
                ]
            )
        elif self._cov_mode == KmeansCluster.COV_DIAG:
            self._X2 = np.zeros([self.num_clusters, self.feature_dimensionality])

    def UpdateParameters(self) -> tuple[float, list]:
        """Update model parameters from inside training variables.
        Note that this method does not clear training variables.

        Raises:
            RuntimeError: invalid training setting

        Returns:
            float: total loss in current training variables
            list: number of samples aligned to each cluster
        """
        if self._trainable is False:
            raise RuntimeError("model is not set to training mode.")
        for k in range(self.num_clusters):
            if self._X0[k] == 0:
                continue
            self.Mu[k, :] = self._X1[k, :] / self._X0[k]
            if self._cov_mode in [KmeansCluster.COV_FULL, KmeansCluster.COV_DIAG]:
                self.Sigma = self._X2 / self._X0[k]
            if self._dist_mode == KmeansCluster.DISTANCE_KL_DIVERGENCE:
                self.Mu[k, :] = self.Mu[k, :] / np.sum(self.Mu[k, :])
        return self._loss, self._X0

    @property
    def loss(self) -> float:
        """total loss in current training variables
        Returns:
            float: total loss
        """
        return self._loss

    @staticmethod
    def KL_divergence(x: np.ndarray, y: np.ndarray, **kwargs) -> float:
        """Calculate KL divergence beteen two vectors.

        Args:
            x (np.ndarray): probability vector
            y (np.ndarray): probability vector
        Note:
            input vectors must be non negative.

        Returns:
            float: KL divergence (not in log scale)
        """
        _x = x / np.sum(x)
        _y = y / np.sum(y)
        xy_diff = np.log(_x) - np.log(_y)
        _kl_div = np.sum(_x * xy_diff)
        return _kl_div


def kmeans_clustering(X: np.ndarray, mu_init: np.ndarray, **kwargs):
    """Run k-means clustering.

    Args:
        X (np.ndarray): vector samples (N, D)
        mu_init (np.ndarray): initial mean vectors(K, D)
        max_it (int, optional): Iteration steps. Defaults to 20.

        kwargs:
            dist_mode (str): distance mode. "linear"(default), "log", "kldiv"
            plot_ckpt (bool): if True, plot intermediate result at each iteration step
            max_it (int): maximum iteration steps. Default 20.

    Returns:
        _type_: _description_
    """
    K, feature_dim = mu_init.shape
    N, feature_dim2 = X.shape
    if feature_dim != feature_dim2:
        raise ValueError(
            f"Wrong dim. X(N,D={feature_dim2}), mu_init(M, D={feature_dim})"
        )

    max_it = kwargs.get("max_it", 20)
    # save_ckpt = kwargs.get('save_ckpt', False)
    distance_mode = kwargs.get("distance_mode", "linear")

    kmeansparam = KmeansCluster(
        K, feature_dim, trainable=True, distance_mode=distance_mode
    )
    logger.info("initialize: %s", str(kmeansparam))
    kmeansparam.Mu = mu_init
    cost_history = []
    align_history = []
    for it in range(max_it):
        kmeansparam.ClearTrainingVariables()
        # Push all samples (align samples to clusters)
        for n in range(N):
            _, _ = kmeansparam.PushSample(X[n, :])

        # Update parameters(centroids)
        loss, align_dist = kmeansparam.UpdateParameters()

        logger.info("iteration %d loss %f", it, loss / N)
        cost_history.append(loss)
        align_history.append(align_dist)

        # Convergence validation
        if len(cost_history) > 1:
            cost_diff = cost_history[-2] - cost_history[-1]
            align_diff = [x - y for x, y in zip(align_history[-1], align_history[-2])]
            assert cost_diff >= 0.0
            logger.debug(
                "iteration step=%d cost_diff = %f" + "alignment change=%s",
                it,
                cost_diff,
                align_diff,
            )
            if 0.0 <= cost_diff < 1.0e-6 and np.sum(align_diff) < 1.0e-6:
                logger.info("converged at iteration %d, alignment not changed", it)
                break
        if kwargs.get("plot_ckpt", False):
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(1, 1, figsize=(6, 6))
            r = kmeansparam.get_alignment(X)
            plot_data_with_centroid(
                ax=ax, x=X, r=r, mu=kmeansparam.Mu, kmeans_param_ref={}
            )
            fig.suptitle(
                "K-means iteration {it} loss={loss:.2f}".format(it=it, loss=loss / N)
            )
            fig.savefig(f"kmeans_iter{it:03d}.png")
            plt.close(fig)

    return kmeansparam, cost_history


def pickle_kmeans_and_data_by_dict(
    out_file: Path, kmeans_param_dict: dict, X: np.ndarray
):
    """
    Serializes and saves KMeans clustering parameters and data to a file.

    Args:
        out_file (Path): The path to the output file where the serialized data will be saved.
        kmeans_param_dict: dict: A dictionary containing KMeans clustering parameters.
            Expected keys in the dictionary:
                - Mu: The cluster centroids.
                - Sigma: The cluster covariances.
                - Pi: The cluster weights.
                - covariance_mode: The type of covariance used in the model.
                - train_vars_mode: The training variables mode.
                - DistanceType: The distance metric used.
        X (np.ndarray): The data samples to be saved along with the model parameters.

    Returns:
        None: This function does not return a value. It writes the serialized data to the specified file.

    Raises:
        IOError: If there is an issue writing to the specified file.
    """
    import pickle

    with open(out_file, "wb") as f:
        pickle.dump(
            {
                "model_param": kmeans_param_dict,
                "sample": X,
                "model_type": "KmeansClustering",
            },
            f,
        )
