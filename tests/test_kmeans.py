import numpy as np
import pytest

from cmla.models.kmeans import KmeansCluster
from cmla.models.kmeans import kmeans_clustering


class TestKmeansCluster:
    def test_kl_divergence(self):
        x = np.array([0.5, 0.3, 0.2])
        y = np.array([0.4, 0.4, 0.2])
        result = KmeansCluster.KL_divergence(x, y)
        assert isinstance(result, float)
        assert result >= 0

    def test_trainable(self):
        cluster = KmeansCluster(3, 2, trainable=False)
        with pytest.raises(RuntimeError, match="model is not set to training mode."):
            cluster.UpdateParameters()


def test_kmeans_constructor():
    """Test KmeansCluster constructor with various covariance_mode values"""

    # Valid covariance modes should work
    valid_modes = ["diag", "full", "none"]
    for mode in valid_modes:
        cluster = KmeansCluster(10, 10, covariance_mode=mode)
        assert cluster is not None
        # assert cluster.covariance_mode == mode

    # Invalid covariance mode should raise ValueError
    bad_covariance_mode = "foofoo"
    with pytest.raises(
        ValueError, match=f"covariance mode is wrong. got {bad_covariance_mode}"
    ):
        KmeansCluster(10, 10, covariance_mode=bad_covariance_mode)


def test_kmeans_constructor_parameters():
    """Test KmeansCluster constructor parameter validation"""

    # Valid parameters
    for mode in ["linear", "log", "kldiv"]:
        for cov_mode in ["diag", "full", "none"]:
            cluster = KmeansCluster(
                5, 3, trainable=False, distance_mode=mode, covariance_mode=cov_mode
            )
            assert cluster is not None
            assert cluster.distance_mode == mode
            assert cluster.covariance_mode == cov_mode
            assert not cluster.trainable
            assert cluster.num_clusters == 5
            assert cluster.feature_dimensionality == 3

    # Test different parameter combinations
    with pytest.raises(ValueError):
        KmeansCluster(3, 2, distance_mode="invalid_mode")  # Invalid distance_mode

    # Invalid parameters (if applicable)
    with pytest.raises(ValueError):
        KmeansCluster(0, 2)  # K should be > 0

    with pytest.raises(ValueError):
        KmeansCluster(3, 0)  # D should be > 0


def test_kmeans_constructor_default_values():
    """Test that constructor sets correct default values"""
    cluster = KmeansCluster(3, 2)
    assert cluster is not None
    assert cluster.trainable  # default value is True
    assert cluster.distance_mode == "linear"  # default value
    assert cluster.covariance_mode == "diag"  # default value


def test_kmeans_fit_predict():
    np.random.seed(0)
    n_samples = 300
    n_features = 5
    n_clusters = 3
    X = np.random.rand(n_samples, n_features)
    print(f"{X.shape=}")

    mu_init = np.random.rand(n_clusters, n_features)
    kmeans, cost_history = kmeans_clustering(X, mu_init)
    for i, c in enumerate(cost_history):
        print(f"itr={i} cost={c}")
        assert isinstance(c, float)
        assert c >= 0
        if i > 1:
            assert cost_history[i] <= cost_history[i - 1]
    labels = kmeans.predict(X)
    assert labels.shape == (n_samples,)
    assert set(labels) <= set(range(n_clusters))
