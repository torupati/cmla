import numpy as np
import pytest

from cmla.models.gmm import GaussianMixtureModel, train_gmm


class TestGaussianMixtureModel:
    def test_constructor(self):
        """Test GaussianMixtureModel constructor"""
        M, D = 3, 2
        gmm = GaussianMixtureModel(num_components=M, feature_dimension=D)

        assert gmm.num_components == M
        assert gmm.feature_dimension == D
        assert gmm.Pi.shape == (M,)
        assert gmm.Mu.shape == (M, D)
        assert gmm.Sigma.shape == (M, D, D)

        # Check that Pi sums to 1
        assert np.allclose(gmm.Pi.sum(), 1.0)

        # Check that all Pi values are positive
        assert np.all(gmm.Pi > 0)

    def test_constructor_invalid_parameters(self):
        """Test constructor with invalid parameters"""
        with pytest.raises(ValueError):
            GaussianMixtureModel(num_components=0, feature_dimension=2)

        with pytest.raises(ValueError):
            GaussianMixtureModel(num_components=3, feature_dimension=0)

    def test_update_e_step(self):
        """Test E-step (expectation step)"""
        np.random.seed(42)
        M, D = 2, 3
        N = 100

        gmm = GaussianMixtureModel(num_components=M, feature_dimension=D)
        X = np.random.randn(N, D)

        gamma, log_likelihood = gmm.update_e_step(X)

        # Check gamma shape and properties
        assert gamma.shape == (N, M)
        assert np.allclose(gamma.sum(axis=1), 1.0)  # Each row sums to 1
        assert np.all(gamma >= 0)  # All probabilities are non-negative
        assert np.all(gamma <= 1)  # All probabilities are <= 1

        # Check log_likelihood is a scalar
        assert isinstance(log_likelihood, (float, np.float64))

    def test_update_m_step(self):
        """Test M-step (maximization step)"""
        np.random.seed(42)
        M, D = 2, 3
        N = 100

        gmm = GaussianMixtureModel(num_components=M, feature_dimension=D)
        X = np.random.randn(N, D)

        # Create dummy gamma (responsibilities)
        gamma = np.random.rand(N, M)
        gamma /= gamma.sum(axis=1, keepdims=True)  # Normalize rows to sum to 1

        # Store original parameters
        original_pi = gmm.Pi.copy()
        original_mu = gmm.Mu.copy()
        # original_sigma = gmm.Sigma.copy()

        # Run M-step
        gmm.update_m_step(X, gamma)

        # Check that parameters have been updated
        assert not np.allclose(gmm.Pi, original_pi)
        assert not np.allclose(gmm.Mu, original_mu)
        # assert not np.allclose(gmm.Sigma, original_sigma)

        # Check Pi properties
        assert np.allclose(gmm.Pi.sum(), 1.0)
        assert np.all(gmm.Pi > 0)

        # Check shapes remain the same
        assert gmm.Pi.shape == (M,)
        assert gmm.Mu.shape == (M, D)
        assert gmm.Sigma.shape == (M, D, D)

    def test_covariance_matrices_positive_definite(self):
        """Test that covariance matrices remain positive definite"""
        np.random.seed(42)
        M, D = 2, 3
        N = 100

        gmm = GaussianMixtureModel(num_components=M, feature_dimension=D)
        X = np.random.randn(N, D)
        gamma = np.random.rand(N, M)
        gamma /= gamma.sum(axis=1, keepdims=True)

        gmm.update_m_step(X, gamma)

        # Check that all covariance matrices are positive definite
        for m in range(M):
            eigenvalues = np.linalg.eigvals(gmm.Sigma[m])
            assert np.all(eigenvalues > 0), (
                f"Covariance matrix {m} is not positive definite"
            )

    @pytest.mark.parametrize("M,D", [(2, 1), (3, 2), (4, 3)])
    def test_different_dimensions(self, M, D):
        """Test GMM with different numbers of mixtures and dimensions"""
        gmm = GaussianMixtureModel(num_components=M, feature_dimension=D)
        assert gmm.num_components == M
        assert gmm.feature_dimension == D

        # Test with random data
        np.random.seed(42)
        N = 50
        X = np.random.randn(N, D)

        gamma, log_likelihood = gmm.update_e_step(X)
        assert gamma.shape == (N, M)
        assert isinstance(log_likelihood, (float, np.float64))


class TestTrainGMM:
    def test_train_gmm_basic(self):
        """Test basic GMM training"""
        np.random.seed(42)
        M, D = 2, 2
        N = 100

        # Create simple 2D data with two clusters
        cluster1 = np.random.multivariate_normal([2, 2], [[1, 0], [0, 1]], N // 2)
        cluster2 = np.random.multivariate_normal([-2, -2], [[1, 0], [0, 1]], N // 2)
        X = np.vstack([cluster1, cluster2])

        # Initialize GMM
        gmm = GaussianMixtureModel(num_components=M, feature_dimension=D)

        # Train
        trained_gmm, log_likelihood_history = train_gmm(
            gmm, X, max_it=10, plot_ckpt=False
        )

        # Check that training completed
        assert trained_gmm is not None
        assert len(log_likelihood_history) <= 10  # Should not exceed max iterations

        # Check that log likelihood increased (or at least didn't decrease much)
        if len(log_likelihood_history) > 1:
            # Allow small numerical decreases due to floating point precision
            for i in range(1, len(log_likelihood_history)):
                assert log_likelihood_history[i] >= log_likelihood_history[i - 1] - 1e-6

    def test_train_gmm_convergence(self):
        """Test that GMM training converges"""
        np.random.seed(42)
        M, D = 2, 2
        N = 100

        X = np.random.randn(N, D)
        gmm = GaussianMixtureModel(num_components=M, feature_dimension=D)

        _, log_likelihood_history = train_gmm(
            gmm, X, max_it=50, tolerance=1e-6, plot_ckpt=False
        )

        assert len(log_likelihood_history) > 0
        assert all(isinstance(ll, (float, np.float64)) for ll in log_likelihood_history)

    def test_train_gmm_parameters_unchanged_reference(self):
        """Test that original GMM object is modified (not copied)"""
        np.random.seed(42)
        M, D = 2, 2
        N = 50

        X = np.random.randn(N, D)
        gmm = GaussianMixtureModel(num_components=M, feature_dimension=D)

        original_pi = gmm.Pi.copy()

        trained_gmm, _ = train_gmm(gmm, X, max_it=5, plot_ckpt=False)

        # The returned GMM should be the same object
        assert trained_gmm is gmm

        # Parameters should have been updated
        assert not np.allclose(gmm.Pi, original_pi)

    def test_train_gmm_with_small_dataset(self):
        """Test GMM training with small dataset"""
        np.random.seed(42)
        M, D = 2, 2
        N = 10  # Very small dataset

        X = np.random.randn(N, D)
        gmm = GaussianMixtureModel(num_components=M, feature_dimension=D)

        # Should not crash with small dataset
        trained_gmm, log_likelihood_history = train_gmm(
            gmm, X, max_it=5, plot_ckpt=False
        )

        assert trained_gmm is not None
        assert len(log_likelihood_history) > 0

    @pytest.mark.parametrize("max_it", [1, 5, 10])
    def test_train_gmm_max_iterations(self, max_it):
        """Test that training respects maximum iterations"""
        np.random.seed(42)
        M, D = 2, 2
        N = 50

        X = np.random.randn(N, D)
        gmm = GaussianMixtureModel(num_components=M, feature_dimension=D)

        _, log_likelihood_history = train_gmm(gmm, X, max_it=max_it, plot_ckpt=False)

        assert len(log_likelihood_history) <= max_it
