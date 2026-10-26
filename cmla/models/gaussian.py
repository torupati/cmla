# Vectorised Gaussian density helpers shared by mixture models.

import numpy as np

LOG_2PI = np.log(2.0 * np.pi)


def diag_gaussian_log_pdf(
    x: np.ndarray, means: np.ndarray, variances: np.ndarray
) -> np.ndarray:
    """Log density of diagonal-covariance Gaussians for all samples and components.

    log N(x; mu, diag(var)) = -1/2 * (D log 2pi + sum_d log var_d + sum_d (x_d - mu_d)^2 / var_d)

    Args:
        x (np.ndarray): samples, shape (T, D)
        means (np.ndarray): component means, shape (..., D)
        variances (np.ndarray): component variances (diagonal of covariance), shape (..., D)

    Returns:
        np.ndarray: log densities, shape (T, ...)
    """
    x = np.atleast_2d(np.asarray(x, dtype=float))
    T, D = x.shape
    if means.shape[-1] != D or variances.shape != means.shape:
        raise ValueError(
            f"shape mismatch: x {x.shape}, means {means.shape}, variances {variances.shape}"
        )
    lead_shape = means.shape[:-1]
    mu = means.reshape(-1, D)  # (C, D)
    var = variances.reshape(-1, D)
    diff = x[:, np.newaxis, :] - mu[np.newaxis, :, :]  # (T, C, D)
    mahalanobis = np.sum(diff**2 / var, axis=2)  # (T, C)
    log_det = np.sum(np.log(var), axis=1)  # (C,)
    log_pdf = -0.5 * (D * LOG_2PI + log_det + mahalanobis)
    return log_pdf.reshape((T,) + lead_shape)
