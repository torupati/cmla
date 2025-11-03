# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [v0.1.0-alpha] - 2025-11-03

### Added
- **Core Algorithms**
  - K-means clustering with multiple covariance matrix types (full, diagonal, spherical)
  - Gaussian Mixture Model (GMM) with EM algorithm implementation
  - Hidden Markov Model (HMM) with Forward-Backward, Viterbi, and Baum-Welch algorithms

- **Command-Line Interfaces**
  - `cmla-kmeans`: CLI for K-means clustering with data visualization
  - `cmla-hmm`: CLI for HMM training and inference

- **Utility Functions**
  - HMM parameter randomization utilities in `cmla.models.utils`
  - Model serialization support for both pickle and JSON formats

- **Documentation**
  - Comprehensive Sphinx documentation with mathematical derivations
  - Algorithm explanations with examples and usage patterns
  - GitHub Pages deployment with RTD theme

- **Testing & Quality**
  - Full test suite with pytest
  - Parameterized tests for different model configurations
  - Code formatting with ruff and quality checks

- **Project Infrastructure**
  - Modern Python packaging with uv and pyproject.toml
  - GitHub Actions CI/CD pipeline
  - Documentation auto-deployment

### Technical Details
- Python 3.13+ support
- Dependencies: NumPy, SciPy, matplotlib, tqdm
- Package structure: `cmla.models` for algorithms, `cmla.scripts` for CLI tools

### Notes
This is the initial alpha release providing core machine learning algorithm implementations for educational and research purposes. The API may change in future releases as we gather feedback and improve the design.
