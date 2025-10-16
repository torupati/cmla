Contributing
============

We welcome contributions to CMLA! This guide will help you get started.

Types of Contributions
----------------------

* Bug reports and fixes
* Feature requests and implementations
* Documentation improvements
* Tutorial additions
* Code optimizations

Development Setup
-----------------

1. **Fork and Clone**

   .. code-block:: bash

      git clone https://github.com/yourusername/cmla.git
      cd cmla

2. **Development Environment**

   .. code-block:: bash

      # Install development dependencies
      uv sync --group dev --group docs

      # Install pre-commit hooks
      uv run pre-commit install

3. **Running Tests**

   .. code-block:: bash

      # Run all tests
      uv run pytest

      # Run with coverage
      uv run pytest --cov=cmla

Code Standards
--------------

* **Formatting**: We use `ruff` for code formatting
* **Linting**: Pre-commit hooks ensure code quality
* **Type Hints**: Use type annotations where appropriate
* **Docstrings**: Follow NumPy docstring conventions

Submitting Changes
------------------

1. Create a feature branch
2. Make your changes
3. Add tests for new functionality
4. Update documentation if needed
5. Submit a pull request

Documentation
-------------

To build documentation locally:

.. code-block:: bash

   cd docs
   uv run make html

The documentation will be available in `_build/html/index.html`.

Questions?
----------

* Open an issue for bug reports or feature requests
* Start a discussion for questions about usage or design
