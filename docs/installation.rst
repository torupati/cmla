Installation
============

Requirements
------------

CMLA requires Python 3.13 or later and the following dependencies:

* numpy >= 2.3.4
* scipy >= 1.11.0
* matplotlib >= 3.7.0
* tqdm >= 4.66.0

Installation Methods
--------------------

From Source (Recommended)
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   # Clone the repository
   git clone https://github.com/torupati/cmla.git
   cd cmla

   # Install using uv (recommended)
   uv sync
   uv pip install -e .

   # Or using pip
   pip install -e .

Using uv (Modern Python Package Manager)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If you don't have uv installed:

.. code-block:: bash

   # Install uv
   curl -LsSf https://astral.sh/uv/install.sh | sh

   # Or using pip
   pip install uv

Development Installation
~~~~~~~~~~~~~~~~~~~~~~~~

For development with additional tools:

.. code-block:: bash

   # Install with development dependencies
   uv sync --group dev --group docs

   # Install pre-commit hooks
   uv run pre-commit install

Verification
------------

Verify the installation by running the tests:

.. code-block:: bash

   # Run all tests
   uv run pytest

   # Run specific algorithm tests
   uv run pytest tests/test_kmeans.py
   uv run pytest tests/test_gmm.py
   uv run pytest tests/test_hmm_*.py

Quick Test
~~~~~~~~~~

.. code-block:: python

   # Test basic functionality
   import numpy as np
   from cmla.models.kmeans import kmeans_clustering

   # Generate sample data
   data = np.random.randn(100, 2)
   centroids, labels = kmeans_clustering(data, k=3)
   print(f"Successfully clustered {len(data)} points into {len(centroids)} clusters")

Optional Dependencies
---------------------

For enhanced functionality:

.. code-block:: bash

   # For Jupyter notebook examples
   uv add jupyter

   # For advanced plotting
   uv add seaborn plotly

   # For data manipulation
   uv add pandas

Troubleshooting
---------------

Common Issues
~~~~~~~~~~~~~

**ImportError: No module named 'cmla'**

Ensure you've installed the package in development mode:

.. code-block:: bash

   uv pip install -e .

**ModuleNotFoundError for dependencies**

Reinstall dependencies:

.. code-block:: bash

   uv sync --reinstall

**Permission errors on Linux/macOS**

Use virtual environment (automatically handled by uv):

.. code-block:: bash

   # uv automatically creates and manages virtual environments
   uv sync

Platform-Specific Notes
~~~~~~~~~~~~~~~~~~~~~~~~

**Windows**

.. code-block:: bash

   # Use PowerShell or Command Prompt
   git clone https://github.com/torupati/cmla.git
   cd cmla
   uv sync

**macOS**

.. code-block:: bash

   # May need to install Xcode command line tools
   xcode-select --install

   # Then proceed with normal installation
   uv sync

**Linux**

.. code-block:: bash

   # Most distributions work out of the box
   uv sync

   # On some minimal systems, you might need:
   sudo apt-get update
   sudo apt-get install build-essential
