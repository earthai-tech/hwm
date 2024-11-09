
.. _installation_guide: 

========================
Installation Guide
========================

The `hwm` package offers advanced tools for dynamic system 
modeling, nonlinear regression, and data analysis. Follow 
these steps to install and start using `hwm`.

Prerequisites
---------------
The following software and packages are required before 
installing `hwm`:

- **Python 3.9+**: Ensure Python 3.9 or later is 
  installed on your system.
- **pip**: The package installer for Python. Keep it 
  up-to-date for smoother installation.
- **virtualenv** (recommended): A virtual environment 
  helps isolate dependencies and prevents conflicts 
  with other packages.

Quick Installation
--------------------
To install `hwm` from PyPI, run the following command:

.. code-block:: bash

    pip install hwm

This command installs `hwm` along with its required 
dependencies, making it ready for immediate use.

Installation from Source
--------------------------
For the latest features and updates, you can install 
`hwm` directly from the source code repository.

1. **Clone the repository**:

   .. code-block:: bash

       git clone https://github.com/earthai-tech/hwm.git
       cd hwm

2. **Install the package**:

   .. code-block:: bash

       pip install .

3. **Verify installation**:

   After installation, verify it by running:

   .. code-block:: python

       import hwm
       print(hwm.__version__)

   This command should output the `hwm` package version, 
   confirming a successful installation.

Optional Dependencies
-----------------------
Installing these optional dependencies enables additional 
`hwm` features:

- **numpy** and **scipy**: Core libraries for numerical 
  computations.
- **matplotlib**: Required for visualization capabilities.
- **scikit-learn**: Useful for Scikit-learn compatibility 
  with certain estimators.
- **pandas**: Recommended for data handling and 
  preprocessing tasks.

To install `hwm` with all optional dependencies:

.. code-block:: bash

    pip install hwm[all]

Setting Up a Virtual Environment
----------------------------------
For a clean installation and to avoid conflicts, it’s 
recommended to set up a virtual environment:

1. **Create a virtual environment**:

   .. code-block:: bash

       python3 -m venv hwm_env

2. **Activate the environment**:

   - On macOS/Linux:

     .. code-block:: bash

         source hwm_env/bin/activate

   - On Windows:

     .. code-block:: bash

         hwm_env\Scripts\activate

3. **Install `hwm` in the environment**:

   .. code-block:: bash

       pip install hwm

Troubleshooting
-----------------
If you encounter issues during installation, consider the 
following steps:

1. **Check Python version**: Confirm that Python 3.9 or 
   newer is installed.

2. **Upgrade pip**: An outdated pip version may cause 
   issues. Upgrade pip using:

   .. code-block:: bash

       pip install --upgrade pip

3. **Verify dependencies**: Ensure required packages 
   like `numpy` and `scipy` are installed by running:

   .. code-block:: bash

       pip install numpy scipy

Uninstallation
----------------
To remove `hwm` from your system, use:

.. code-block:: bash

    pip uninstall hwm

This will uninstall `hwm` and remove its dependencies 
installed with the package.

Getting Started
-----------------
Once installed, refer to the :ref:`API Reference <api>` 
for detailed examples and guides on using `hwm` for dynamic 
system modeling, regression, and data analysis tasks.


