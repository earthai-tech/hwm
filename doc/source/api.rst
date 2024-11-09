
.. _api_ref:
    
=====================
hwm API Reference
=====================

Overview
--------
The :mod:`hwm` package is a comprehensive package for nonlinear dynamic modeling and analysis, 
primarily focusing on regression and classification within time-dependent and complex data scenarios. 
The package provides robust tools for working with datasets, machine learning estimators, 
and custom evaluation metrics, supporting a range of applications in fields such as 
system dynamics, financial forecasting, and geophysical analysis.

Subpackages
-------------
- **api**: Provides core API functionalities and property management.
- **compat**: Compatibility layer to ensure integration with Scikit-learn utilities and functions.
- **datasets**: A collection of synthetic datasets to support experimentation, benchmarking, and practice.
- **estimators**: Advanced machine learning estimators for handling regression and classification tasks 
  in dynamic systems.
- **utils**: Utility functions and helper modules for data transformation, context management, validation, and more.
- **metrics**: Custom evaluation metrics and performance measures designed for tracking model stability and accuracy.

Usage
-------
To use the functionalities provided by `hwm`, import the required subpackage or module:

.. code-block:: python

   import hwm
   from hwm.estimators import HammersteinWienerRegressor
   from hwm.datasets import make_system_dynamics
   from hwm.metrics import prediction_stability_score

Each subpackage is designed to integrate seamlessly into data science workflows, making it easier to 
analyze complex datasets, create advanced machine learning models, and evaluate model performance.
Whether conducting dynamic system modeling or applying nonlinear transformations, 
the `hwm` library provides tools to streamline these tasks.

.. toctree::
   :maxdepth: 2

   hwm.api
   hwm.compat
   hwm.datasets
   hwm.estimators
   hwm.metrics
   hwm.utils

.. _api_ref:

:mod:`hwm.api`: Core API Functionality
=========================================

The `hwm.api` module provides essential API functions and property management 
tools for handling core functionalities within the `hwm` package. This includes 
attribute handling and dynamic property management across various model implementations.

.. automodule:: hwm.api.property
   :no-members:
   :no-inherited-members:

**User guide:** Refer to the :ref:`api <api_ref>` section for detailed usage and examples.

.. currentmodule:: hwm

.. autosummary::
   :toctree: generated/
   :template: function.rst

    api.property.LearnerMeta
    api.property.HelpMeta


.. _compat_ref:

:mod:`hwm.compat`: Compatibility Layer
========================================

The `hwm.compat` module offers compatibility with external packages, particularly Scikit-learn, 
to ensure seamless integration and interoperability. It includes functions and classes that 
adapt Scikit-learn components for enhanced functionality in `hwm`.

.. automodule:: hwm.compat.sklearn
   :no-members:
   :no-inherited-members:

**User guide:** See the :ref:`compat <compat>` section for more information.

.. autosummary::
   :toctree: generated/
   :template: function.rst

    compat.sklearn.Interval
    compat.sklearn.get_sgd_loss_param


.. _datasets_ref:

:mod:`hwm.datasets`: Dataset Collection
=========================================

The `hwm.datasets` subpackage provides synthetic datasets that allow users to experiment with 
and benchmark machine learning models. These datasets are designed for training, testing, and 
evaluating models within the `hwm` framework, supporting use cases in dynamic systems, 
financial trends, and more.

.. automodule:: hwm.datasets._datasets
   :no-members:
   :no-inherited-members:

**User guide:** See the :ref:`datasets <datasets>` section for dataset details and usage.

.. autosummary::
   :toctree: generated/
   :template: function.rst

    datasets.make_system_dynamics
    datasets.make_financial_market_trends


.. _estimators_ref:

:mod:`hwm.estimators`: Advanced Estimators
============================================

The `hwm.estimators` module provides machine learning estimators specialized for dynamic systems. 
Key models include the Hammerstein-Wiener regression and classification models, which combine 
nonlinear and linear components to handle time-dependent and complex data. 

.. automodule:: hwm.estimators.dynamic_system
   :no-members:
   :no-inherited-members:

**User guide:** See the :ref:`estimators <estimators>` section for in-depth model documentation.

.. autosummary::
   :toctree: generated/
   :template: class.rst

    estimators.HammersteinWienerRegressor
    estimators.HammersteinWienerClassifier


.. _metrics_ref:

:mod:`hwm.metrics`: Evaluation Metrics
========================================

The `hwm.metrics` module includes specialized evaluation metrics that measure model stability, 
accuracy, and other performance aspects. These metrics are particularly useful for models applied 
to dynamic systems and time-series data, providing insights into prediction consistency and temporal stability.

.. automodule:: hwm.metrics
   :no-members:
   :no-inherited-members:

**User guide:** See the :ref:`metrics <metrics>` section for metric descriptions and examples.

.. autosummary::
   :toctree: generated/
   :template: function.rst

    metrics.prediction_stability_score
    metrics.twa_score


.. _utils_ref:

:mod:`hwm.utils`: Utilities and Helper Functions
==================================================

The `hwm.utils` subpackage offers a range of utility functions and helper modules for common 
data processing tasks, including validation, context management, and array operations. These 
utilities support efficient data handling and preprocessing, helping users streamline their 
workflows within the `hwm` ecosystem.

.. automodule:: hwm.utils
   :no-members:
   :no-inherited-members:

**User guide:** See the :ref:`utils <utils>` section for detailed information on utility functions.

.. autosummary::
   :toctree: generated/
   :template: function.rst

    utils.activator
    utils.gen_X_y_batches
    utils.resample_data
    utils.context.EpochBar

