
.. _package_architecture:
    
========================
Package Architecture
========================

The `hwm` package is organized into a modular architecture, inspired 
by Scikit-learn's design principles, to facilitate advanced dynamic 
system modeling. This modularity enhances flexibility, ease of use, 
and scalability, allowing users to work with dynamic system models, 
custom metrics, and compatibility layers seamlessly.

Overview of `hwm` Structure
---------------------------
The `hwm` package is organized into subpackages and modules, each 
dedicated to a specific functionality. This structure enables users 
to utilize only the required components without dependency on the 
entire package, making `hwm` a versatile tool for various data-driven 
applications.

Here is an overview of the main package components and their 
dependencies:

.. code-block:: text

    hwm/
    ├── api/
    │   └── property.py                 # Core API for property handling
    ├── compat/
    │   └── sklearn.py                  # Compatibility layer for Scikit-learn
    ├── datasets/
    │   └── _datasets.py                # Synthetic datasets for experiments
    ├── estimators/
    │   ├── _dynamic_system.py          # Base model classes
    │   └── dynamic_system.py           # Hammerstein-Wiener models
    ├── utils/
    │   ├── _array_api.py               # Array operations and management
    │   ├── _core.py                    # Core utilities for batch generation
    │   ├── bunch.py                    # Data organization and bundling
    │   ├── context.py                  # Context management utilities
    │   └── validator.py                # Data validation and checks
    ├── __init__.py                     # Package initialization
    ├── exceptions.py                   # Custom exceptions
    └── metrics.py                      # Custom metrics for model evaluation

Subpackages Overview
----------------------
Each subpackage within `hwm` plays a crucial role in supporting the 
core functionality of dynamic system modeling and regression.

- **api**: Core API utilities to dynamically manage model attributes 
  and properties. This module provides tools for efficient property 
  handling across the package.

- **compat**: Ensures compatibility with Scikit-learn and related 
  libraries. It includes utilities for parameter validation and 
  provides the necessary hooks to enable smooth integration with 
  Scikit-learn’s API.

- **datasets**: Contains synthetic datasets designed for testing and 
  benchmarking `hwm` models. It includes functions to generate time-
  series and sequence-based data, particularly useful in dynamic system 
  applications.

- **estimators**: This subpackage implements the main models in `hwm`, 
  including `HammersteinWienerClassifier` and `HammersteinWienerRegressor`. 
  These models are built with a flexible architecture to handle complex 
  nonlinear and dynamic relationships.

- **utils**: Provides a set of helper modules for tasks like data 
  validation, context management, array manipulation, and batch 
  processing. The utility functions support other subpackages and 
  streamline data transformations.

Modules and Dependencies
--------------------------
Each module in `hwm` has been designed with specific functionalities 
in mind. The following schema shows dependencies among modules, 
illustrating how different components interact within the package:

.. mermaid::

    graph TD;
        A[hwm/__init__.py] --> B[hwm/api/property.py]
        A --> C[hwm/compat/sklearn.py]
        A --> D[hwm/datasets/_datasets.py]
        A --> E[hwm/estimators/dynamic_system.py]
        E --> F[hwm/estimators/_dynamic_system.py]
        E --> G[hwm/metrics.py]
        F --> G
        E --> H[hwm/utils/_core.py]
        E --> I[hwm/utils/validator.py]
        G --> J[hwm/utils/_array_api.py]
        G --> K[hwm/utils/context.py]
        G --> L[hwm/exceptions.py]

.. note::

    The dependencies between modules ensure modularity, where each 
    component can be used independently or in combination, depending 
    on the user's requirements.

Architectural Highlights
--------------------------
The following design principles were applied in the construction of 
`hwm`:

- **Modularity**: Each subpackage and module is dedicated to a 
  specific function, allowing users to import and use only the 
  components they need.

- **Compatibility**: `hwm` follows Scikit-learn’s API conventions, 
  providing seamless integration with Scikit-learn and other 
  popular data science libraries.

- **Extensibility**: The architecture supports future expansion. 
  New models, metrics, or datasets can be added without affecting 
  existing components, ensuring the package is flexible and future-proof.

- **Optimization**: By implementing utility functions in `utils`, 
  the package enhances performance for tasks such as batch generation, 
  data validation, and array operations, especially for time-series 
  and dynamic system data.

Using `hwm`
-------------
To get started, import and initialize models and utilities as needed:

.. code-block:: python

    from hwm.estimators import HammersteinWienerClassifier
    from hwm.datasets import make_system_dynamics
    from hwm.metrics import prediction_stability_score

    # Load a sample dataset
    X, y = make_system_dynamics(n_samples=1000, sequence_length=10)

    # Initialize and train the model
    model = HammersteinWienerClassifier(p=5)
    model.fit(X, y)

    # Calculate metrics
    predictions = model.predict(X)
    stability_score = prediction_stability_score(predictions)

This modular and structured approach allows users to combine various 
components for comprehensive and flexible modeling of dynamic systems. 

For further details on usage, please refer to the :ref:`User Guide <user_guide>` 
and :ref:`API Reference <api_ref>`.
