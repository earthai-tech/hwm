
.. _user_guide:
    
==========================
User Guide
==========================

Welcome to the `hwm` package user guide! This guide provides 
comprehensive instructions on using `hwm` for dynamic system modeling, 
nonlinear regression, and other data-driven tasks. With a modular 
structure inspired by Scikit-learn, `hwm` offers flexibility, 
consistency, and ease of integration for handling complex, time-dependent 
data.

Getting Started
-----------------
`hwm` is designed for users who need robust tools for nonlinear and 
dynamic modeling. The package includes estimators for classification 
and regression, synthetic datasets for experimentation, and custom 
metrics to evaluate model performance. To get started:

1. **Install** `hwm` by following the steps in the 
   :ref:`Installation Guide <installation_guide>`.
2. **Set up a project** and familiarize yourself with the core modules 
   outlined in this guide.
3. **Explore examples** in the `examples/` directory for end-to-end 
   demonstrations.

API Structure
---------------
The `hwm` package is structured into subpackages, each with specific 
functionalities. Here’s an overview of each:

- **api**: Provides core API functionalities and dynamic property 
  management for `hwm` models.
- **compat**: A compatibility layer to ensure seamless integration 
  with Scikit-learn and related libraries.
- **datasets**: Synthetic datasets for experimentation and testing 
  models in controlled scenarios.
- **estimators**: Advanced machine learning estimators for dynamic 
  system regression and classification.
- **metrics**: Evaluation metrics tailored for time-dependent 
  modeling, providing insights into model stability and accuracy.
- **utils**: Utility functions to assist with data handling, 
  validation, and transformation.

Core Modules
--------------

api
~~~~

The `hwm.api` module contains foundational elements for managing 
properties and attributes across models. This module is essential 
for handling dynamic properties and simplifying attribute access.

Example:

.. code-block:: python

    from hwm.api.property import get_attribute, set_attribute
    
    class MyModel:
        def __init__(self):
            self.name = "DynamicModel"

    model = MyModel()
    set_attribute(model, "new_attr", "value")
    print(get_attribute(model, "new_attr"))  # Outputs: "value"

compat
~~~~~~~~~

`hwm.compat` offers compatibility tools, allowing the `hwm` estimators 
to align with Scikit-learn’s infrastructure. It includes methods for 
parameter validation, interval setting, and loss parameter management.

Example:

.. code-block:: python

    from hwm.compat.sklearn import get_sgd_loss_param

    # Retrieve default SGD loss parameter for classification tasks
    loss_param = get_sgd_loss_param()
    print(loss_param)

datasets
~~~~~~~~~

The `hwm.datasets` module provides synthetic datasets to support 
experimentation and benchmarking. Key datasets include system dynamics 
and financial trend data, each created with specific properties 
for dynamic modeling.

Example:

.. code-block:: python

    from hwm.datasets import make_system_dynamics

    # Generate a system dynamics dataset
    X, y = make_system_dynamics(samples=10000, base_price=100, price_noise_level=0.1)

estimators
~~~~~~~~~~~~

The `hwm.estimators` module houses advanced machine learning models 
for dynamic system classification and regression. The flagship 
models, `HammersteinWienerClassifier` and `HammersteinWienerRegressor`, 
offer powerful nonlinear modeling capabilities with Scikit-learn API 
compatibility.

Example:

.. code-block:: python

    from hwm.estimators import HammersteinWienerRegressor
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split

    # Sample data generation and preprocessing
    X, y = make_system_dynamics(samples=10000, base_price=10)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    # Initialize and train the model
    hw_regressor = HammersteinWienerRegressor(p=2, loss="huber")
    hw_regressor.fit(X_train, y_train)

    # Predictions
    predictions = hw_regressor.predict(X_test)

metrics
~~~~~~~~

The `hwm.metrics` module includes custom metrics tailored for time-series 
and dynamic modeling, such as `prediction_stability_score` and `twa_score`. 
These metrics provide deeper insights into model performance, especially 
in time-dependent scenarios.

Example:

.. code-block:: python

    from hwm.metrics import prediction_stability_score, twa_score

    # Calculate Prediction Stability Score and Time-Weighted Accuracy
    stability = prediction_stability_score(predictions)
    twa = twa_score(y_test, predictions)

    print(f"Prediction Stability Score: {stability:.4f}")
    print(f"Time-Weighted Accuracy: {twa:.4f}")

utils
~~~~~~~

The `hwm.utils` module offers auxiliary functions for data handling, 
validation, and processing, facilitating smooth data transformation 
and model configuration. Key utilities include batch generation, 
context management, and array operations.

Example:

.. code-block:: python

    from hwm.utils import _core, validator

    # Generate batches of data
    batches = _core.gen_X_y_batches(X_train, y_train, batch_size=32)

    # Validate data compatibility
    validator.check_X_y(X, y)

Usage Examples
--------------
The `examples/` directory provides detailed scripts demonstrating 
the use of `hwm` for various applications. These examples cover 
a range of scenarios, including:

- **Dynamic System Regression**: Building and evaluating a regression 
  model with time-based weighting for system dynamics data.
- **Financial Trend Forecasting**: Using synthetic data to train a 
  nonlinear regression model on financial trends.
- **Custom Metrics for Model Evaluation**: Applying metrics like 
  `prediction_stability_score` to assess model performance on dynamic data.

Best Practices
--------------
- **Data Preparation**: Use standardized preprocessing techniques 
  (e.g., `StandardScaler`) to normalize inputs for `hwm` models, 
  especially when working with time-series data.
- **Parameter Tuning**: Experiment with model parameters, such as 
  `p` (lag parameter) and `loss` (e.g., `huber`, `mae`), to optimize 
  performance for your dataset.
- **Model Evaluation**: Use `hwm.metrics` for detailed insights 
  into model behavior over time, ensuring stability and accuracy.

Contributions and Feedback
--------------------------
Contributions are welcomed! Follow the :ref:`Development Guide <development>` 
for instructions on contributing to `hwm`. Feedback and suggestions 
are also appreciated to improve the package’s functionality and user 
experience.

For questions or further support, please consult our GitHub repository 
or join the community forum.

