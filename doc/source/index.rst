.. hwm documentation master file, created by
   sphinx-quickstart on Sat Nov  9 10:16:42 2024.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to the hwm Documentation!
===================================

The `hwm` package offers a powerful and flexible implementation of the 
Adaptive Hammerstein-Wiener (HW) Model, designed to handle dynamic system 
modeling in intelligent computing environments. By integrating both linear 
and nonlinear dependencies, the HW model is capable of effectively capturing 
complex, time-dependent patterns in sequential data.

The package is particularly suitable for advanced applications such as 
network intrusion detection, financial forecasting, industrial automation, 
and other tasks where accurate predictions and adaptability to dynamic systems 
are essential. The model’s architecture supports both regression and classification 
tasks, making it versatile for a variety of machine learning challenges.

This documentation will guide you through installing and using the `hwm` package, 
and provide detailed explanations of its components and functionalities.

Overview of the Adaptive Hammerstein-Wiener Model
---------------------------------------------------

The Adaptive Hammerstein-Wiener (HW) Model is an extension of traditional 
Hammerstein-Wiener systems, optimized for machine learning tasks. It consists 
of the following components:

- **Nonlinear Input Transformation**:  
  This module encodes the complex relationships between input features. It 
  applies a nonlinear transformation to the input data to model its intricate 
  characteristics before passing it to the linear dynamic block. This step is 
  crucial for capturing the nonlinearities in real-world data.

- **Linear Dynamic System Block**:  
  The linear dynamic system block captures the temporal dependencies across 
  multiple time steps. It models how current states depend on past states, 
  facilitating the modeling of dynamic relationships in sequential data. This 
  block can handle both stationary and non-stationary data and is a core feature 
  for time series forecasting.

- **Nonlinear Output Transformation**:  
  After the linear dynamic block, the output is transformed through a nonlinear 
  mapping to produce task-specific predictions. This ensures that the model can 
  adapt to various types of outputs, such as categorical labels (for classification) 
  or continuous values (for regression).

The unique architecture of the HW model is designed to deliver both high 
interpretability and computational efficiency. It is particularly useful in 
scenarios where accurate predictions and temporal stability are required, making 
it well-suited for real-time applications.

Additionally, the package includes custom evaluation metrics:

- **Prediction Stability Score (PSS)**:  
  Measures the consistency of the model’s predictions over time, assessing 
  its robustness in dynamic environments.
  
- **Time-weighted Accuracy (TWA)**:  
  Evaluates the model's accuracy while taking into account the timing of 
  predictions, ensuring that the model performs well in time-sensitive tasks.

Contents
----------

.. toctree::
   :maxdepth: 2
   :caption: Table of Contents:

   installation
   user_guide
   api
   compat
   datasets
   estimators
   utils
   metrics
   examples
   package_architecture
   development

Subpackages
-------------

The `hwm` package is divided into several subpackages, each serving a specific 
function to ensure the model’s proper operation in various machine learning 
workflows:

- **installation**:  
  This subpackage provides all the necessary instructions to install and 
  configure the `hwm` package. It includes detailed steps for setting up 
  dependencies and environment requirements, ensuring users can get started 
  quickly and without hassle. It also includes troubleshooting tips for common 
  installation issues.

- **examples**:  
  The `examples` subpackage includes a collection of scripts designed to 
  demonstrate the HW model’s application across multiple domains. Examples 
  cover tasks such as network intrusion detection, time series forecasting, 
  and industrial system monitoring, showcasing how the HW model can be used 
  to solve real-world problems.

- **api**:  
  The :mod:`hwm.api` subpackage contains core components that define the HW model’s 
  settings, configurations, and hyperparameters. The primary file, `property.py`, 
  manages model parameters and configuration options, allowing users to customize 
  the model's behavior for different tasks. This is where users can define 
  model-specific properties, such as input-output transformations, time lags, 
  and regularization settings.

- **compat**:  
  The :mod:`hwm.compat` subpackage contains compatibility tools that help integrate 
  the `hwm` model with other popular machine learning libraries, such as 
  scikit-learn. This ensures seamless interoperability with standard data 
  science workflows and simplifies the process of including the HW model into 
  existing pipelines.

- **datasets**:  
  The :mod:`hwm.datasets` subpackage includes utilities for managing datasets, including 
  data loaders and preprocessors. The primary file, `_datasets.py`, provides 
  functions for loading common datasets, handling missing data, and preprocessing 
  input features. It is designed to make it easier to work with large datasets 
  and integrate them with the HW model for training and evaluation.

- **estimators**:  
  The :mod:`hwm.estimators` subpackage contains the core model classes that implement 
  the HW model’s nonlinear and dynamic processing blocks. The files 
  `_dynamic_system.py` and `dynamic_system.py` define the internal structure 
  of the model, including the nonlinear input transformation, linear dynamic 
  system, and nonlinear output transformation. These estimators can be customized 
  for specific applications, such as classification or regression tasks, and 
  are the heart of the model’s predictive capabilities.

- **utils**:  
  The :mod:`hwm.utils` subpackage provides a variety of general-purpose utility 
  functions that support the overall functionality of the `hwm` package. Key 
  modules include:
  
  - **:mod:`hwm.utils.context.py**: Manages the context in which the model is run, helping 
    to organize and track the model’s state during training and evaluation.
  - **:mod:`hwm.utils.validator.py**: Validates input data, ensuring that the data used 
    for training or testing the model is correctly formatted and meets the 
    necessary requirements.
  - Other utility functions support tasks like logging, error handling, 
    and model evaluation.

- **metrics**:  
  The :mod:`hwm.metrics` subpackage defines custom evaluation metrics that are 
  specifically designed for the HW model. These include:
  
  - **Prediction Stability Score (PSS)**: A metric that measures the 
    consistency of the model’s predictions over time, highlighting the 
    model’s ability to handle dynamic, time-varying data.
  - **Time-weighted Accuracy (TWA)**: A metric that evaluates the model’s 
    time-sensitive performance, giving more weight to accurate predictions 
    made during critical time windows.

Indices and Tables
====================

The following resources are available to help you navigate the `hwm` documentation:

* :ref:`genindex` 
* :ref:`modindex` 
* :ref:`search` 

