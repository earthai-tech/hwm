.. _estimators:

===========================
Dynamic System Estimators
===========================

The dynamic system estimators are designed to model complex, 
time-dependent data by combining dynamic system theory with 
machine learning techniques. These models consist of nonlinear 
input-output relationships and linear dynamic components, making 
them suitable for both classification and regression tasks.

This module includes the following estimators:

- **Hammerstein-Wiener Classifier**: A classifier for modeling 
  dynamic systems with nonlinear input-output relationships and 
  linear dynamics.
- **Hammerstein-Wiener Regressor**: A regressor for modeling 
  dynamic systems with nonlinear input-output relationships and 
  linear dynamics.

Overview of Common Parameters
===============================

The common parameters across both :class:`~hwm.estimators.dynamic_system.HammersteinWienerClassifier`
and :class:`~hwm.estimators.dynamic_system.HammersteinWienerRegressor` allow 
for fine-tuning the model's behavior during training. These include:

1. **nonlinear_input_estimator** (`estimator`, default=None):
   - The estimator used to model the nonlinear relationship at the 
     input. This can be any estimator from scikit-learn that implements 
     `fit` and either `transform` or `predict`. 
   - If `None`, no nonlinear transformation is applied to the input 
     data.

2. **nonlinear_output_estimator** (`estimator`, default=None):
   - The estimator used to model the nonlinear relationship at the 
     output. This must implement `fit` and either `transform` or 
     `predict`.
   - If `None`, no nonlinear transformation is applied to the output 
     data.

3. **p** (`int`, default=1):
   - The number of lagged observations (past time steps) included in 
     the model. This parameter determines how many past values of the 
     input are used to predict the current output.

4. **loss** (`str`, default="mse" for Regressor, "cross_entropy" 
   for Classifier):
   - The loss function used to optimize the model during training. 
     This dictates how the error between the predicted and true outputs 
     is computed.
     - For Regressor: 
       - `"mse"`: Mean Squared Error, commonly used for regression 
         tasks.
       - `"mae"`: Mean Absolute Error.
       - `"huber"`: Huber Loss, a robust loss function.
       - `"time_weighted_mse"`: Time-weighted Mean Squared Error.
     - For Classifier:
       - `"cross_entropy"`: The negative log-likelihood loss function, 
         used for classification tasks.
       - `"log_loss"`: A common alternative for binary or multi-class 
         classification tasks.

5. **output_scale** (`tuple` or `None`, default=None):
   - If provided, the predictions will be scaled to fit within the 
     specified range (min-max scaling). 
   - Example: `output_scale=(0, 1)` scales the predictions to the range 
     [0, 1]. If `None`, no scaling is applied.

6. **time_weighting** (`str` or `None`, default="linear"):
   - Defines how time-based weights are applied to the loss function, 
     affecting the emphasis given to recent versus older data points.
     - `"linear"`: Linearly increasing weights over time.
     - `"exponential"`: Exponentially increasing weights.
     - `"inverse"`: Weights inversely proportional to time.
     - `None`: Equal weights for all time steps.

7. **feature_engineering** (`str`, default='auto'):
   - Specifies how features are generated for the input data.
     - `'auto'`: Automatically generates features based on the number 
       of lagged observations (`p`).

8. **delta** (`float`, default=1.0):
   - The threshold parameter for the Huber loss function. This defines 
     the point where the loss transitions from quadratic to linear, 
     making the model less sensitive to outliers.

9. **epsilon** (`float`, default=1e-8):
   - A small constant added to avoid division by zero when scaling 
     the output or applying other transformations.

10. **shuffle** (`bool`, default=True):
    - Whether to shuffle the training data before each epoch during 
      training. This helps prevent overfitting and ensures better 
      generalization.

11. **batch_size** (`int` or `str`, default='auto'):
    - Determines the number of samples per gradient update during 
      training.
    - If set to `'auto'`, the batch size is determined automatically 
      based on the dataset size.

12. **optimizer** (`str`, default='adam'):
    - The optimization algorithm used to train the linear dynamic block.
      - `'sgd'`: Stochastic Gradient Descent.
      - `'adam'`: Adaptive Moment Estimation, often used for training 
        deep learning models.
      - `'adagrad'`: Adaptive Gradient Algorithm.

13. **learning_rate** (`float`, default=0.001):
    - The learning rate for the optimizer, which controls the step size 
      during gradient descent updates.

14. **max_iter** (`int`, default=1000):
    - The maximum number of iterations (epochs) for training the model.

15. **tol** (`float`, default=1e-3):
    - The tolerance for the optimization process. Training stops when 
      the loss improvement is below this threshold.

16. **early_stopping** (`bool`, default=False):
    - If set to `True`, training will stop early if the validation 
      loss does not improve after a certain number of iterations.

17. **validation_fraction** (`float`, default=0.1):
    - The proportion of the training data to reserve for validation 
      during early stopping.

18. **n_iter_no_change** (`int`, default=5):
    - The number of iterations with no improvement before stopping the 
      training process early.

19. **random_state** (`int`, `RandomState`, or `None`, default=None):
    - Controls the random number generation for reproducibility of 
      results across different runs.

20. **n_jobs** (`int` or `None`, default=None):
    - The number of CPU cores to use for training. If set to `-1`, all 
      available cores will be used.

21. **verbose** (`int`, default=0):
    - Controls the verbosity of the training process. Higher values 
      result in more detailed logs.

Hammerstein-Wiener Classifier
===============================

The `HammersteinWienerClassifier`( :class:`hwm.estimators.HWClassifier` ) is a dynamic system model for 
classification tasks. It utilizes the Hammerstein-Wiener model 
structure, which is composed of a nonlinear input block, a linear 
dynamic block, and a nonlinear output block. This allows it to capture 
complex relationships within time-series or sequential data.

Mathematical Formulation
--------------------------

The Hammerstein-Wiener model for classification is represented as:

.. math::
    \mathbf{y} = f_{\text{output}}\left( \mathbf{H} 
    f_{\text{input}}\left( \mathbf{X} \right) \right)

where:
- :math:`f_{\text{input}}` is the nonlinear input estimator (e.g., a 
  neural network or polynomial function) that maps the input data 
  :math:`\mathbf{X}` to a higher-dimensional space.
- :math:`\mathbf{H}` is the linear dynamic block (e.g., a set of 
  regression coefficients) that captures the linear relationships and 
  dynamics in the data.
- :math:`f_{\text{output}}` is the nonlinear output estimator (e.g., 
  a sigmoid or softmax function) that maps the linear dynamic output 
  to the class probabilities or labels.

The model works by first transforming the input data through a nonlinear 
function, then applying a linear dynamic process, and finally applying 
another nonlinear transformation at the output to generate the predicted 
class probabilities or labels.

Additional Parameters
-----------------------

In addition to the common parameters shared with the regressor, the 
classifier has the following specific parameters:

1. **loss** (`str`, default="cross_entropy"):
   - The loss function used for classification. The default is 
     `"cross_entropy"`, which is commonly used in classification 
     problems. Other options include:
       - `"log_loss"`: A common alternative for binary and multi-class 
         classification tasks.
   
   The loss is defined as the negative log-likelihood of the true class 
   labels given the predicted probabilities:

   .. math::
       \text{Loss} = - \sum_{i=1}^{N} y_i \log(p_i)

   where :math:`y_i` is the true label for the :math:`i`-th sample, and 
   :math:`p_i` is the predicted probability for the true class.

2. **time_weighting** (`str` or `None`, default="linear"):
   - Defines how time-based weights are applied to the loss function, 
     affecting the emphasis given to recent versus older data points.
     - `"linear"`: Linearly increasing weights over time.
     - `"exponential"`: Exponentially increasing weights.
     - `"inverse"`: Weights inversely proportional to time.
     - `None`: Equal weights for all time steps.

3. **feature_engineering** (`str`, default='auto'):
   - Specifies how features are generated for the input data.
     - `'auto'`: Automatically generates

Example Usage
---------------

The following examples demonstrate training and evaluating 
the `HammersteinWienerClassifier` on synthetic classification data 
and a system dynamics dataset.

**Example 1: Synthetic Classification Data**

.. code-block:: python

    from sklearn.preprocessing import StandardScaler
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    from hwm.estimators import HWClassifier

    # Generate synthetic data
    X, y = make_classification(n_samples=1000, n_features=20, 
                               n_informative=15, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Initialize classifier with nonlinear transformations
    hw_classifier = HWClassifier(
        nonlinear_input_estimator=StandardScaler(),
        nonlinear_output_estimator=StandardScaler(),
        p=2,
        loss="cross_entropy",
        time_weighting="linear",
        optimizer='adam',
        learning_rate=0.01,
        batch_size=64,
        max_iter=500,
        tol=1e-4,
        early_stopping=True,
        validation_fraction=0.2,
        n_iter_no_change=10,
        shuffle=True,
        epsilon=1e-10,
        n_jobs=-1,
        verbose=1
    )

    # Fit model
    hw_classifier.fit(X_train, y_train)

    # Predict class labels
    predictions = hw_classifier.predict(X_test)

    # Predict probabilities
    probabilities = hw_classifier.predict_proba(X_test)
    

**Example 2: System Dynamics Dataset**

This example demonstrates using the `make_system_dynamics` dataset from 
`hwm.datasets`, scaling the data, training a classifier, and evaluating 
performance using `twa_score` and `prediction_stability_score`.


.. code-block:: python

    import numpy as np
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split
    from hwm.estimators import HWClassifier
    from hwm.datasets import make_system_dynamics
    from hwm.metrics import twa_score, prediction_stability_score
    
    # Generate system dynamics data
    X, y = make_system_dynamics(
        n_samples=20000, 
        sequence_length=10, 
        noise=0.1, 
        return_X_y=True 
    )

    # Categorize 'y' into three classes for classification
    # Define bins to convert continuous 'y' values into categorical classes
    y = np.digitize(y, bins=[-0.5, 0.5, 1.5])

    # Scale features
    scaler = StandardScaler()
    X = scaler.fit_transform(X.reshape(-1, X.shape[-1])).reshape(X.shape)

    # Split data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Initialize classifier
    hw_classifier = HWClassifier(
        p=3,
        nonlinear_input_estimator=StandardScaler(),
        nonlinear_output_estimator=StandardScaler(),
        time_weighting="exponential",
        optimizer="adam",
        learning_rate=0.001,
        batch_size="auto",
        max_iter=1000,
        tol=1e-3,
        early_stopping=True,
        validation_fraction=0.2,
        n_jobs=-1,
    )

    # Fit model
    hw_classifier.fit(X_train, y_train)

    # Make predictions
    predictions = hw_classifier.predict(X_test)

    # Calculate TWA (Time-Weighted Accuracy) score
    twa = twa_score(y_test, predictions)

    # Calculate Prediction Stability Score
    stability_score = prediction_stability_score(predictions)

    # Print evaluation metrics
    print(f"Time-Weighted Accuracy (TWA): {twa:.4f}")
    print(f"Prediction Stability Score: {stability_score:.4f}")


.. note::

    - **Lag Parameter (`p`)**: Setting the `p` parameter affects 
      temporal dependencies. A higher value of `p` allows the model 
      to consider more past data points, enhancing the predictive power 
      for time-series tasks. For example, a `p` value of 5 includes 
      observations from the previous 5 time steps, which may improve 
      prediction accuracy in systems with significant temporal dependencies.

    - **Time Weighting**: Time-based weighting can be used to emphasize 
      recent data over older observations. The `"linear"` or `"exponential"` 
      schemes apply increasing weights over time, which is beneficial for 
      time-sensitive applications such as predicting real-time events or trends. 
      The choice of weighting scheme (`linear`, `exponential`, `inverse`, or `None`) 
      depends on the application and how much importance should be placed on 
      recent data points.

    - **Optimizer Selection**: Different optimizers perform differently 
      depending on the dataset and application. While `adam` is a robust 
      optimizer for most cases, it is worth experimenting with others, such as 
      `sgd` or `adagrad`, especially if your model encounters issues with 
      convergence or training time. You may find that adjusting the learning 
      rate or changing the optimizer improves model performance, particularly 
      in cases with noisy or sparse data.

    - **Learning Rate and Batch Size**: The `learning_rate` parameter controls 
      how quickly the model adjusts its parameters during training. Lower learning 
      rates might yield more stable training but could take longer to converge, 
      while higher rates could speed up training but risk overshooting the optimal 
      parameters. Similarly, the `batch_size` controls how many samples are 
      used to compute the gradient at each update step. A larger batch size 
      leads to more stable gradients, while a smaller batch size can result 
      in faster convergence with noisier updates.

.. seealso::

    - :class:`~hwm.estimators.HammersteinWienerRegressor`
    - :class:`~sklearn.linear_model.SGDRegressor`
    - :class:`~sklearn.linear_model.LogisticRegression`

Hammerstein-Wiener Regressor
==============================

The :class:`~hwm.estimators.dynamic_system.HWRegressor` class implements a nonlinear regression 
model based on the Hammerstein-Wiener (HW) architecture. This block-structured model combines a nonlinear 
input transformation, a linear dynamic system block, and a nonlinear output transformation, making it highly 
suitable for regression tasks where data exhibit complex, time-dependent relationships. The HW model is 
designed to capture both nonlinear and linear dependencies, offering robust predictive performance while 
preserving interpretability.

.. math::

    \mathbf{y} = f_{\text{output}} \left( \mathbf{H} f_{\text{input}}
    \left( \mathbf{X} \right) \right)

where:
- :math:`f_{\text{input}}` is the nonlinear input estimator,
- :math:`\mathbf{H}` represents the linear dynamic block,
- :math:`f_{\text{output}}` is the nonlinear output estimator.

Additional Parameters
-----------------------

These parameters are specific to :class:`~hwm.estimators.dynamic_system.HammersteinWienerRegressor`
 and complement the standard parameters shared with other estimators in this package.

- **nonlinear_input_estimator**: (estimator, default=None) 
  Estimator for capturing nonlinear relationships at the input stage. 
  It should implement `fit` and either `transform` or `predict`. If `None`, 
  no nonlinear transformation is applied to the input data.

- **nonlinear_output_estimator**: (estimator, default=None) 
  Estimator for nonlinear output transformations, with methods `fit` and 
  either `transform` or `predict`. If `None`, no nonlinear transformation 
  is applied at the output stage.

- **loss**: (str, default="mse") Specifies the loss function used during 
  training. Available options include:
  
  - `"mse"`: Mean Squared Error (standard loss for regression).
  - `"mae"`: Mean Absolute Error (robust against outliers).
  - `"huber"`: Huber Loss (combines MSE and MAE benefits, handling 
    outliers effectively).
  - `"time_weighted_mse"`: Time-Weighted Mean Squared Error (applies 
    time-based importance to errors, emphasizing recent observations).

- **output_scale**: (tuple or None, default=None) Desired output range for 
  predictions. When provided, min-max scaling adjusts outputs to this 
  range, useful for bounding predictions within a specific interval.

- **delta**: (float, default=1.0) Threshold for the Huber loss, defining 
  the transition point from quadratic to linear loss behavior.

Attributes
------------

- **linear_model_**: Represents the linear dynamic block trained via 
  stochastic gradient descent, central to capturing linear dependencies.

- **best_loss_**: Tracks the best validation loss encountered during 
  training, used in early stopping to optimize convergence.

- **initial_loss_**: Initial loss on the full dataset post-training, 
  providing a baseline for performance evaluation.

Example Usage
---------------

Below are example applications of :class:`hwm.estimators.HWRegressor`, 
demonstrating initialization, training, and evaluation on synthetic data. 
The first example shows time-series data, while the second highlights 
financial trend forecasting.

**Example 1: Basic Time-Series Regression**

This example demonstrates using `HWRegressor` to train 
and predict on synthetic time-series data.

.. code-block:: python

    from hwm.estimators import HWRegressor
    from sklearn.preprocessing import StandardScaler
    from sklearn.datasets import make_regression
    from sklearn.model_selection import train_test_split
    from hwm.datasets import make_system_dynamics

    # Generate synthetic time-series data
    X, y = make_regression(n_samples=20000, sequence_length=10, noise=0.1, return_X_y=True)

    # Scale input data
    scaler = StandardScaler()
    X = scaler.fit_transform(X.reshape(-1, X.shape[-1])).reshape(X.shape)

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize Hammerstein-Wiener Regressor
    hw_regressor = HammersteinWienerRegressor(
        p=2,
        nonlinear_input_estimator=StandardScaler(),
        nonlinear_output_estimator=StandardScaler(),
        loss="huber",
        output_scale=(-1, 1),
        time_weighting="linear",
        optimizer="adam",
        learning_rate=0.001,
        batch_size=64,
        max_iter=500,
        tol=1e-4,
        early_stopping=True,
        validation_fraction=0.2,
        n_jobs=-1
    )

    # Train the model
    hw_regressor.fit(X_train, y_train)

    # Predict on test data
    predictions = hw_regressor.predict(X_test)


**Example 2: Financial Market Trend Forecasting**

This example demonstrates using `HWRegressor` on synthetic financial 
trend data, emphasizing the model’s adaptability to financial forecasting tasks.

.. code-block:: python

    from hwm.estimators import HWRegressor
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split
    from hwm.datasets import make_financial_market_trends
    from hwm.metrics import prediction_stability_score

    # Generate synthetic financial market trend data
    X, y = make_financial_market_trends(
        n_samples=20000, 
        price_noise_level=0.2,
        volatility_level=0.03,
        nonlinear_trend=True,
        base_price=100.0,
        trend_frequency=1/252,
        market_sensitivity=0.07,
        trend_strength=0.02,
        return_X_y=True
    )

    # Scale input features
    scaler = StandardScaler()
    X = scaler.fit_transform(X.reshape(-1, X.shape[-1])).reshape(X.shape)

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize Hammerstein-Wiener Regressor
    hw_regressor = HWRegressor(
        p=3,
        nonlinear_input_estimator=StandardScaler(),
        nonlinear_output_estimator=StandardScaler(),
        loss="huber",
        output_scale=(-1, 1),
        time_weighting="exponential",
        optimizer="adam",
        learning_rate=0.001,
        batch_size=64,
        max_iter=500,
        tol=1e-4,
        early_stopping=True,
        validation_fraction=0.2,
        n_jobs=-1
    )

    # Train the model
    hw_regressor.fit(X_train, y_train)

    # Make predictions on the test set
    predictions = hw_regressor.predict(X_test)

    # Calculate Prediction Stability Score, indicating temporal consistency
    stability_score = prediction_stability_score(predictions)
    print(f"Prediction Stability Score: {stability_score:.4f}")

.. note::

    - **Lag Parameter (`p`)**: Setting a higher `p` value allows the model 
      to incorporate additional lagged observations, enhancing its ability 
      to model temporal dependencies. However, higher values may increase 
      computational costs.

    - **Loss Function Selection**: The `"huber"` loss function provides 
      robustness against outliers, which can be advantageous for noisy data. 
      Other loss options like `"mse"` or `"mae"` may suit different datasets 
      depending on the presence of outliers and the overall objective.

    - **Output Scaling**: Setting an `output_scale` (e.g., `(-1, 1)`) 
      constrains predictions within a specified range, which is beneficial 
      in cases where the target values should remain bounded.

    - **Time Weighting**: Time-based weighting options, such as `"linear"` 
      or `"exponential"`, allow the model to prioritize recent data, which 
      is particularly useful in forecasting contexts where recent trends 
      are more relevant for prediction.

.. seealso::

    - :class:`~hwm.estimators.HWClassifier`
    - :class:`~sklearn.linear_model.SGDRegressor`


References
------------

.. [1] Hammerstein, W. (1950). "Beiträge zum Problem der 
   adaptiven Regelung". *Zeitschrift für angewandte Mathematik 
   und Mechanik*, 30(3), 345-367.

.. [2] Wiener, N. (1949). "Extrapolation, Interpolation, and 
   Smoothing of Stationary Time Series". *The MIT Press*.

.. [3] Ljung, L. (1999). *System Identification: Theory for the 
   User*. Prentice Hall.

.. [4] Goodfellow, I., Bengio, Y., & Courville, A. (2016). *Deep 
   Learning*. MIT Press.

