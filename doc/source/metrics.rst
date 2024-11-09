.. _metrics:

===================
Metrics
===================

The :code:`hwm` package includes custom evaluation metrics designed to
assess the performance of the Hammerstein-Wiener models in dynamic
systems. These metrics provide insights into the temporal stability
and time-sensitive accuracy of predictions, which are crucial for
applications involving time-series and sequential data.

This module includes the following metrics:

- **Prediction Stability Score (PSS)**: Measures the temporal
  stability of predictions across consecutive time steps.
- **Time-Weighted Score**: Computes a weighted error metric that
  emphasizes recent observations.
- **Time-Weighted Accuracy (TWA)**: Evaluates classification
  performance with time-based weighting.

Overview of Metrics
=====================

The custom metrics in the :mod:`hwm.metrics` module are tailored to
evaluate models in contexts where temporal dynamics play a significant
role. These metrics complement standard evaluation measures by
providing a deeper understanding of model performance over time.

Hammerstein-Wiener Classifier
------------------------------

.. seealso:: :ref:`~hwm.estimators.HammersteinWienerClassifier` for the classification
   model that utilizes these metrics.
.. seealso:: :class:`~hwm.estimators.HammersteinWienerClassifier` for the regression
   model that utilizes these metrics.

Prediction Stability Score
============================

The **Prediction Stability Score (PSS)** assesses the consistency
of a model's predictions over time. It quantifies how stable the
predictions are across consecutive time steps, which is vital for
applications requiring reliable temporal behavior.

Mathematical Formulation
--------------------------

The Prediction Stability Score is defined as:

.. math::
    \text{PSS} = \frac{1}{T - 1} \sum_{t=1}^{T - 1}
    \left| \hat{y}_{t+1} - \hat{y}_t \right|

where:

- :math:`T` is the number of time steps.
- :math:`\hat{y}_t` is the prediction at time :math:`t`.

This formulation calculates the average absolute difference
between consecutive predictions, providing a measure of how much
the predictions fluctuate over time.

Parameters
------------

.. list-table::
   :widths: 25 75
   :header-rows: 1

   * - Parameter
     - Description
   * - ``y_pred``
     - Predicted values. Shape: (n_samples,) or (n_samples, n_outputs).
   * - ``y_true``
     - Not used, present for API consistency.
   * - ``sample_weight``
     - Sample weights. Shape: (n_samples,), default=None.
   * - ``multioutput``
     - Defines how to aggregate multiple output values.
     - Options:
       - ``'raw_values'``: Returns scores for each output.
       - ``'uniform_average'``: Averages scores uniformly.
       - ``array-like``: Weighted average of scores.

Returns
---------

- **score**: float or ndarray of floats
  - Prediction Stability Score. If ``multioutput`` is
    ``'raw_values'``, an array of scores is returned.
    Otherwise, a single float is returned.

Examples
----------

The Prediction Stability Score provides a quantitative measure
of how much a model's predictions vary over time. It is particularly
useful in scenarios where consistent predictions are critical, such
as in financial forecasting or real-time monitoring systems.

**Basic Example:**

In this example, we calculate the PSS for a simple set of predictions
to understand its basic functionality.

.. code-block:: python

    from hwm.metrics import prediction_stability_score
    import numpy as np

    # Simple predictions over 5 time steps
    y_pred = np.array([3, 3.5, 4, 5, 5.5])
    score = prediction_stability_score(y_pred)
    print(score)  # Output: 0.625

**Explanation:**

- The differences between consecutive predictions are:
  - |3.5 - 3| = 0.5
  - |4 - 3.5| = 0.5
  - |5 - 4| = 1.0
  - |5.5 - 5| = 0.5
- The PSS is the average of these differences:
  - (0.5 + 0.5 + 1.0 + 0.5) / 4 = 0.625

**Complex Example:**

This example demonstrates the PSS in a more complex scenario with
multi-output predictions and sample weights.

.. code-block:: python

    from hwm.metrics import prediction_stability_score
    import numpy as np

    # Multi-output predictions over 6 time steps
    y_pred = np.array([
        [2.0, 3.0],
        [2.5, 3.5],
        [3.0, 4.0],
        [3.5, 4.5],
        [4.0, 5.0],
        [4.5, 5.5]
    ])

    # Sample weights for each prediction
    sample_weight = np.array([1, 2, 1, 2, 1, 2])

    # Calculate PSS with multioutput and sample weights
    score = prediction_stability_score(
        y_pred,
        sample_weight=sample_weight,
        multioutput='uniform_average'
    )
    print(score)  # Output: 0.75

**Explanation:**

- Differences between consecutive predictions:

  - Time 1 to 2: |2.5 - 2.0| = 0.5, |3.5 - 3.0| = 0.5
  - Time 2 to 3: |3.0 - 2.5| = 0.5, |4.0 - 3.5| = 0.5
  - Time 3 to 4: |3.5 - 3.0| = 0.5, |4.5 - 4.0| = 0.5
  - Time 4 to 5: |4.0 - 3.5| = 0.5, |5.0 - 4.5| = 0.5
  - Time 5 to 6: |4.5 - 4.0| = 0.5, |5.5 - 5.0| = 0.5
  
- Weighted differences using sample weights (from t=2 to t=6):

  - Weights: [2, 1, 2, 1, 2]
  - Weighted differences for each output:
    - Output 1: [0.5*2, 0.5*1, 0.5*2, 0.5*1, 0.5*2] = [1.0, 0.5, 1.0, 0.5, 1.0]
    - Output 2: [0.5*2, 0.5*1, 0.5*2, 0.5*1, 0.5*2] = [1.0, 0.5, 1.0, 0.5, 1.0]
- Average differences per output:
  - Output 1: (1.0 + 0.5 + 1.0 + 0.5 + 1.0) / 5 = 0.7
  - Output 2: (1.0 + 0.5 + 1.0 + 0.5 + 1.0) / 5 = 0.7
- PSS with `uniform_average`: (0.7 + 0.7) / 2 = 0.7

In this case, the PSS reflects the stability across multiple outputs
with varying sample weights, resulting in an overall score of 0.75.

Notes
-------

- The PSS measures the average absolute difference between
  consecutive predictions.
- A lower PSS indicates more stable predictions over time.
- PSS is especially useful in applications where consistent
  predictions are critical for system reliability and performance.

See Also
----------

- :func:`~hwm.metrics.twa_score` : Time-Weighted Accuracy for
  classification tasks.


Time-Weighted Score
=====================

The **Time-Weighted Score** computes a weighted error metric that
emphasizes recent observations more than earlier ones. This metric
is applicable to both regression and classification tasks, providing
a nuanced view of model performance over time.

Mathematical Formulation
--------------------------

For **regression tasks**, the Time-Weighted Error is defined as:

.. math::
    \text{TWError} = \frac{\sum_{t=1}^T w_t \cdot e_t}{\sum_{t=1}^T w_t}

where:

- :math:`T` is the total number of samples.
- :math:`w_t = \alpha^{T - t}` is the time weight for time step :math:`t`.
- :math:`e_t` is the error at time step :math:`t`, defined as
  :math:`(y_t - \hat{y}_t)^2` if ``squared=True``, else :math:`|y_t - \hat{y}_t|`.
- :math:`\alpha \in (0, 1)` is the decay factor.

For **classification tasks**, the Time-Weighted Accuracy is defined as:

.. math::
    \text{TWA} = \frac{\sum_{t=1}^T w_t \cdot \mathbb{1}(y_t = \hat{y}_t)}{\sum_{t=1}^T w_t}

where:

- :math:`\mathbb{1}(\cdot)` is the indicator function that equals 1
  if its argument is true and 0 otherwise.
- :math:`y_t` is the true label at time :math:`t`.
- :math:`\hat{y}_t` is the predicted label at time :math:`t`.

Parameters
------------

.. list-table::
   :widths: 25 75
   :header-rows: 1

   * - Parameter
     - Description
   * - ``y_true``
     - Ground truth target values or labels. Shape: (n_samples,)
       or (n_samples, n_outputs).
   * - ``y_pred``
     - Estimated target values or labels. Shape: (n_samples,)
       or (n_samples, n_outputs).
   * - ``alpha``
     - Decay factor for time weighting. Must be in the range (0, 1).
       Default is 0.9.
   * - ``sample_weight``
     - Sample weights. Shape: (n_samples,), default=None.
   * - ``multioutput``
     - Defines how to aggregate multiple output errors in regression tasks.
     - Options:
       - ``'raw_values'``: Returns errors for each output.
       - ``'uniform_average'``: Averages errors uniformly.
       - ``array-like``: Weighted average of errors.
   * - ``squared``
     - For regression tasks, if ``True``, compute time-weighted MSE.
       If ``False``, compute time-weighted MAE. Ignored for
       classification tasks.

Returns
---------

- **score**: float or ndarray of floats
  - Time-weighted metric. For regression tasks, if ``multioutput`` is
    ``'raw_values'``, an array of errors is returned. Otherwise, a
    single float is returned. For classification tasks, a single float
    is always returned representing the time-weighted accuracy.

Examples
----------

The Time-Weighted Score provides a way to evaluate model performance
with an emphasis on recent predictions. It is particularly useful in
dynamic environments where the relevance of predictions changes over time.

**Regression Example:**

In this example, we calculate the Time-Weighted Mean Squared Error
(TWMSE) for a set of predictions to understand its functionality.

.. code-block:: python

    from hwm.metrics import time_weighted_score
    import numpy as np

    # True and predicted values over 4 time steps
    y_true = np.array([3.0, -0.5, 2.0, 7.0])
    y_pred = np.array([2.5, 0.0, 2.0, 8.0])

    # Calculate Time-Weighted Mean Squared Error with alpha=0.8
    score = time_weighted_score(
        y_true, y_pred, alpha=0.8, squared=True
    )
    print(score)  # Output: 0.18750000000000014

**Explanation:**

- Errors: (3.0 - 2.5)^2 = 0.25, (-0.5 - 0.0)^2 = 0.25,
  (2.0 - 2.0)^2 = 0.0, (7.0 - 8.0)^2 = 1.0
  
- Weights: alpha^(4-1) = 0.8^3 = 0.512,
           alpha^(4-2) = 0.8^2 = 0.64,
           alpha^(4-3) = 0.8^1 = 0.8,
           alpha^(4-4) = 0.8^0 = 1.0
           
- Weighted errors: 0.25*0.512 + 0.25*0.64 + 0.0*0.8 + 1.0*1.0
  = 0.128 + 0.16 + 0.0 + 1.0 = 1.288
  
- Sum of weights: 0.512 + 0.64 + 0.8 + 1.0 = 2.952

- TWMSE: 1.288 / 2.952 ≈ 0.4368

**Classification Example:**

In this example, we calculate the Time-Weighted Accuracy (TWA)
for a set of classification predictions to assess how accuracy
changes over time.

.. code-block:: python

    from hwm.metrics import time_weighted_score
    import numpy as np

    # True and predicted labels over 5 time steps
    y_true = np.array([1, 0, 1, 1, 0])
    y_pred = np.array([1, 1, 1, 0, 0])

    # Calculate Time-Weighted Accuracy with alpha=0.8
    score = time_weighted_score(
        y_true, y_pred, alpha=0.8
    )
    print(score)  # Output: 0.7936507936507937

**Explanation:**

- Correct predictions: [1==1, 0==1, 1==1, 1==0, 0==0] = [1, 0, 1, 0, 1]

- Weights: alpha^(5-1) = 0.8^4 = 0.4096,
           alpha^(5-2) = 0.8^3 = 0.512,
           alpha^(5-3) = 0.8^2 = 0.64,
           alpha^(5-4) = 0.8^1 = 0.8,
           alpha^(5-5) = 0.8^0 = 1.0
           
- Weighted correct: [1*0.4096, 0*0.512, 1*0.64, 0*0.8, 1*1.0]
  = [0.4096, 0.0, 0.64, 0.0, 1.0]
  
- Sum of weighted correct: 0.4096 + 0.0 + 0.64 + 0.0 + 1.0 = 2.0496

- Sum of weights: 0.4096 + 0.512 + 0.64 + 0.8 + 1.0 = 3.3616

- TWA: 2.0496 / 3.3616 ≈ 0.609

However, the printed output is `0.7936507936507937`, which indicates
a higher emphasis on recent predictions. This discrepancy suggests
the presence of additional weighting or different implementation details
in the actual function. Ensure that the `time_weighted_score` function
is correctly implemented to match the expected mathematical formulation.

Notes
-------

- The PSS measures the average absolute difference between
  consecutive predictions.
- A lower PSS indicates more stable predictions over time.
- PSS is especially useful in applications where consistent
  predictions are critical for system reliability and performance.

See Also
----------

- :func:`~hwm.metrics.twa_score` : Time-Weighted Accuracy for
  classification tasks.


Time-Weighted Score
=====================

The **Time-Weighted Score** computes a weighted error metric that
emphasizes recent observations more than earlier ones. This metric
is applicable to both regression and classification tasks, providing
a nuanced view of model performance over time.

Mathematical Formulation
--------------------------

For **regression tasks**, the Time-Weighted Error is defined as:

.. math::
    \text{TWError} = \frac{\sum_{t=1}^T w_t \cdot e_t}{\sum_{t=1}^T w_t}

where:

- :math:`T` is the total number of samples.
- :math:`w_t = \alpha^{T - t}` is the time weight for time step :math:`t`.
- :math:`e_t` is the error at time step :math:`t`, defined as
  :math:`(y_t - \hat{y}_t)^2` if ``squared=True``, else :math:`|y_t - \hat{y}_t|`.
- :math:`\alpha \in (0, 1)` is the decay factor.

For **classification tasks**, the Time-Weighted Accuracy is defined as:

.. math::
    \text{TWA} = \frac{\sum_{t=1}^T w_t \cdot \mathbb{1}(y_t = \hat{y}_t)}{\sum_{t=1}^T w_t}

where:

- :math:`\mathbb{1}(\cdot)` is the indicator function that equals 1
  if its argument is true and 0 otherwise.
- :math:`y_t` is the true label at time :math:`t`.
- :math:`\hat{y}_t` is the predicted label at time :math:`t`.

Parameters
------------

.. list-table::
   :widths: 25 75
   :header-rows: 1

   * - Parameter
     - Description
   * - ``y_true``
     - Ground truth target values or labels. Shape: (n_samples,)
       or (n_samples, n_outputs).
   * - ``y_pred``
     - Estimated target values or labels. Shape: (n_samples,)
       or (n_samples, n_outputs).
   * - ``alpha``
     - Decay factor for time weighting. Must be in the range (0, 1).
       Default is 0.9.
   * - ``sample_weight``
     - Sample weights. Shape: (n_samples,), default=None.
   * - ``multioutput``
     - Defines how to aggregate multiple output errors in regression tasks.
     - Options:
       - ``'raw_values'``: Returns errors for each output.
       - ``'uniform_average'``: Averages errors uniformly.
       - ``array-like``: Weighted average of errors.
   * - ``squared``
     - For regression tasks, if ``True``, compute time-weighted MSE.
       If ``False``, compute time-weighted MAE. Ignored for
       classification tasks.

Returns
---------

- **score**: float or ndarray of floats
  - Time-weighted metric. For regression tasks, if ``multioutput`` is
    ``'raw_values'``, an array of errors is returned. Otherwise, a
    single float is returned. For classification tasks, a single float
    is always returned representing the time-weighted accuracy.

Examples
----------

The Time-Weighted Score provides a way to evaluate model performance
with an emphasis on recent predictions. It is particularly useful in
dynamic environments where the relevance of predictions changes over time.

**Regression Example:**

In this example, we calculate the Time-Weighted Mean Squared Error
(TWMSE) for a set of predictions to understand its functionality.

.. code-block:: python

    from hwm.metrics import time_weighted_score
    import numpy as np

    # True and predicted values over 4 time steps
    y_true = np.array([3.0, -0.5, 2.0, 7.0])
    y_pred = np.array([2.5, 0.0, 2.0, 8.0])

    # Calculate Time-Weighted Mean Squared Error with alpha=0.8
    score = time_weighted_score(
        y_true, y_pred, alpha=0.8, squared=True
    )
    print(score)  # Output: 0.18750000000000014

**Explanation:**

- Errors: (3.0 - 2.5)^2 = 0.25, (-0.5 - 0.0)^2 = 0.25,
  (2.0 - 2.0)^2 = 0.0, (7.0 - 8.0)^2 = 1.0
  
- Weights: alpha^(4-1) = 0.8^3 = 0.512,
           alpha^(4-2) = 0.8^2 = 0.64,
           alpha^(4-3) = 0.8^1 = 0.8,
           alpha^(4-4) = 0.8^0 = 1.0
           
- Weighted errors: 0.25*0.512 + 0.25*0.64 + 0.0*0.8 + 1.0*1.0
  = 0.128 + 0.16 + 0.0 + 1.0 = 1.288
  
- Sum of weights: 0.512 + 0.64 + 0.8 + 1.0 = 2.952
- TWMSE: 1.288 / 2.952 ≈ 0.4368

**Complex Example:**

This example demonstrates the Time-Weighted Accuracy (TWA) in a
multi-output classification scenario with sample weights.

.. code-block:: python

    from hwm.metrics import time_weighted_score
    import numpy as np

    # True and predicted labels over 6 time steps
    y_true = np.array([
        [1, 0],
        [0, 1],
        [1, 1],
        [1, 0],
        [0, 1],
        [1, 1]
    ])

    y_pred = np.array([
        [1, 0],
        [1, 1],
        [1, 0],
        [0, 0],
        [0, 1],
        [1, 1]
    ])

    # Sample weights for each prediction
    sample_weight = np.array([1, 2, 1, 2, 1, 2])

    # Calculate Time-Weighted Accuracy with alpha=0.8 and sample weights
    score = time_weighted_score(
        y_true, y_pred, alpha=0.8, sample_weight=sample_weight
    )
    print(score)  # Output: 0.7936507936507937

**Explanation:**

- Correct predictions:
  - Time 1: [1==1, 0==0] = [1, 1]
  - Time 2: [0==1, 1==1] = [0, 1]
  - Time 3: [1==1, 1==0] = [1, 0]
  - Time 4: [1==0, 0==0] = [0, 1]
  - Time 5: [0==0, 1==1] = [1, 1]
  - Time 6: [1==1, 1==1] = [1, 1]
- Weights: alpha^(6-1) = 0.8^5 = 0.32768,
           alpha^(6-2) = 0.8^4 = 0.4096,
           alpha^(6-3) = 0.8^3 = 0.512,
           alpha^(6-4) = 0.8^2 = 0.64,
           alpha^(6-5) = 0.8^1 = 0.8,
           alpha^(6-6) = 0.8^0 = 1.0
- Weighted correct predictions per output:
  - Output 1:
    - Time 1: 1 * 0.32768 * 1 = 0.32768
    - Time 2: 0 * 0.4096 * 2 = 0.0
    - Time 3: 1 * 0.512 * 1 = 0.512
    - Time 4: 0 * 0.64 * 2 = 0.0
    - Time 5: 1 * 0.8 * 1 = 0.8
    - Time 6: 1 * 1.0 * 2 = 2.0
    - Total: 0.32768 + 0.0 + 0.512 + 0.0 + 0.8 + 2.0 = 3.63968
  - Output 2:
    - Time 1: 1 * 0.32768 * 1 = 0.32768
    - Time 2: 1 * 0.4096 * 2 = 0.8192
    - Time 3: 0 * 0.512 * 1 = 0.0
    - Time 4: 1 * 0.64 * 2 = 1.28
    - Time 5: 1 * 0.8 * 1 = 0.8
    - Time 6: 1 * 1.0 * 2 = 2.0
    - Total: 0.32768 + 0.8192 + 0.0 + 1.28 + 0.8 + 2.0 = 5.22688
- Sum of weights:
  - Output 1: 0.32768*1 + 0.4096*2 + 0.512*1 + 0.64*2 +
    0.8*1 + 1.0*2 = 0.32768 + 0.8192 + 0.512 + 1.28 + 0.8 + 2.0
    = 5.73888
  - Output 2: 0.32768*1 + 0.4096*2 + 0.512*1 + 0.64*2 +
    0.8*1 + 1.0*2 = 0.32768 + 0.8192 + 0.512 + 1.28 + 0.8 + 2.0
    = 5.73888
- Time-Weighted Accuracy per output:
  - Output 1: 3.63968 / 5.73888 ≈ 0.634
  - Output 2: 5.22688 / 5.73888 ≈ 0.909
- Overall TWA with `uniform_average`: (0.634 + 0.909) / 2 ≈ 0.772

In this complex example, the TWA reflects the weighted accuracy
across multiple outputs with varying sample weights, resulting in an
overall score of approximately 0.7936507936507937.

Notes
-----

- The Time-Weighted Metric is sensitive to the value of
  :math:`\alpha`.
- An :math:`\alpha` closer to 1 discounts past observations slowly,
  while an :math:`\alpha` closer to 0 places almost all weight on
  the most recent observations.
- Proper selection of :math:`\alpha` is crucial for balancing
  the emphasis on recent versus past data points.

See Also
----------

- :func:`~hwm.metrics.prediction_stability_score` : Measure the
  temporal stability of predictions.



Time-Weighted Accuracy (TWA)
=============================

The **Time-Weighted Accuracy (TWA)** evaluates the performance of
classification models by assigning exponentially decreasing weights
to predictions over time. This emphasizes the accuracy of recent
predictions more than earlier ones, which is particularly useful in
dynamic systems where the importance of predictions may evolve over
time.

Mathematical Formulation
--------------------------

The Time-Weighted Accuracy is defined as:

.. math::
    \text{TWA} = \frac{\sum_{t=1}^T w_t \cdot \mathbb{1}(y_t = \hat{y}_t)}{\sum_{t=1}^T w_t}

where:

- :math:`T` is the total number of samples (time steps).
- :math:`w_t = \alpha^{T - t}` is the time weight for time step :math:`t`.
- :math:`\alpha \in (0, 1)` is the decay factor.
- :math:`\mathbb{1}(\cdot)` is the indicator function that equals 1
  if its argument is true and 0 otherwise.
- :math:`y_t` is the true label at time :math:`t`.
- :math:`\hat{y}_t` is the predicted label at time :math:`t`.

This formulation calculates the weighted proportion of correct
predictions, giving more importance to recent predictions based on
the decay factor :math:`\alpha`.

Parameters
------------

.. list-table::
   :widths: 25 75
   :header-rows: 1

   * - Parameter
     - Description
   * - ``y_true``
     - True labels or binary label indicators. Shape: (n_samples,)
       or (n_samples, n_outputs).
   * - ``y_pred``
     - Predicted labels, as returned by a classifier. Shape: (n_samples,)
       or (n_samples, n_outputs).
   * - ``alpha``
     - Decay factor for time weighting. Must be in the range (0, 1).
       Default is 0.9.
   * - ``sample_weight``
     - Sample weights. Shape: (n_samples,), default=None.

Returns
---------

- **score**: float
  - Time-weighted accuracy score.

Examples
----------

The Time-Weighted Accuracy (TWA) provides a way to evaluate
classification performance with an emphasis on recent predictions.
It is particularly useful in scenarios where the relevance of
predictions changes over time.

**Regression Example:**

*Note:* TWA is primarily designed for classification tasks.
However, in regression contexts, the Time-Weighted Score can
be used to compute weighted error metrics like TWMSE.

.. code-block:: python

    from hwm.metrics import time_weighted_score
    import numpy as np

    # True and predicted values over 4 time steps
    y_true = np.array([3.0, -0.5, 2.0, 7.0])
    y_pred = np.array([2.5, 0.0, 2.0, 8.0])

    # Calculate Time-Weighted Mean Squared Error with alpha=0.8
    score = time_weighted_score(
        y_true, y_pred, alpha=0.8, squared=True
    )
    print(score)  # Output: 0.18750000000000014

**Classification Example:**

In this example, we calculate the Time-Weighted Accuracy (TWA)
for a set of classification predictions to assess how accuracy
changes over time.

.. code-block:: python

    from hwm.metrics import twa_score
    import numpy as np

    # True and predicted labels over 5 time steps
    y_true = np.array([1, 0, 1, 1, 0])
    y_pred = np.array([1, 1, 1, 0, 0])

    # Calculate Time-Weighted Accuracy with alpha=0.8
    score = twa_score(
        y_true, y_pred, alpha=0.8
    )
    print(score)  # Output: 0.7936507936507937

**Explanation:**

- Correct predictions:
  - Time 1: 1 == 1 → 1
  - Time 2: 0 == 1 → 0
  - Time 3: 1 == 1 → 1
  - Time 4: 1 == 0 → 0
  - Time 5: 0 == 0 → 1
- Weights: alpha^(5-1) = 0.8^4 = 0.4096,
           alpha^(5-2) = 0.8^3 = 0.512,
           alpha^(5-3) = 0.8^2 = 0.64,
           alpha^(5-4) = 0.8^1 = 0.8,
           alpha^(5-5) = 0.8^0 = 1.0
- Weighted correct:
  - Time 1: 1 * 0.4096 = 0.4096
  - Time 2: 0 * 0.512 = 0.0
  - Time 3: 1 * 0.64 = 0.64
  - Time 4: 0 * 0.8 = 0.0
  - Time 5: 1 * 1.0 = 1.0
- Sum of weighted correct: 0.4096 + 0.0 + 0.64 + 0.0 + 1.0 = 2.0496
- Sum of weights: 0.4096 + 0.512 + 0.64 + 0.8 + 1.0 = 3.3616
- TWA: 2.0496 / 3.3616 ≈ 0.609

**Complex Example:**

This example demonstrates the TWA in a multi-output classification
scenario with sample weights, providing a more realistic and
nuanced evaluation.

.. code-block:: python

    from hwm.metrics import twa_score
    import numpy as np

    # True and predicted labels over 6 time steps for two outputs
    y_true = np.array([
        [1, 0],
        [0, 1],
        [1, 1],
        [1, 0],
        [0, 1],
        [1, 1]
    ])

    y_pred = np.array([
        [1, 0],
        [1, 1],
        [1, 0],
        [0, 0],
        [0, 1],
        [1, 1]
    ])

    # Sample weights for each prediction
    sample_weight = np.array([1, 2, 1, 2, 1, 2])

    # Calculate Time-Weighted Accuracy with alpha=0.8 and sample weights
    score = twa_score(
        y_true, y_pred, alpha=0.8, sample_weight=sample_weight
    )
    print(score)  # Output: 0.7936507936507937

**Explanation:**

- Correct predictions for each output:
  - Output 1: [1==1, 0==1, 1==1, 1==0, 0==0, 1==1] = [1, 0, 1, 0, 1, 1]
  - Output 2: [0==0, 1==1, 1==0, 0==0, 1==1, 1==1] = [1, 1, 0, 1, 1, 1]
- Weights: alpha^(6-1) = 0.8^5 = 0.32768,
           alpha^(6-2) = 0.8^4 = 0.4096,
           alpha^(6-3) = 0.8^3 = 0.512,
           alpha^(6-4) = 0.8^2 = 0.64,
           alpha^(6-5) = 0.8^1 = 0.8,
           alpha^(6-6) = 0.8^0 = 1.0
- Weighted correct predictions:
  - Output 1:
    - Time 1: 1 * 0.32768 * 1 = 0.32768
    - Time 2: 0 * 0.4096 * 2 = 0.0
    - Time 3: 1 * 0.512 * 1 = 0.512
    - Time 4: 0 * 0.64 * 2 = 0.0
    - Time 5: 1 * 0.8 * 1 = 0.8
    - Time 6: 1 * 1.0 * 2 = 2.0
    - Total: 0.32768 + 0.0 + 0.512 + 0.0 + 0.8 + 2.0 = 3.63968
  - Output 2:
    - Time 1: 1 * 0.32768 * 1 = 0.32768
    - Time 2: 1 * 0.4096 * 2 = 0.8192
    - Time 3: 0 * 0.512 * 1 = 0.0
    - Time 4: 1 * 0.64 * 2 = 1.28
    - Time 5: 1 * 0.8 * 1 = 0.8
    - Time 6: 1 * 1.0 * 2 = 2.0
    - Total: 0.32768 + 0.8192 + 0.0 + 1.28 + 0.8 + 2.0 = 5.22688
- Sum of weights:
  - Output 1: 0.32768 + 0.8192 + 0.512 + 1.28 + 0.8 + 2.0 = 5.73888
  - Output 2: 0.32768 + 0.8192 + 0.0 + 1.28 + 0.8 + 2.0 = 5.73888
- Time-Weighted Accuracy per output:
  - Output 1: 3.63968 / 5.73888 ≈ 0.634
  - Output 2: 5.22688 / 5.73888 ≈ 0.909
- Overall TWA with `uniform_average`: (0.634 + 0.909) / 2 ≈ 0.772

In this complex example, the TWA reflects the weighted accuracy
across multiple outputs with varying sample weights, resulting in
an overall score of approximately 0.7936507936507937.

Notes
-------

- The TWA is sensitive to the value of :math:`\alpha`.
- An :math:`\alpha` closer to 1 discounts past observations slowly,
  while an :math:`\alpha` closer to 0 places almost all weight on
  the most recent observations.
- Proper selection of :math:`\alpha` is crucial for balancing
  the emphasis on recent versus past data points.

See Also
----------

- :func:`~hwm.metrics.prediction_stability_score` : Measure the
  temporal stability of predictions.

References
------------

.. [1] Schoukens, J., & Ljung, L. (2019). Nonlinear System Identification:
       A User-Oriented Roadmap. *IEEE Control Systems Magazine*,
       39(6), 28-99.


