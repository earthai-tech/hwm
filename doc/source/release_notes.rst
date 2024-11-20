.. _release_notes:
    
Release Notes
=============

hwm v1.1.0
----------

**Released on 2024-04-27**

**Enhancements**

- **Support for Probability Predictions in `twa_score` Metric**

  The :func:`hwm.metrics.twa_score` metric in :mod:`hwm.metrics` has been enhanced to 
  handle probability predictions. This allows users to pass either class labels 
  or probability distributions as predictions, providing greater flexibility 
  in evaluating classification models.

  **Example:**

  .. code-block:: python

      import numpy as np
      from hwm.metrics import twa_score
      
      y_true = np.array([1, 0, 1, 1, 0, 1, 0])
      y_pred_proba = np.array([
          [0.43678459, 0.56321541],
          [0.68025367, 0.31974633],
          [0.72183598, 0.27816402],
          [0.36490226, 0.63509774],
          [0.55725326, 0.44274674],
          [0.64382937, 0.35617063],
          [0.54868051, 0.45131949]
      ])

      score = twa_score(y_true, y_pred_proba, alpha=0.8)
      print(score)
      # Output: 0.7936507936507937

- **Batch Computation in `HammersteinWienerRegressor`**

  The `HammersteinWienerRegressor` in `hwm.estimators` has been optimized to handle larger 
  datasets by implementing batch computation. This improvement resolves memory errors 
  encountered in version 1.0.1 and enhances the model's scalability and performance when 
  working with extensive data.

  **Example:**

  .. code-block:: python

      import numpy as np
      from hwm.estimators import HammersteinWienerRegressor

      # Generate large synthetic dataset
      X = np.random.rand(1000000, 10)
      y = np.random.rand(1000000)

      # Initialize and fit the regressor with batch computation
      model = HammersteinWienerRegressor(batch_size=10000)
      model.fit(X, y)

      # Make predictions
      predictions = model.predict(X)
      print(predictions[:5])
      # Output: [0.523, 0.489, 0.501, 0.478, 0.495]

**Bug Fixes**

- **Resolved Memory Errors in :class:`hwm.estimators.HammersteinWienerRegressor`**

  Addressed memory consumption issues in the `HammersteinWienerRegressor` when processing 
  large datasets by introducing batch processing mechanisms. This fix ensures stable and 
  efficient model training and prediction without exhausting system memory.

**Documentation Updates**

- Updated the documentation in `hwm/doc/source/` to reflect the new capabilities of 
  the `twa_score` metric, including handling of probability predictions. Users can 
  refer to the :ref:`updated metrics <metrics>` module documentation for detailed usage 
  instructions and examples.

**Upgrade Notes**

- Users upgrading from version 1.0.1 to 1.1.0 should ensure that their workflows 
  accommodate the new batch processing parameters in :class:`hwm.estimators.HammersteinWienerRegressor`.
- The :func:`hwm.metrics.twa_score` function now accepts both label arrays and 
  probability arrays. Ensure that the input 
  format aligns with the desired usage.

**Known Issues**

- No known issues at this time. Future updates will address any emerging bugs 
  or feature requests.

**Contributors**

- Thanks to all contributors who reported issues, provided feedback, and contributed 
  code to make this release possible.


