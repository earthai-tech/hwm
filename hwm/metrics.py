# -*- coding: utf-8 -*-
#   License: BSD-3-Clause
#   Author: LKouadio <etanoyau@gmail.com>

from sklearn.metrics._regression import _check_reg_targets
from sklearn.utils.validation import check_array, check_consistent_length
from sklearn.utils.multiclass import type_of_target 
import numpy as np 

from .utils.validator import _ensure_y_is_valid

__all__=['prediction_stability_score', 'time_weighted_score', 'twa_score']

def prediction_stability_score(
    y_pred,
    y_true=None,      
    sample_weight=None,
    multioutput='uniform_average'
    ):
    """
    Calculate the Prediction Stability Score (PSS), which assesses the temporal
    stability of predictions across consecutive time steps.

    The Prediction Stability Score is defined as:

    .. math::
        \\text{PSS} = \\frac{1}{T - 1} \\sum_{t=1}^{T - 1}
        \\left| \\hat{y}_{t+1} - \\hat{y}_t \\right|

    where:

    - :math:`T` is the number of time steps.
    - :math:`\\hat{y}_t` is the prediction at time :math:`t`.

    Parameters
    ----------
    y_pred : array-like of shape (n_samples,) or (n_samples, n_outputs)
        Predicted values.

    y_true : None
        Not used, present for API consistency by convention.

    sample_weight : array-like of shape (n_samples,), default=None
        Sample weights. If provided, these are used to weight the differences
        between consecutive predictions.

    multioutput : {'raw_values', 'uniform_average'} or array-like of shape \
            (n_outputs,), default='uniform_average'
        Defines how to aggregate multiple output values.

        - ``'raw_values'`` :
          Returns a full set of scores in case of multioutput input.
        - ``'uniform_average'`` :
          Scores of all outputs are averaged with uniform weight.
        - array-like :
          Weighted average of the output scores.

    Returns
    -------
    score : float or ndarray of floats
        Prediction Stability Score. If ``multioutput`` is ``'raw_values'``,
        then an array of scores is returned. Otherwise, a single float is
        returned.

    Examples
    --------
    >>> from hwm.metrics import prediction_stability_score
    >>> import numpy as np
    >>> y_pred = np.array([3, 3.5, 4, 5, 5.5])
    >>> prediction_stability_score(y_pred)
    0.625

    Notes
    -----
    The Prediction Stability Score measures the average absolute difference
    between consecutive predictions. A lower score indicates more stable
    predictions over time.

    See Also
    --------
    twa_score : Time-weighted accuracy for classification tasks.

    References
    ----------
    .. [1] Schoukens, J., & Ljung, L. (2019). Nonlinear System Identification:
           A User-Oriented Roadmap. IEEE Control Systems Magazine, 39(6),
           28-99.
    """
    # Ensure y_pred is a numpy array
    y_pred = check_array(y_pred, ensure_2d=False, dtype=None)

    # For multi-output regression, y_pred can be 2D
    if y_pred.ndim == 1:
        # Reshape to 2D array with one output
        y_pred = y_pred.reshape(-1, 1)

    # Number of samples and outputs
    n_samples, n_outputs = y_pred.shape

    # Compute absolute differences between consecutive predictions
    # diff shape: (n_samples - 1, n_outputs)
    diff = np.abs(y_pred[1:] - y_pred[:-1])

    # Adjust sample_weight for differences
    if sample_weight is not None:
        sample_weight = np.asarray(sample_weight)
        if sample_weight.ndim > 1:
            sample_weight = sample_weight.squeeze()
        # Ensure sample_weight has correct length
        if len(sample_weight) != n_samples:
            raise ValueError(
                "sample_weight must have the same length as y_pred"
            )
        # Use weights from t=1 to T-1
        sample_weight = sample_weight[1:]

    # Compute mean absolute difference per output
    if sample_weight is not None:
        # Weighted average over time steps
        score = np.average(
            diff,
            weights=sample_weight[:, np.newaxis],
            axis=0
        )
    else:
        # Unweighted average over time steps
        score = np.mean(diff, axis=0)

    # Handle multioutput parameter
    if multioutput == 'raw_values':
        # Return array of shape (n_outputs,)
        return score
    elif multioutput == 'uniform_average':
        # Average over outputs
        return np.mean(score)
    elif isinstance(multioutput, (list, np.ndarray)):
        # Weighted average over outputs
        output_weights = np.asarray(multioutput)
        if output_weights.shape[0] != n_outputs:
            raise ValueError(
                "output_weights must have the same length as n_outputs"
            )
        return np.average(score, weights=output_weights)
    else:
        raise ValueError("Invalid value for multioutput parameter.")

def time_weighted_score(
    y_true,
    y_pred,
    *,
    alpha=0.9,
    sample_weight=None,
    multioutput='uniform_average',
    squared=True
    ):
    """
    Compute the Time-Weighted Metric for regression and classification tasks.

    For **regression tasks**, this function calculates the time-weighted mean
    squared error (MSE) or mean absolute error (MAE) between the true and
    predicted values. For **classification tasks**, it computes the time-weighted
    Time-Weighted Accuracy (TWA). It assigns exponentially decreasing weights
    over time, emphasizing recent observations more than earlier ones.

    The time-weighted metric is defined differently for regression and
    classification:

    **Regression:**

    .. math::
        \\text{TWError} = \\frac{\\sum_{t=1}^T w_t \\cdot e_t}{\\sum_{t=1}^T w_t}

    where:

    - :math:`T` is the total number of samples (time steps).
    - :math:`w_t = \\alpha^{T - t}` is the time weight for time step :math:`t`.
    - :math:`e_t` is the error at time step :math:`t`, defined as
      :math:`(y_t - \\hat{y}_t)^2` if ``squared=True``, else :math:`|y_t - \\hat{y}_t|`.
    - :math:`\\alpha \\in (0, 1)` is the decay factor.

    **Classification:**

    .. math::
        \\text{TWA} = \\frac{\\sum_{t=1}^T w_t \\cdot \\mathbb{1}(y_t = \\hat{y}_t)}{\\sum_{t=1}^T w_t}

    where:

    - :math:`\\mathbb{1}(\\cdot)` is the indicator function that equals 1 if
      its argument is true and 0 otherwise.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,) or (n_samples, n_outputs)
        Ground truth (correct) target values or labels.

    y_pred : array-like of shape (n_samples,) or (n_samples, n_outputs)
        Estimated target values or labels.

    alpha : float, default=0.9
        Decay factor for time weighting. Must be in the range (0, 1).
        A higher value places more emphasis on recent observations.

    sample_weight : array-like of shape (n_samples,), default=None
        Sample weights. If provided, combines with time weights to compute a
        weighted metric.

    multioutput : {'raw_values', 'uniform_average'} or array-like of shape \
            (n_outputs,), default='uniform_average'
        Defines how to aggregate multiple output errors in regression tasks.

        - ``'raw_values'`` :
          Returns a full set of errors in case of multioutput input.
        - ``'uniform_average'`` :
          Errors of all outputs are averaged with uniform weight.
        - array-like :
          Weighted average of the output errors.

        For classification tasks, this parameter is ignored.

    squared : bool, default=True
        For regression tasks, if True, compute time-weighted mean squared error.
        If False, compute time-weighted mean absolute error.

        For classification tasks, this parameter is ignored.

    Returns
    -------
    score : float or ndarray of floats
        Time-weighted metric. For regression tasks, if ``multioutput`` is
        ``'raw_values'``, then an array of errors is returned. Otherwise, a
        single float is returned. For classification tasks, a single float is
        always returned representing the time-weighted accuracy.

    Examples
    --------
    **Regression Example:**

    >>> from hwm.metrics import time_weighted_score
    >>> import numpy as np
    >>> y_true = np.array([3.0, -0.5, 2.0, 7.0])
    >>> y_pred = np.array([2.5, 0.0, 2.0, 8.0])
    >>> time_weighted_score(y_true, y_pred, alpha=0.8, squared=True)
    0.18750000000000014

    **Classification Example:**

    >>> from hwm.metrics import time_weighted_score
    >>> import numpy as np
    >>> y_true = np.array([1, 0, 1, 1, 0])
    >>> y_pred = np.array([1, 1, 1, 0, 0])
    >>> time_weighted_score(y_true, y_pred, alpha=0.8)
    0.7936507936507937

    Notes
    -----
    The Time-Weighted Metric is sensitive to the value of :math:`\\alpha`.
    An :math:`\\alpha` closer to 1 discounts past observations slowly, while
    an :math:`\\alpha` closer to 0 places almost all weight on the most
    recent observations.

    See Also
    --------
    prediction_stability_score : Measure the temporal stability of predictions.

    References
    ----------
    .. [1] Schoukens, J., & Ljung, L. (2019). Nonlinear System Identification:
           A User-Oriented Roadmap. IEEE Control Systems Magazine, 39(6), 28-99.
    """
    # Validate input arrays
    y_true = check_array(y_true, ensure_2d=False, dtype=None)
    y_pred = check_array(y_pred, ensure_2d=False, dtype=None)
    check_consistent_length(y_true, y_pred, sample_weight)

    # Determine the type of target
    y_type = type_of_target(y_true)

    # Validate sample_weight
    if sample_weight is not None:
        sample_weight = np.asarray(sample_weight)
        if sample_weight.ndim > 1:
            sample_weight = sample_weight.squeeze()
        if len(sample_weight) != len(y_true):
            raise ValueError(
                "sample_weight must have the same length as y_true"
            )

    # Compute time weights: w_t = alpha^(T - t - 1)
    T = len(y_true)
    t = np.arange(T)
    weights = alpha ** (T - t - 1)

    # Combine time weights with sample weights if provided
    if sample_weight is not None:
        weights = weights * sample_weight
        

    valid_types =[
        'binary', 'multiclass', 
        'multilabel-indicator', 
        'multiclass-multioutput', 
        'continuous', 
        'continuous-multioutput'
    ]
    if y_type not in valid_types: 
        raise ValueError(
            f"Target `y` must be one of the valid types: {valid_types}")
    
    # Handle different types of targets
    if y_type in ('continuous', 'continuous-multioutput'):
        # Regression task
        # Ensure y_true and y_pred are numpy arrays
        y_true, y_pred = _ensure_y_is_valid(y_true, y_pred, multi_output=True )
  
        # For multioutput regression, handle multioutput parameter
        y_type, y_true, y_pred, multioutput = _check_reg_targets(
            y_true, y_pred, multioutput)
        
        # Compute errors
        if squared:
            errors = (y_true - y_pred) ** 2
        else:
            errors = np.abs(y_true - y_pred)

        # Multiply errors by weights
        if errors.ndim == 1:
            # Reshape to (n_samples, 1) for consistent processing
            errors = errors.reshape(-1, 1)
        weighted_errors = weights[:, np.newaxis] * errors

        # Sum over samples
        numerator = np.sum(weighted_errors, axis=0)
        # Sum of weights
        denominator = np.sum(weights)

        # Compute time-weighted error per output
        error_per_output = numerator / denominator

        # Handle multioutput parameter
        if multioutput == 'raw_values':
            # Return array of shape (n_outputs,)
            return error_per_output
        elif multioutput == 'uniform_average':
            # Average over outputs
            return np.mean(error_per_output)
        elif isinstance(multioutput, (list, np.ndarray)):
            # Weighted average over outputs
            output_weights = np.asarray(multioutput)
            if output_weights.shape[0] != error_per_output.shape[0]:
                raise ValueError(
                    "output_weights must have the same length as n_outputs"
                )
            return np.average(error_per_output, weights=output_weights)
        else:
            raise ValueError("Invalid value for multioutput parameter.")

    elif y_type in valid_types[:-2]:
        # Classification task
        # Compute correct predictions (1 if correct, 0 if incorrect)
        correct = (y_true == y_pred).astype(int)

        if y_type in ('multilabel-indicator', 'multiclass-multioutput'):
            # Ensure y_true and y_pred are 2D arrays
            if y_true.ndim == 1:
                y_true = y_true.reshape(-1, 1)
            if y_pred.ndim == 1:
                y_pred = y_pred.reshape(-1, 1)
            # Multiply weights with correct predictions
            weighted_correct = weights[:, np.newaxis] * correct
            # Sum over samples for each label
            numerator = np.sum(weighted_correct, axis=0)
            # Sum of weights
            denominator = np.sum(weights)
            # Calculate accuracy per label
            score_per_label = numerator / denominator
            # Average over labels to get overall score
            score = np.mean(score_per_label)
            return score
        else:
            # Multiply weights with correct predictions
            weighted_correct = weights * correct
            # Sum over samples
            numerator = np.sum(weighted_correct)
            # Sum of weights
            denominator = np.sum(weights)
            # Calculate time-weighted accuracy
            score = numerator / denominator
            return score


def twa_score(
    y_true,
    y_pred,
    *,
    alpha=0.9,
    sample_weight=None
    ):
    """
    Compute the Time-Weighted Accuracy (TWA) for classification tasks.

    The Time-Weighted Accuracy assigns exponentially decreasing weights
    to predictions over time, emphasizing recent predictions more than
    earlier ones. This is particularly useful in dynamic systems where
    the importance of correct predictions may change over time.

    The TWA is defined as:

    .. math::
        \\text{TWA} = \\frac{\\sum_{t=1}^T w_t \\cdot \\mathbb{1}(y_t = \\hat{y}_t)}{\\sum_{t=1}^T w_t}

    where:

    - :math:`T` is the total number of samples (time steps).
    - :math:`w_t = \\alpha^{T - t}` is the time weight for time step :math:`t`.
    - :math:`\\alpha \\in (0, 1)` is the decay factor.
    - :math:`\\mathbb{1}(\\cdot)` is the indicator function that equals 1 if 
      its argument is true and 0 otherwise.
    - :math:`y_t` is the true label at time :math:`t`.
    - :math:`\\hat{y}_t` is the predicted label at time :math:`t`.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,) or (n_samples, n_outputs)
        True labels or binary label indicators.

    y_pred : array-like of shape (n_samples,) or (n_samples, n_outputs)
        Predicted labels, as returned by a classifier.

    alpha : float, default=0.9
        Decay factor for time weighting. Must be in the range (0, 1).
        A higher value places more emphasis on recent predictions.

    sample_weight : array-like of shape (n_samples,), default=None
        Sample weights. If provided, combines with time weights to
        compute a weighted accuracy.

    Returns
    -------
    score : float
        Time-weighted accuracy score.

    Examples
    --------
    >>> from hwm.metrics import twa_score
    >>> import numpy as np
    >>> y_true = np.array([1, 0, 1, 1, 0])
    >>> y_pred = np.array([1, 1, 1, 0, 0])
    >>> twa_score(y_true, y_pred, alpha=0.8)
    0.7936507936507937

    Notes
    -----
    The TWA is sensitive to the value of :math:`\\alpha`. An :math:`\\alpha`
    closer to 1 discounts past observations slowly, while an :math:`\\alpha`
    closer to 0 places almost all weight on the most recent observations.

    See Also
    --------
    prediction_stability_score : Measure the temporal stability of predictions.

    References
    ----------
    .. [1] Schoukens, J., & Ljung, L. (2019). Nonlinear System Identification:
           A User-Oriented Roadmap. IEEE Control Systems Magazine, 39(6), 28-99.
    """
    # Validate input arrays
    y_true = check_array(y_true, ensure_2d=False, dtype=None)
    y_pred = check_array(y_pred, ensure_2d=False, dtype=None)
    check_consistent_length(y_true, y_pred)

    # Determine the type of target (e.g., binary, multiclass, multilabel)
    y_type = type_of_target(y_true)

    # Validate sample_weight
    if sample_weight is not None:
        sample_weight = np.asarray(sample_weight)
        if sample_weight.ndim > 1:
            sample_weight = sample_weight.squeeze()
        if len(sample_weight) != len(y_true):
            raise ValueError(
                "sample_weight must have the same length as y_true"
            )

    # Compute time weights: w_t = alpha^(T - t)
    T = len(y_true)
    t = np.arange(T)
    weights = alpha ** (T - t - 1)

    # Combine time weights with sample weights if provided
    if sample_weight is not None:
        weights = weights * sample_weight

    valid_types={
            'binary', 'multiclass', 'multilabel-indicator',
            'multiclass-multioutput'}
    
    if y_type not in valid_types: 
        raise ValueError(
            f"Target `y` must be one of the valid types: {valid_types}")
    
    # Handle different types of classification targets
    if y_type in ('binary', 'multiclass'):
        # Compute correct predictions (1 if correct, 0 if incorrect)
        correct = (y_true == y_pred).astype(int)
        # Compute weighted sum of correct predictions
        numerator = np.sum(weights * correct)
        # Sum of weights
        denominator = np.sum(weights)
        # Calculate time-weighted accuracy
        score = numerator / denominator
        return score
    elif y_type in ('multilabel-indicator', 'multiclass-multioutput'):
        # Ensure y_true and y_pred are 2D arrays
        if y_true.ndim == 1:
            y_true = y_true.reshape(-1, 1)
        if y_pred.ndim == 1:
            y_pred = y_pred.reshape(-1, 1)
        # Compute correct predictions per label
        correct = (y_true == y_pred).astype(int)
        # Multiply weights with correct predictions
        weighted_correct = weights[:, np.newaxis] * correct
        # Sum over samples for each label
        numerator = np.sum(weighted_correct, axis=0)
        # Sum of weights
        denominator = np.sum(weights)
        # Calculate accuracy per label
        score_per_label = numerator / denominator
        # Average over labels to get overall score
        score = np.mean(score_per_label)
        return score
