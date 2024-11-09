.. _compat:

=========================
Compatibility Utilities
=========================

This section provides documentation for compatibility utilities in the
:mod:`hwm.compat` module. These utilities ensure smooth operation across
various versions of scikit-learn (sklearn) by handling breaking changes and
deprecated features. The module includes resampling utilities, scorer
functions, and compatibility checks to maintain functionality across different
sklearn versions.

Attributes
----------
.. autoattribute:: SKLEARN_VERSION
    :members:

.. autoattribute:: SKLEARN_LT_0_22
    :members:

.. autoattribute:: SKLEARN_LT_0_23
    :members:

.. autoattribute:: SKLEARN_LT_0_24
    :members:

.. autoattribute:: SKLEARN_LT_1_3
    :members:

.. _Interval:

Interval
==========


.. class:: Interval(*args, inclusive=None, closed='left', **kwargs)

Compatibility wrapper for scikit-learn's `Interval` class to handle
versions that do not include the `inclusive` argument.

Parameters
----------
*args : tuple
    Positional arguments passed to the `Interval` class, typically
    the expected data types and the range boundaries for the validation
    interval.

inclusive : bool, optional
    Specifies whether the interval includes its bounds. Only supported
    in scikit-learn versions that accept the `inclusive` parameter. If
    `True`, the interval includes the bounds. Default is `None` for
    older versions where this argument is not available.

closed : str, optional
    Defines how the interval is closed. Can be "left", "right", "both",
    or "neither". This argument is accepted by both older and newer
    scikit-learn versions. Default is "left" (includes the left bound,
    but excludes the right bound).

**kwargs : dict
    Additional keyword arguments passed to the `Interval` class for
    compatibility, including any additional arguments required by the
    current scikit-learn version.

Returns
---------
Interval
    A compatible `Interval` object based on the scikit-learn version,
    with or without the `inclusive` argument.

Raises
--------
ValueError
    If an unsupported version of scikit-learn is used or the parameters
    are not valid for the given version.

Notes
-------
This class provides a compatibility layer for creating `Interval`
objects in different versions of scikit-learn. The `inclusive` argument
was introduced in newer versions, so this class removes it if not
supported in older versions.

If you are using scikit-learn versions that support the `inclusive`
argument (e.g., version 1.2 or later), it will be included in the call
to `Interval`. Otherwise, the argument will be excluded.

Examples
----------
In newer scikit-learn versions (e.g., >=1.2), you can include the
`inclusive` parameter:

.. code-block:: python
    :linenos:

    from numbers import Integral
    from hwm.compat import Interval

    # Create Interval with inclusive=True
    interval = Interval(Integral, 1, 10, closed="left", inclusive=True)
    print(interval)
    # Output: Interval(Integral, 1, 10, closed='left')

In older versions of scikit-learn that don't support `inclusive`, it
will automatically be removed:

.. code-block:: python
    :linenos:

    from numbers import Integral
    from hwm.compat import Interval

    # Create Interval without inclusive
    interval = Interval(Integral, 1, 10, closed="left")
    print(interval)
    # Output: Interval(Integral, 1, 10, closed='left')

See Also
----------
:class:`sklearn.utils._param_validation.Interval` : Original scikit-learn
    `Interval` class used for parameter validation.

References
----------
.. [1] Pedregosa, F. et al. (2011). "Scikit-learn: Machine Learning in
       Python." *Journal of Machine Learning Research*, 12, 2825-2830.

.. [2] Buitinck, L., Louppe, G., Blondel, M., et al. (2013). "API design
       for machine learning software: experiences from the scikit-learn
       project." *arXiv preprint arXiv:1309.0238*.

.. _get_sgd_loss_param:

get_sgd_loss_param
=====================


.. function:: get_sgd_loss_param()

Get the correct argument of loss parameter for `SGDClassifier` based on
scikit-learn version.

This function determines which loss parameter to use for the
`SGDClassifier` depending on the installed version of scikit-learn.
In versions 0.24 and newer, the loss parameter should be set to
`'log_loss'`. In older versions, it should be set to `'log'`.

Returns
-------
str
    The appropriate loss parameter for the `SGDClassifier`.

Examples
----------
The following examples demonstrate how to use the `get_sgd_loss_param`
function to obtain the correct loss parameter for `SGDClassifier`.

**Basic Example:**

.. code-block:: python
    :linenos:

    from hwm.compat import get_sgd_loss_param
    from sklearn.linear_model import SGDClassifier

    # Get the appropriate loss parameter
    loss_param = get_sgd_loss_param()
    print(loss_param)
    # Output: 'log_loss'  # If using scikit-learn 0.24 or newer

    # Example usage with SGDClassifier
    clf = SGDClassifier(loss=get_sgd_loss_param(), max_iter=1000)
    clf.fit(X_train, y_train)

Notes
-------
This function is useful for maintaining compatibility with different
versions of scikit-learn, ensuring that the model behaves as expected
regardless of the library version being used.

See Also
----------
:class:`sklearn.linear_model.SGDClassifier` : Linear classifier with
    SGD training.

References
------------
.. [1] Scikit-learn. "sklearn.linear_model.SGDClassifier". Available at
       https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.SGDClassifier.html

validate_params
=================

.. _validate_params:

**validate_params**
---------------------

.. function:: validate_params(params, *args, prefer_skip_nested_validation=True, **kwargs)

Compatibility wrapper for scikit-learn's `validate_params` function
to handle versions that require the `prefer_skip_nested_validation` argument,
with a default value that can be overridden by the user.

Parameters
------------

.. list-table::
   :widths: 25 75
   :header-rows: 1

   * - **params**
     - A dictionary that defines the validation rules for the parameters.
       Each key in the dictionary should represent the name of a parameter
       that requires validation, and its associated value should be a list
       of expected types (e.g., ``[int, str]``). The function will validate
       that the parameters passed to the decorated function match the
       specified types.
       
       For example, if `params` is:
       
       .. code-block:: python

           params = {
               'step_name': [str],
               'n_trials': [int]
           }
       
       Then, the `step_name` parameter must be of type `str`, and
       `n_trials` must be of type `int`.
   * - **prefer_skip_nested_validation**
     - If ``True`` (the default), the function will attempt to skip
       nested validation of complex objects (e.g., dictionaries or lists),
       focusing only on the top-level structure. This option can be useful
       for improving performance when validating large, complex objects
       where deep validation is unnecessary.
       
       Set to ``False`` to enable deep validation of nested objects.
   * - **args**
     - Additional positional arguments to pass to `validate_params`.
   * - **kwargs**
     - Additional keyword arguments to pass to `validate_params`. These can
       include options such as `prefer_skip_nested_validation` and other
       custom behavior depending on the context of validation.

Returns
---------
function
    Returns the `validate_params` function with appropriate argument
    handling for scikit-learn's internal parameter validation. This
    function can be used as a decorator to ensure type safety and
    parameter consistency in various machine learning pipelines.

Notes
-------
The `validate_params` function provides a robust way to enforce
type and structure validation on function arguments, especially
in the context of machine learning workflows. By ensuring that
parameters adhere to a predefined structure, the function helps
prevent runtime errors due to unexpected types or invalid argument
configurations.

In the case where a user sets `prefer_skip_nested_validation` to
``True``, the function optimizes the validation process by skipping
nested structures (e.g., dictionaries or lists), focusing only on
validating the top-level parameters. When set to ``False``, a deeper
validation process occurs, checking every element within nested
structures.

The validation process can be represented mathematically as:

.. math::

    V(p_i) = 
    \begin{cases}
    1, & \text{if} \, \text{type}(p_i) \in T(p_i) \\
    0, & \text{otherwise}
    \end{cases}

where :math:`V(p_i)` is the validation function for parameter :math:`p_i`,
and :math:`T(p_i)` represents the set of expected types for :math:`p_i`.
The function returns 1 if the parameter matches the expected type,
otherwise 0.

Examples
----------
The following examples demonstrate how to use the `validate_params`
function to enforce parameter validation in machine learning pipelines.

**Basic Example:**

Ensuring that parameters match expected types using the `validate_params`
decorator.

.. code-block:: python
    :linenos:

    from hwm.compat import validate_params

    @validate_params({
        'step_name': [str],
        'param_grid': [dict],
        'n_trials': [int],
        'eval_metric': [str]
    }, prefer_skip_nested_validation=False)
    def tune_hyperparameters(step_name, param_grid, n_trials, eval_metric):
        print(f"Hyperparameters tuned for step: {step_name}")

    # Correct usage
    tune_hyperparameters(
        step_name='TrainModel', 
        param_grid={'learning_rate': [0.01, 0.1]}, 
        n_trials=5, 
        eval_metric='accuracy'
    )
    # Output: Hyperparameters tuned for step: TrainModel

**Incorrect Usage:**

Attempting to pass parameters with incorrect types will raise a validation error.

.. code-block:: python
    :linenos:

    from hwm.compat import validate_params

    @validate_params({
        'step_name': [str],
        'n_trials': [int]
    })
    def initialize_step(step_name, n_trials):
        pass

    # Incorrect usage: n_trials should be int
    initialize_step(step_name='Init', n_trials='five')
    # Raises: ValueError: Parameter 'n_trials' must be of type int.

See Also
----------
:func:`sklearn.utils.validate_params` : Original scikit-learn function for
    parameter validation. Refer to scikit-learn documentation for more
    detailed information.

References
------------
.. [1] Pedregosa, F. et al. (2011). "Scikit-learn: Machine Learning in
       Python." *Journal of Machine Learning Research*, 12, 2825-2830.

.. [2] Buitinck, L., Louppe, G., Blondel, M., et al. (2013). "API design for
       machine learning software: experiences from the scikit-learn project."
       *arXiv preprint arXiv:1309.0238*.
