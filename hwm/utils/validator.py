# -*- coding: utf-8 -*-
#   License: BSD-3-Clause
#   Author: LKouadio <etanoyau@gmail.com>

import inspect 
import warnings 
import scipy.sparse as sp

from datetime import datetime
import numpy as np
import pandas as pd 
from ._array_api import get_namespace, _asarray_with_order

from sklearn.utils.validation import ( 
    check_consistent_length, 
    check_array, 
    check_X_y, 
    ) 
 
__all__= [
    'validate_dates', 
    'check_consistent_length', 
    'check_array', 
    'check_X_y', 
    'check_is_fitted', 
    "validate_batch_size",
    'array_to_frame', 
    'assert_all_finite',
    'check_is_fitted',
    'check_y',
    'convert_array_to_pandas',
    'validate_batch_size',
    'validate_dates', 
    'validate_length_range',
]


def validate_dates(
        start_date, end_date, return_as_date_str=False, date_format="%Y-%m-%d"):
    """
    Validates and parses start and end years/dates, with options for output formatting.

    This function ensures the validity of provided start and end years or dates, checks
    if they fall within a reasonable range, and allows the option to return the validated
    years or dates in a specified string format.

    Parameters
    ----------
    start_date : int, float, or str
        The starting year or date. Can be an integer, float (converted to integer),
        or string in "YYYY" or "YYYY-MM-DD" format.
    end_date : int, float, or str
        The ending year or date, with the same format options as `start_date`.
    return_as_date_str : bool, optional
        If True, returns the start and end dates as strings in the specified format.
        Default is False, returning years as integers.
    date_format : str, optional
        The format string for output dates if `return_as_date_str` is True.
        Default format is "%Y-%m-%d".

    Returns
    -------
    tuple
        A tuple of two elements, either integers (years) or strings (formatted dates),
        representing the validated start and end years or dates.

    Raises
    ------
    ValueError
        If the input years or dates are invalid, out of the acceptable range,
        or if the start year/date does not precede the end year/date.

    Examples
    --------
    >>> from hwm.utils.validator import validate_dates
    >>> validate_dates(1999, 2001)
    (1999, 2001)

    >>> validate_dates("1999/01/01", "2001/12/31", return_as_date_str=True)
    ('1999-01-01', '2001-12-31')

    >>> validate_dates("1999", "1998")
    ValueError: The start date/time must precede the end date/time.

    >>> validate_years("1899", "2001")
    ValueError: Years must be within the valid range: 1900 to [current year].

    Notes
    -----
    The function supports flexible input formats for years and dates, including
    handling both slash "/" and dash "-" separators in date strings. It enforces
    logical and chronological order between start and end inputs and allows
    customization of the output format for date strings.
    """
    def parse_year_input(year_input):
        if isinstance(year_input, (int, float)):
            return datetime(int(year_input), 1, 1)
        elif isinstance(year_input, str):
            year_input = year_input.replace("/", "-")
            try:
                return  datetime.strptime(year_input, date_format)
            except ValueError:
                try: 
                    # Fallback to parsing as year only
                    return datetime(int(year_input), 1, 1)
                except TypeError as type_err: 
                    raise TypeError (
                        "Expected int, float, or str for"
                        f" year, got {type(year_input)}."
                        ) from type_err 
                except ValueError as value_err : 
                    raise ValueError (
                        "Check your date data. For datetime value, set `date_format`"
                        " to '%Y-%m-%d %H:%M:%S'") from value_err
        raise TypeError(f"Invalid input '{year_input}'."
                        " Expected format: YYYY or YYYY-MM-DD.")

    start_date, end_date = map(parse_year_input, [start_date, end_date])

    if start_date >= end_date:
        raise ValueError("Start date/time must be earlier than end date/time.")

    if return_as_date_str:
        return start_date.strftime(date_format), end_date.strftime(date_format)

    current_year = datetime.now().year
    for year in (start_date.year, end_date.year):
        if not 1900 <= year <= current_year:
            raise ValueError(f"Year {year} is out of the valid"
                             f" range: 1900 to {current_year}.")

    # Additional validation for non-string return format
    if ( 
        start_date.year == end_date.year 
        and start_date != end_date 
        and not return_as_date_str
        ):
        raise ValueError(
            "Start and end dates are within the same year but not the same date. "
            "Consider using return_as_date_str=True or providing specific dates.")

    return start_date.year, end_date.year

def validate_batch_size(
        batch_size, n_samples, min_batch_size=1, max_batch_size=None):
    """
    Validate the batch size against the number of samples.

    This function checks whether the provided `batch_size` is appropriate 
    given the total number of samples `n_samples`. It ensures that the batch 
    size meets specified minimum and maximum limits, raising appropriate 
    errors if any constraints are violated.

    Parameters
    ----------
    batch_size : int
        The size of each batch. This must be a positive integer, as batches 
        must contain at least one sample. A ValueError will be raised if this 
        value is less than the minimum allowed batch size or exceeds the 
        total number of samples.

    n_samples : int
        The total number of samples in the dataset. This value must be 
        positive and greater than or equal to the `batch_size`. If `batch_size` 
        is greater than `n_samples`, a ValueError is raised.

    min_batch_size : int, optional
        The minimum allowed batch size (default is 1). This parameter defines 
        the smallest permissible batch size. A ValueError will be raised if 
        the `batch_size` is less than this value.

    max_batch_size : int, optional
        The maximum allowed batch size (default is None, meaning no upper limit). 
        This parameter can be used to restrict the size of the batch to a 
        specified maximum value. If `max_batch_size` is provided, a ValueError 
        will be raised if the `batch_size` exceeds this limit.
    
    Return 
    ------
        batch_size: Validated number of batch size 
    
    Raises
    ------
    ValueError
        If the `batch_size` is less than the `min_batch_size`, greater than the 
        `n_samples`, or exceeds the `max_batch_size` if specified. Additionally, 
        if `batch_size` is not a positive integer, a ValueError is raised.

    Notes
    ------
    Let `B` represent the `batch_size` and `N` represent the `n_samples`. 
    The validation can be expressed mathematically as:

    .. math::
        \text{If } B < \text{min\_batch\_size} \text{ or } B > N \text{ or } B > \text{max\_batch\_size}:
        \quad \text{raise ValueError}

    Examples
    --------
    >>> from hwm.utils.validators import validate_batch_size
    >>> validate_batch_size(32, 100)  # Valid case
    >>> validate_batch_size(0, 100)  # Raises ValueError
    >>> validate_batch_size(150, 100)  # Raises ValueError
    >>> validate_batch_size(32, 100, max_batch_size=32)  # Valid case
    >>> validate_batch_size(40, 100, max_batch_size=32)  # Raises ValueError

    Notes
    -----
    This function is essential for managing data batching in machine learning 
    workflows, where improper batch sizes can lead to inefficient training or 
    runtime errors.

    See Also
    --------
    - Other validation functions in the `hwm.utils.validators` module
    - Documentation on batch processing in machine learning frameworks

    References
    ----------
    .. [1] Goodfellow, I., Bengio, Y., & Courville, A. (2016). Deep Learning. 
       MIT Press. https://www.deeplearningbook.org/
    """
    n_samples = validate_positive_integer(n_samples, "N-samples")
    
    # Check if batch_size is a positive integer
    batch_size = validate_positive_integer(batch_size, "Batch size", msg= ( 
        f"Batch size must be a positive integer. Given: {batch_size}.")
        )
    
    # Check if batch_size meets the minimum requirement
    if batch_size < min_batch_size:
        raise ValueError(
            f"Batch size ({batch_size}) cannot be less than"
            f" the minimum allowed ({min_batch_size})."
        )

    # Check if batch_size exceeds the maximum limit, if provided
    if max_batch_size is not None and batch_size > max_batch_size:
        raise ValueError(
            f"Batch size ({batch_size}) cannot exceed"
            f" the maximum allowed ({max_batch_size})."
        )

    # Check if batch_size exceeds the total number of samples
    if batch_size > n_samples:
        raise ValueError(
            f"Batch size ({batch_size}) cannot exceed"
            f" number of samples ({n_samples})."
        )
    return batch_size

def validate_positive_integer(
        value, variable_name, include_zero=False, round_float=None, 
        msg=None
        ):
    """
    Validates whether the given value is a positive integer or zero based 
    on the parameter and rounds float values according to the specified method.

    Parameters:
    ----------
    value : int or float
        The value to validate.
    variable_name : str
        The name of the variable for error message purposes.
    include_zero : bool, optional
        If True, zero is considered a valid value. Default is False.
    round_float : str, optional
        If "ceil", rounds up float values; if "floor", rounds down float values;
        if None, truncates float values to the nearest whole number towards zero.
    msg: str, optional, 
        Error message when checking for proper type failed.
    Returns:
    -------
    int
        The validated value converted to an integer.

    Raises:
    ------
    ValueError
        If the value is not a positive integer or zero (based on `include_zero`),
        or if the `round_float` parameter is improperly specified.
    """
    import math
    
    # Determine the minimum acceptable value
    min_value = 0 if include_zero else 1

    if isinstance(value, str):
         # Try to convert it if possible
         try:
             value = int(value)
         except ValueError:
             # Raise a nice informative error message
             raise ValueError(f"Value {value} is not convertible to an integer.")

    # Check for proper type and round if necessary
    if not isinstance(value, (int, float, np.integer, np.floating)):
        msg = msg or f"{variable_name} must be an integer or float. Got {value}"
        raise ValueError(msg)
        
    if isinstance(value, float):
        if round_float == "ceil":
            value = math.ceil(value)
        elif round_float == "floor":
            value = math.floor(value)
        elif round_float is None:
            value = int(value)
        else:
            raise ValueError(f"Invalid rounding method '{round_float}'."
                             " Choose 'ceil', 'floor', or None.")
    # if isinstance(value, float) and not value.is_integer():
    #     raise ValueError(f"{variable_name} must be a whole number, got {value}.")
    if value < min_value:
        condition = "a non-negative integer" if include_zero else "a positive integer"
        raise ValueError(f"{variable_name} must be {condition}, got {value}.")

    return int(value)

def validate_length_range(length_range, sorted_values=True, param_name=None):
    """
    Validates the review length range ensuring it's a tuple with two integers 
    where the first value is less than the second.

    Parameters:
    ----------
    length_range : tuple
        A tuple containing two integers that represent the minimum and maximum
        lengths of reviews.
    sorted_values: bool, default=True 
        If True, the function expects the input length range to be sorted in 
        ascending order and will automatically sort it if not. If False, the 
        input length range is not expected to be sorted, and it will remain 
        as provided.
    param_name : str, optional
        The name of the parameter being validated. If None, the default name 
        'length_range' will be used in error messages.
        
    Returns
    -------
    tuple
        The validated length range.

    Raise
    ------
    ValueError
        If the length range does not meet the requirements.
        
    Examples 
    --------
    >>> from hwm.utils.validator import validate_length_range
    >>> validate_length_range ( (202, 25) )
    (25, 202)
    >>> validate_length_range ( (202,) )
    ValueError: length_range must be a tuple with two elements.
    """
    param_name = param_name or "length_range" 
    if not isinstance(length_range, ( list, tuple) ) or len(length_range) != 2:
        raise ValueError(f"{param_name} must be a tuple with two elements.")

    min_length, max_length = length_range

    if not all(isinstance(x, ( float, int, np.integer, np.floating)
                          ) for x in length_range):
        raise ValueError(f"Both elements in {param_name} must be integers.")
    
    if sorted_values: 
        length_range  = tuple  (sorted ( [min_length, max_length] )) 
        if length_range[0] >= length_range[1]:
            raise ValueError(
                f"The first element in {param_name} must be less than the second.")
    else : 
        length_range = tuple ([min_length, max_length] )
  
    return length_range 

def _ensure_y_is_valid(y_true, y_pred, **kwargs):
    """
    Validates that the true and predicted target arrays are suitable for further
    processing. This involves ensuring that both arrays are non-empty, of the
    same length, and meet any additional criteria specified by keyword arguments.

    Parameters
    ----------
    y_true : array-like
        The true target values.
    y_pred : array-like
        The predicted target values.
    **kwargs : dict
        Additional keyword arguments to pass to the check_y function for any
        extra validation criteria.

    Returns
    -------
    y_true : array-like
        Validated true target values.
    y_pred : array-like
        Validated predicted target values.

    Raises
    ------
    ValueError
        If the validation checks fail, indicating that the input arrays do not
        meet the required criteria for processing.

    Examples
    --------
    Suppose `check_y` validates that the input is a non-empty numpy array and
    `check_consistent_length` ensures the arrays have the same number of elements.
    Then, usage could be as follows:

    >>> y_true = np.array([1, 2, 3])
    >>> y_pred = np.array([1.1, 2.1, 3.1])
    >>> y_true_valid, y_pred_valid = _ensure_y_is_valid(y_true, y_pred)
    >>> print(y_true_valid, y_pred_valid)
    [1 2 3] [1.1 2.1 3.1]
    """
    # Convert y_true and y_pred to numpy arrays if they are not already
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    
    # Ensure individual array validity
    y_true = check_y(y_true, **kwargs)
    y_pred = check_y(y_pred, **kwargs)

    # Check if the arrays have consistent lengths
    check_consistent_length(y_true, y_pred)

    return y_true, y_pred


def convert_array_to_pandas(X, *, to_frame=False, columns=None, input_name='X'):
    """
    Converts an array-like object to a pandas DataFrame or Series, applying
    provided column names or series name.

    Parameters
    ----------
    X : array-like
        The array to convert to a DataFrame or Series.
    to_frame : bool, default=False
        If True, converts the array to a DataFrame. Otherwise, returns the array unchanged.
    columns : str or list of str, optional
        Name(s) for the columns of the resulting DataFrame or the name of the Series.
    input_name : str, default='X'
        The name of the input variable; used in constructing error messages.

    Returns
    -------
    pd.DataFrame or pd.Series
        The converted DataFrame or Series. If `to_frame` is False, returns `X` unchanged.
    columns : str or list of str
        The column names of the DataFrame or the name of the Series, if applicable.

    Raises
    ------
    TypeError
        If `X` is not array-like or if `columns` is neither a string nor a list of strings.
    ValueError
        If the conversion to DataFrame is requested but `columns` is not provided,
        or if the length of `columns` does not match the number of columns in `X`.
    """
    # Check if the input is string, which is a common mistake
    if isinstance(X, str):
        raise TypeError(f"The parameter '{input_name}' should be an array-like"
                        " or sparse matrix, but a string was passed.")
    
    # Validate the type of X
    if not (hasattr(X, '__array__') or isinstance(
            X, (np.ndarray, pd.Series, list)) or sp.issparse(X)):
        raise TypeError(f"The parameter '{input_name}' should be array-like"
                        f" or a sparse matrix. Received: {type(X).__name__!r}")
    
    # Preserve existing DataFrame or Series column names
    if hasattr(X, 'columns'):
        columns = X.columns
    elif hasattr(X, 'name'):
        columns = X.name

    if to_frame and not sp.issparse(X):
        if columns is None:
            raise ValueError("Columns must be provided for DataFrame conversion.")

        # Ensure columns is list-like for DataFrame conversion, single string for Series
        if isinstance(columns, str):
            columns = [columns]

        if not hasattr(columns, '__len__') or isinstance(columns, str):
            raise TypeError(f"Columns for {input_name} must be a list or a single string.")

        # Convert to Series or DataFrame based on dimensionality
        if X.ndim == 1 or len(X) == len(columns) == 1:  # 1D array or single-column DataFrame
            X = pd.Series(X, name=columns[0])
        elif X.ndim == 2:  # 2D array to DataFrame
            if X.shape[1] != len(columns):
                raise ValueError(f"Shape of passed values is {X.shape},"
                                 f" but columns implied {len(columns)}")
            X = pd.DataFrame(X, columns=columns)
        else:
            raise ValueError(f"{input_name} cannot be converted to DataFrame with given columns.")

    return X, columns


def _check_estimator_name(estimator):
    if estimator is not None:
        if isinstance(estimator, str):
            return estimator
        else:
            return estimator.__class__.__name__
    return None

def check_y(y, 
    multi_output=False, 
    y_numeric=False, 
    input_name ="y", 
    estimator=None, 
    to_frame=False,
    allow_nan= False, 
    ):
    """
    Validates the target array `y`, ensuring it is suitable for classification 
    or regression tasks based on its content and the specified strategy.
    
    Parameters 
    -----------
    multi_output : bool, default=False
        Whether to allow 2D y (array or sparse matrix). If false, y will be
        validated as a vector. y cannot have np.nan or np.inf values if
        multi_output=True.
    y_numeric : bool, default=False
        Whether to ensure that y has a numeric type. If dtype of y is object,
        it is converted to float64. Should only be used for regression
        algorithms.
    input_name : str, default="y"
       The data name used to construct the error message. In particular
       if `input_name` is "y".    
    estimator : str or estimator instance, default=None
        If passed, include the name of the estimator in warning messages.
    allow_nan : bool, default=False
       If True, do not throw error when `y` contains NaN.
    to_frame:bool, default=False, 
        reconvert array to its initial type if it is given as pd.Series or
        pd.DataFrame. 
    Returns
    --------
    y: array-like, 
    y_converted : object
        The converted and validated y.
        
    """
    y, column_orig = convert_array_to_pandas(y, input_name= input_name ) 
    if multi_output:
        y = check_array(
            y,
            accept_sparse="csr",
            force_all_finite= True if not allow_nan else "allow-nan",
            ensure_2d=False,
            dtype=None,
            input_name=input_name,
            estimator=estimator,
        )
    else:
        estimator_name = _check_estimator_name(estimator)
        y = _check_y_1d(y, warn=True, input_name=input_name)
        _assert_all_finite(y, input_name=input_name, 
                           estimator_name=estimator_name, 
                           allow_nan=allow_nan , 
                           )
        _ensure_no_complex_data(y)
    if y_numeric and y.dtype.kind == "O":
        y = y.astype(np.float64)
        
    if to_frame: 
        y = array_to_frame (
            y, to_frame =to_frame , 
            columns = column_orig,
            input_name=input_name,
            raise_warning="mute", 
            )
       
    return y

def _object_dtype_isnan(X):
    return X != X

def _assert_all_finite(
    X, allow_nan=False, msg_dtype=None, estimator_name=None, input_name=""
):
    """Like assert_all_finite, but only for ndarray."""

    err_msg=(
        f"{input_name} does not accept missing values encoded as NaN"
        " natively. Alternatively, it is possible to preprocess the data,"
        " for instance by using the imputer transformer like the ufunc"
        " 'soft_imputer' in 'gofast.tools.mlutils.soft_imputer'."
        )
    
    xp, _ = get_namespace(X)

    # if _get_config()["assume_finite"]:
    #     return
    X = xp.asarray(X)

    # for object dtype data, we only check for NaNs (GH-13254)
    if X.dtype == np.dtype("object") and not allow_nan:
        if _object_dtype_isnan(X).any():
            raise ValueError("Input contains NaN. " + err_msg)

    # We need only consider float arrays, hence can early return for all else.
    if X.dtype.kind not in "fc":
        return

    # First try an O(n) time, O(1) space solution for the common case that
    # everything is finite; fall back to O(n) space `np.isinf/isnan` or custom
    # Cython implementation to prevent false positives and provide a detailed
    # error message.
    with np.errstate(over="ignore"):
        first_pass_isfinite = xp.isfinite(xp.sum(X))
    if first_pass_isfinite:
        return
    # Cython implementation doesn't support FP16 or complex numbers
    # use_cython = (
    #     xp is np and X.data.contiguous and X.dtype.type in {np.float32, np.float64}
    # )
    # if use_cython:
    #     out = cy_isfinite(X.reshape(-1), allow_nan=allow_nan)
    #     has_nan_error = False if allow_nan else out == FiniteStatus.has_nan
    #     has_inf = out == FiniteStatus.has_infinite
    # else:
    has_inf = np.isinf(X).any()
    has_nan_error = False if allow_nan else xp.isnan(X).any()
    if has_inf or has_nan_error:
        if has_nan_error:
            type_err = "NaN"
        else:
            msg_dtype = msg_dtype if msg_dtype is not None else X.dtype
            type_err = f"infinity or a value too large for {msg_dtype!r}"
        padded_input_name = input_name + " " if input_name else ""
        msg_err = f"Input {padded_input_name}contains {type_err}."
        if estimator_name and input_name == "X" and has_nan_error:
            # Improve the error message on how to handle missing values in
            # scikit-learn.
            msg_err += (
                f"\n{estimator_name} does not accept missing values"
                " encoded as NaN natively. For supervised learning, you might want"
                " to consider sklearn.ensemble.HistGradientBoostingClassifier and"
                " Regressor which accept missing values encoded as NaNs natively."
                " Alternatively, it is possible to preprocess the data, for"
                " instance by using an imputer transformer in a pipeline or drop"
                " samples with missing values. See"
                " https://scikit-learn.org/stable/modules/impute.html"
                " You can find a list of all estimators that handle NaN values"
                " at the following page:"
                " https://scikit-learn.org/stable/modules/impute.html"
                "#estimators-that-handle-nan-values"
            )
        elif estimator_name is None and has_nan_error: 
            msg_err += f"\n{err_msg}"
            
        raise ValueError(msg_err)
        

def array_to_frame(
    X, 
    *, 
    to_frame=False, 
    columns=None, 
    raise_exception=False, 
    raise_warning=True, 
    input_name='', 
    force=False
):
    """
    Validates and optionally converts an array-like object to a pandas DataFrame,
    applying specified column names if provided or generating them if the `force`
    parameter is set.

    Parameters
    ----------
    X : array-like
        The array to potentially convert to a DataFrame.
    columns : str or list of str, optional
        The names for the resulting DataFrame columns or the Series name.
    to_frame : bool, default=False
        If True, converts `X` to a DataFrame if it isn't already one.
    input_name : str, default=''
        The name of the input variable, used for error and warning messages.
    raise_warning : bool, default=True
        If True and `to_frame` is True but `columns` are not provided,
        a warning is issued unless `force` is True.
    raise_exception : bool, default=False
        If True, raises an exception when `to_frame` is True but columns
        are not provided and `force` is False.
    force : bool, default=False
        Forces the conversion of `X` to a DataFrame by generating column names
        based on `input_name` if `columns` are not provided.

    Returns
    -------
    pd.DataFrame or pd.Series
        The potentially converted DataFrame or Series, or `X` unchanged.

    Examples
    --------
    >>> from hwm.utils.validator import array_to_frame
    >>> from sklearn.datasets import load_iris
    >>> data = load_iris()
    >>> X = data.data
    >>> array_to_frame(X, to_frame=True, columns=['sepal_length', 'sepal_width',
                                                  'petal_length', 'petal_width'])
    """
    # Determine if conversion to frame is needed
    if to_frame and not isinstance(X, (pd.DataFrame, pd.Series)):
        # Handle force conversion without provided column names
        if columns is None and force:
            columns = [f"{input_name}_{i}" for i in range(X.shape[1])]
        elif columns is None:
            msg = (
                f"Array '{input_name}' requires column names for conversion to a DataFrame. "
                 "Provide `columns` or set `force=True` to auto-generate column names."
            )
            if raise_exception:
                raise ValueError(msg)
            if raise_warning and raise_warning not in ("silence", "ignore", "mute"):
                warnings.warn(msg)
            return X  # Early return if no columns and not forcing
        
        # Proceed with conversion using the provided or generated column names
        X,_ = convert_array_to_pandas(X, to_frame=True, columns=columns,
                                      input_name=input_name)
    
    return X
   
def _ensure_no_complex_data(array):
    if (
        hasattr(array, "dtype")
        and array.dtype is not None
        and hasattr(array.dtype, "kind")
        and array.dtype.kind == "c"
    ):
        raise ValueError("Complex data not supported\n{}\n".format(array)) 
     
    
def assert_all_finite(
    X,
    *,
    allow_nan=False,
    estimator_name=None,
    input_name="",
):
    """Throw a ValueError if X contains NaN or infinity.
    Parameters
    ----------
    X : {ndarray, sparse matrix}
        The input data.
    allow_nan : bool, default=False
        If True, do not throw error when `X` contains NaN.
    estimator_name : str, default=None
        The estimator name, used to construct the error message.
    input_name : str, default=""
        The data name used to construct the error message. In particular
        if `input_name` is "X" and the data has NaN values and
        allow_nan is False, the error message will link to the imputer
        documentation.
    """
    _assert_all_finite(
        X.data if sp.issparse(X) else X,
        allow_nan=allow_nan,
        estimator_name=estimator_name,
        input_name=input_name,
    )
    
def check_is_fitted(estimator, attributes=None, *, msg=None, all_or_any=all):
    """Perform is_fitted validation for estimator.

    Checks if the estimator is fitted by verifying the presence of
    fitted attributes (ending with a trailing underscore) and otherwise
    raises a NotFittedError with the given message.

    If an estimator does not set any attributes with a trailing underscore, it
    can define a ``__sklearn_is_fitted__`` or ``__gofast_is_fitted__`` method
    returning a boolean to specify if the estimator is fitted or not.

    Parameters
    ----------
    estimator : estimator instance
        Estimator instance for which the check is performed.

    attributes : str, list or tuple of str, default=None
        Attribute name(s) given as string or a list/tuple of strings
        Eg.: ``["coef_", "estimator_", ...], "coef_"``

        If `None`, `estimator` is considered fitted if there exist an
        attribute that ends with a underscore and does not start with double
        underscore.

    msg : str, default=None
        The default error message is, "This %(name)s instance is not fitted
        yet. Call 'fit' with appropriate arguments before using this
        estimator."

        For custom messages if "%(name)s" is present in the message string,
        it is substituted for the estimator name.

        Eg. : "Estimator, %(name)s, must be fitted before sparsifying".

    all_or_any : callable, {all, any}, default=all
        Specify whether all or any of the given attributes must exist.

    Raises
    ------
    TypeError
        If the estimator is a class or not an estimator instance

    NotFittedError
        If the attributes are not found.
    """
    from ..exceptions import NotFittedError 
    if inspect.isclass(estimator):
        raise TypeError("{} is a class, not an instance.".format(estimator))
    if msg is None:
        msg = (
            "This %(name)s instance is not fitted yet. Call 'fit' with "
            "appropriate arguments before using this estimator."
        )

    if not hasattr(estimator, "fit"):
        raise TypeError("%s is not an estimator instance." % (estimator))

    if attributes is not None:
        if not isinstance(attributes, (list, tuple)):
            attributes = [attributes]
        fitted = all_or_any([hasattr(estimator, attr) for attr in attributes])
    elif hasattr(estimator, "__sklearn_is_fitted__"):
        fitted = estimator.__sklearn_is_fitted__()
    elif hasattr(estimator, "__hwm_is_fitted__"):
        fitted = estimator.__gofast_is_fitted__() 
    else:
        fitted = [
            v for v in vars(estimator) if v.endswith("_") and not v.startswith("__")
        ]

    if not fitted:
        raise NotFittedError(msg % {"name": type(estimator).__name__})
        
def _check_y_1d(y, *, warn=False, input_name ='y'):
    """Ravel column or 1d numpy array, else raises an error.
    
    and Isolated part of check_X_y dedicated to y validation.
    
    Parameters
    ----------
    y : array-like
       Input data.
    warn : bool, default=False
       To control display of warnings.
    Returns
    -------
    y : ndarray
       Output data.
    Raises
    ------
    ValueError
        If `y` is not a 1D array or a 2D array with a single row or column.
    """
    xp, _ = get_namespace(y)
    y = xp.asarray(y)
    shape = y.shape
    if len(shape) == 1:
        return _asarray_with_order(xp.reshape(y, -1), order="C", xp=xp)
    if len(shape) == 2 and shape[1] == 1:
        if warn:
            warnings.warn(
                "A column-vector y was passed when a 1d array was"
                " expected. Please change the shape of y to "
                "(n_samples, ), for example using ravel().",
                DataConversionWarning,
                stacklevel=2,
            )
        return _asarray_with_order(xp.reshape(y, -1), order="C", xp=xp)
    
    raise ValueError(f"{input_name} should be a 1d array, got"
                     f" an array of shape {shape} instead.")

class DataConversionWarning(UserWarning):
    """Warning used to notify implicit data conversions happening in the code.
    This warning occurs when some input data needs to be converted or
    interpreted in a way that may not match the user's expectations.
    For example, this warning may occur when the user
        - passes an integer array to a function which expects float input and
          will convert the input
        - requests a non-copying operation, but a copy is required to meet the
          implementation's data-type expectations;
        - passes an input whose shape can be interpreted ambiguously.
    .. versionchanged:: 0.18
       Moved from sklearn.tools.validation.
    """