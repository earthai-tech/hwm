# -*- coding: utf-8 -*-
#   License: BSD-3-Clause
#   Author: LKouadio <etanoyau@gmail.com>

import re 
import random 
import warnings 
import numpy as np 
import pandas as pd 
from typing import Optional, Tuple, Union 
from typing import Any, Iterable, List

import scipy 
from scipy.special import expit, softmax
from sklearn.model_selection import train_test_split 

from .validator import validate_positive_integer 
from .bunch import Boxspace 

__all__= [
     'activator',
     'add_noises_to',
     'count_functions',
     'ensure_non_empty_batch',
     'gen_X_y_batches',
     'is_in_if',
     'manage_data',
     'safe_slicing',
     'smart_format',
     'str2columns',
     'to_iterable',
     'validate_noise',
     'validate_ratio', 
     'resample_data'
 ]

def activator(z, activation='sigmoid', alpha=1.0, clipping_threshold=250):
    """
    Apply the specified activation function to the input array `z`.

    Parameters
    ----------
    z : array-like
        Input array to which the activation function is applied.
    
    activation : str or callable, default='sigmoid'
        The activation function to apply. Supported activation functions are:
        'sigmoid', 'relu', 'leaky_relu', 'identity', 'elu', 'tanh', 'softmax'.
        If a callable is provided, it should take `z` as input and return the
        transformed output.

    alpha : float, default=1.0
        The alpha value for activation functions that use it (e.g., ELU).

    clipping_threshold : int, default=250
        Threshold value to clip the input `z` to avoid overflow in activation
        functions like 'sigmoid' and 'softmax'.

    Returns
    -------
    activation_output : array-like
        The output array after applying the activation function.

    Notes
    -----
    The available activation functions are defined as follows:

    - Sigmoid: :math:`\sigma(z) = \frac{1}{1 + \exp(-z)}`
    - ReLU: :math:`\text{ReLU}(z) = \max(0, z)`
    - Leaky ReLU: :math:`\text{Leaky ReLU}(z) = \max(0.01z, z)`
    - Identity: :math:`\text{Identity}(z) = z`
    - ELU: :math:`\text{ELU}(z) = \begin{cases}
                  z & \text{if } z > 0 \\
                  \alpha (\exp(z) - 1) & \text{if } z \leq 0
                \end{cases}`
    - Tanh: :math:`\tanh(z) = \frac{\exp(z) - \exp(-z)}{\exp(z) + \exp(-z)}`
    - Softmax: :math:`\text{Softmax}(z)_i = \frac{\exp(z_i)}{\sum_{j} \exp(z_j)}`

    Examples
    --------
    >>> from gofast.estimators.util import activator
    >>> z = np.array([1.0, 2.0, -1.0, -2.0])
    >>> activator(z, activation='relu')
    array([1.0, 2.0, 0.0, 0.0])
    
    >>> activator(z, activation='tanh')
    array([ 0.76159416,  0.96402758, -0.76159416, -0.96402758])
    
    >>> activator(z, activation='softmax')
    array([[0.25949646, 0.70682242, 0.02817125, 0.00550986],
           [0.25949646, 0.70682242, 0.02817125, 0.00550986],
           [0.25949646, 0.70682242, 0.02817125, 0.00550986],
           [0.25949646, 0.70682242, 0.02817125, 0.00550986]])

    See Also
    --------
    GradientDescentBase : Base class for gradient descent-based algorithms.
    
    References
    ----------
    .. [1] Goodfellow, I., Bengio, Y., & Courville, A. (2016). Deep Learning.
           MIT Press. http://www.deeplearningbook.org
    """
    clipping_threshold = validate_positive_integer(
        clipping_threshold, "clipping_threshold"
    )
    if isinstance(activation, str):
        activation = activation.lower()
        if activation == 'sigmoid':
            z = expit(z)
            # z = np.clip(z, -clipping_threshold, clipping_threshold)
            # z= 1 / (1 + np.exp(-z))
            return z
        elif activation == 'relu':
            return np.maximum(0, z)
        elif activation == 'leaky_relu':
            return np.where(z > 0, z, z * 0.01)
        elif activation == 'identity':
            return z
        elif activation == 'elu':
            return np.where(z > 0, z, alpha * (np.exp(z) - 1))
        elif activation == 'tanh':
            return np.tanh(z)
        elif activation == 'softmax':
            exp_z= softmax(z)
            # z = np.clip(z, -clipping_threshold, clipping_threshold)
            # exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
            return exp_z # exp_z / np.sum(exp_z, axis=1, keepdims=True)
        else:
            raise ValueError(f"Unsupported activation function: {activation}")
    elif callable(activation):
        return activation(z)
    else:
        raise ValueError("Activation must be a string or a callable function")
        
def resample_data(
    *data: Any,
    samples: Union[int, float, str] = 1,
    replace: bool = False,
    random_state: int = None,
    shuffle: bool = True
) -> List[Any]:
    """
    Resample multiple data structures (arrays, sparse matrices, Series, 
    DataFrames) based on specified sample size or ratio.

    Parameters
    ----------
    *data : Any
        Variable number of array-like, sparse matrix, pandas Series, or 
        DataFrame objects to be resampled.
        
    samples : Union[int, float, str], optional
        Specifies the number of items to sample from each data structure.
        
        - If an integer greater than 1, it is treated as the exact number 
          of items to sample.
        - If a float between 0 and 1, it is treated as a ratio of the 
          total number of rows to sample.
        - If a string containing a percentage (e.g., "50%"), it calculates 
          the sample size as a percentage of the total data length.
        
        The default is 1, meaning no resampling is performed unless a 
        different value is specified.

    replace : bool, default=False
        Determines if sampling with replacement is allowed, enabling the 
        same row to be sampled multiple times.

    random_state : int, optional
        Sets the seed for the random number generator to ensure 
        reproducibility. If specified, repeated calls with the same 
        parameters will yield identical results.

    shuffle : bool, default=True
        If True, shuffles the data before sampling. Otherwise, rows are 
        selected sequentially without shuffling.

    Returns
    -------
    List[Any]
        A list of resampled data structures, each in the original format 
        (e.g., numpy array, sparse matrix, pandas DataFrame) and with the 
        specified sample size.

    Methods
    -------
    - `_determine_sample_size`: Calculates the sample size based on the 
      `samples` parameter.
    - `_perform_sampling`: Conducts the sampling process based on the 
      calculated sample size, `replace`, and `shuffle` parameters.

    Notes
    -----
    - If `samples` is given as a percentage string (e.g., "25%"), the 
      actual number of rows to sample, :math:`n`, is calculated as:
      
      .. math::
          n = \left(\frac{\text{percentage}}{100}\right) \times N

      where :math:`N` is the total number of rows in the data structure.

    - Resampling supports both dense and sparse matrices. If the input 
      contains sparse matrices stored within numpy objects, the function 
      extracts and samples them directly.

    Examples
    --------
    >>> from hwm.utils._core import resample_data
    >>> import numpy as np
    >>> data = np.arange(100).reshape(20, 5)

    # Resample 10 items from each data structure with replacement
    >>> resampled_data = resample_data(data, samples=10, replace=True)
    >>> print(resampled_data[0].shape)
    (10, 5)
    
    # Resample 50% of the rows from each data structure
    >>> resampled_data = resample_data(data, samples=0.5, random_state=42)
    >>> print(resampled_data[0].shape)
    (10, 5)

    # Resample data with a percentage-based sample size
    >>> resampled_data = resample_data(data, samples="25%", random_state=42)
    >>> print(resampled_data[0].shape)
    (5, 5)

    References
    ----------
    .. [1] Fisher, R.A., "The Use of Multiple Measurements in Taxonomic 
           Problems", Annals of Eugenics, 1936.

    See Also
    --------
    np.random.choice : Selects random samples from an array.
    pandas.DataFrame.sample : Randomly samples rows from a DataFrame.
    """

    resampled_structures = []

    for d in data:
        # Handle sparse matrices encapsulated in numpy objects
        if isinstance(d, np.ndarray) and d.dtype == object and scipy.sparse.issparse(d.item()):
            d = d.item()  # Extract the sparse matrix from the numpy object

        # Determine sample size based on `samples` parameter
        n_samples = _determine_sample_size(d, samples, is_percent="%" in str(samples))
        
        # Sample the data structure based on the computed sample size
        sampled_d = _perform_sampling(d, n_samples, replace, random_state, shuffle)
        resampled_structures.append(sampled_d)
 
    return resampled_structures[0] if len(
        resampled_structures)==1 else resampled_structures

def _determine_sample_size(d: Any, samples: Union[int, float, str], 
                           is_percent: bool) -> int:
    """
    Determine the number of samples to draw based on the input size or ratio.
    """
    if isinstance(samples, str) and is_percent:
        samples = samples.replace("%", "")
    try:
        samples = float(samples)
    except ValueError:
        raise TypeError(f"Invalid type for 'samples': {type(samples).__name__}."
                        " Expected int, float, or percentage string.")
   
    d_length = d.shape[0] if hasattr(d, 'shape') else len(d)
    if samples < 1 or is_percent:
        return max(1, int(samples * d_length))
    return int(samples)

def _perform_sampling(d: Any, n_samples: int, replace: bool, 
                      random_state: int, shuffle: bool) -> Any:
    """
    Perform the actual sampling operation on the data structure.
    """
    if isinstance(d, pd.DataFrame) or isinstance(d, pd.Series):
        return d.sample(n=n_samples, replace=replace, random_state=random_state
                        ) if shuffle else d.iloc[:n_samples]
    elif scipy.sparse.issparse(d):
        if scipy.sparse.isspmatrix_coo(d):
            warnings.warn("coo_matrix does not support indexing. Conversion"
                          " to CSR matrix is recommended.")
            d = d.tocsr()
        indices = np.random.choice(d.shape[0], n_samples, replace=replace
                                   ) if shuffle else np.arange(n_samples)
        return d[indices]
    else:
        d_array = np.array(d) if not hasattr(d, '__array__') else d
        indices = np.random.choice(len(d_array), n_samples, replace=replace
                                   ) if shuffle else np.arange(n_samples)
        return d_array[indices] if d_array.ndim == 1 else d_array[indices, :]
    

def manage_data(
    data, 
    as_frame= False, 
    return_X_y= False, 
    split_X_y= False, 
    target_names= None, 
    test_size= 0.3, 
    noise= None, 
    seed= None, 
    **kwargs
):

    """ Manage the data and setup into an Object 
    
    Parameters
    -----------
    data: Pd.DataFrame 
        The dataset to manage 

    as_frame : bool, default=False
        If True, the data is a pandas DataFrame including columns with
        appropriate dtypes (numeric). The target is
        a pandas DataFrame or Series depending on the number of target columns.
        If `return_X_y` is True, then (`data`, `target`) will be pandas
        DataFrames or Series as described below.

    return_X_y : bool, default=False
        If True, returns ``(data, target)`` instead of a Bowlspace object. See
        below for more information about the `data` and `target` object.
        
    split_X_y: bool, default=False,
        If True, the data is splitted to hold the training set (X, y)  and the 
        testing set (Xt, yt) with the according to the test size ratio. 
        
    target_names: str, optional 
        the name of the target to retreive. If ``None`` the default target columns 
        are collected and may be a multioutput `y`. For a singular classification 
        or regression problem, it is recommended to indicate the name of the target 
        that is needed for the learning task. 
        
    test_size: float, default is {{.3}} i.e. 30% (X, y)
        The ratio to split the data into training (X, y)  and testing (Xt, yt) set 
        respectively
        . 
    noise : float, Optional
        The percentage of values to be replaced with NaN in each column. 
        This must be a number between 0 and 1. Default is None.
        
    seed: int, array-like, BitGenerator, np.random.RandomState, \
        np.random.Generator, optional
       If int, array-like, or BitGenerator, seed for random number generator. 
       If np.random.RandomState or np.random.Generator, use as given.
       
    Returns 
    -------
    data : :class:`~gofast.tools.box.Boxspace` object
        Dictionary-like object, with the following attributes.
        data : {ndarray, dataframe} 
            The data matrix. If ``as_frame=True``, `data` will be a pandas DataFrame.
        target: {ndarray, Series} 
            The classification target. If `as_frame=True`, `target` will be
            a pandas Series.
        feature_names: list
            The names of the dataset columns.
        target_names: list
            The names of target classes.
        frame: DataFrame 
            Only present when `as_frame=True`. DataFrame with `data` and
            `target`.

    data, target: tuple if `return_X_y` is ``True``
        A tuple of two ndarray. The first containing a 2D array of shape
        (n_samples, n_features) with each row representing one sample and
        each column representing the features. The second ndarray of shape
        (n_samples,) containing the target samples.

    X, Xt, y, yt: Tuple if `split_X_y` is ``True`` 
        A tuple of two ndarray (X, Xt). The first containing a 2D array of:
            
        .. math:: 
            
            \\text{shape}(X, y) =  1-  \\text{test_ratio} *\
                (n_{samples}, n_{features}) *100
            
            \\text{shape}(Xt, yt)= \\text{test_ratio} * \
                (n_{samples}, n_{features}) *100
        
        where each row representing one sample and each column representing the 
        features. The second ndarray of shape(n_samples,) containing the target 
        samples.
    
    """
    # Ensure the correct data types for the parameters
    as_frame, return_X_y, split_X_y = map(
        lambda x: bool(x), [as_frame, return_X_y, split_X_y]
    )
    test_size = float(test_size)
   
    if seed is not None:
        seed = int(seed)
    
    if target_names: 
        target_names = to_iterable (
            target_names, exclude_string=True,transform=True )
    frame = data.copy()

    feature_names = (
        is_in_if(list( frame.columns), target_names, return_diff =True )
        if target_names else list(frame.columns )
    )
    y = None
    
    if split_X_y:
        # set to True to get y 
        return_X_y =True 
 
    if return_X_y:
        y = data [target_names].squeeze ()  
        data.drop( columns = target_names, inplace =True )
        
    # # Apply noises: Noises only in the data not in target
    # add_gaussian_noise=False 
    # noise = validate_noise(noise )
    
    # if noise=='gaussian': 
    #     add_gaussian_noise=True 
    #     #Small value of noise. Do nothing when gaussian noises 
    #     # is applied, just to skip value error. 
    #     noise =.1 
        
    data = add_noises_to(
        data, noise=noise, seed=seed)

    if not as_frame:
        data = np.asarray(data)
        y = np.squeeze(np.asarray(y))
    
    if split_X_y:
        return train_test_split(data, y, test_size=test_size, random_state=seed)
    
    if return_X_y:
        return data, y

    frame[feature_names] = add_noises_to(
        frame[feature_names], 
        noise=noise,
        seed=seed, 
        )

    if as_frame:
        return frame
    
    return Boxspace(
        data=data,
        target=frame[target_names].values if target_names else None,
        frame=frame,
        target_names=[target_names] if target_names else [],
        feature_names=feature_names,
        **kwargs
    )

def to_iterable (
        y, exclude_string= False, transform = False , parse_string =False, 
)->Union [bool , list]: 
    """ Asserts iterable object and returns boolean or transform object into
     an iterable.
    
    Function can also transform a non-iterable object to an iterable if 
    `transform` is set to ``True``.
    
    :param y: any, object to be asserted 
    :param exclude_string: bool, does not consider string as an iterable 
        object if `y` is passed as a string object. 
    :param transform: bool, transform  `y` to an iterable objects. But default 
        puts `y` in a list object. 
    :param parse_string: bool, parse string and convert the list of string 
        into iterable object is the `y` is a string object and containg the 
        word separator character '[#&.*@!_,;\s-]'. Refer to the function 
        :func:`~gofast.tools.coreutils.str2columns` documentation.
        
    :returns: 
        - bool, or iterable object if `transform` is set to ``True``. 
        
    .. note:: 
        Parameter `parse_string` expects `transform` to be ``True``, otherwise 
        a ValueError will raise. Note :func:`.is_iterable` is not dedicated 
        for string parsing. It parses string using the default behaviour of 
        :func:`.str2columns`. Use the latter for string parsing instead. 
        
    :Examples: 
    >>> from gofast.coreutils.is_iterable 
    >>> is_iterable ('iterable', exclude_string= True ) 
    Out[28]: False
    >>> is_iterable ('iterable', exclude_string= True , transform =True)
    Out[29]: ['iterable']
    >>> is_iterable ('iterable', transform =True)
    Out[30]: 'iterable'
    >>> is_iterable ('iterable', transform =True, parse_string=True)
    Out[31]: ['iterable']
    >>> is_iterable ('iterable', transform =True, exclude_string =True, 
                     parse_string=True)
    Out[32]: ['iterable']
    >>> is_iterable ('parse iterable object', parse_string=True, 
                     transform =True)
    Out[40]: ['parse', 'iterable', 'object']
    """
    if (parse_string and not transform) and isinstance (y, str): 
        raise ValueError ("Cannot parse the given string. Set 'transform' to"
                          " ``True`` otherwise use the 'str2columns' utils"
                          " from 'gofast.tools.coreutils' instead.")
    y = str2columns(y) if isinstance(y, str) and parse_string else y 
    
    isiter = False  if exclude_string and isinstance (
        y, str) else hasattr (y, '__iter__')
    
    return ( y if isiter else [ y ] )  if transform else isiter 

def is_in_if (o: iter,  items: Union [str , iter], error = 'raise', 
               return_diff =False, return_intersect = False): 
    """ Raise error if item is not  found in the iterable object 'o' 
    
    :param o: unhashable type, iterable object,  
        object for checkin. It assumes to be an iterable from which 'items' 
        is premused to be in. 
    :param items: str or list, 
        Items to assert whether it is in `o` or not. 
    :param error: str, default='raise'
        raise or ignore error when none item is found in `o`. 
    :param return_diff: bool, 
        returns the difference items which is/are not included in 'items' 
        if `return_diff` is ``True``, will put error to ``ignore`` 
        systematically.
    :param return_intersect:bool,default=False
        returns items as the intersection between `o` and `items`.
    :raise: ValueError 
        raise ValueError if `items` not in `o`. 
    :return: list,  
        `s` : object found in ``o` or the difference object i.e the object 
        that is not in `items` provided that `error` is set to ``ignore``.
        Note that if None object is found  and `error` is ``ignore`` , it 
        will return ``None``, otherwise, a `ValueError` raises. 
        
    :example: 
        >>> from gofast.datasets import load_hlogs 
        >>> from gofast.tools.coreutils import is_in_if 
        >>> X0, _= load_hlogs (as_frame =True )
        >>> is_in_if  (X0 , items= ['depth_top', 'top']) 
        ... ValueError: Item 'top' is missing in the object 
        >>> is_in_if (X0, ['depth_top', 'top'] , error ='ignore') 
        ... ['depth_top']
        >>> is_in_if (X0, ['depth_top', 'top'] , error ='ignore',
                       return_diff= True) 
        ... ['sp',
         'well_diameter',
         'layer_thickness',
         'natural_gamma',
         'short_distance_gamma',
         'strata_name',
         'gamma_gamma',
         'depth_bottom',
         'rock_name',
         'resistivity',
         'hole_id']
    """
    
    if isinstance (items, str): 
        items =[items]
    elif not to_iterable(o): 
        raise TypeError (f"Expect an iterable object, not {type(o).__name__!r}")
    # find intersect object 
    s= set (o).intersection (items) 
    
    miss_items = list(s.difference (o)) if len(s) > len(
        items) else list(set(items).difference (s)) 

    if return_diff or return_intersect: 
        error ='ignore'
    
    if len(miss_items)!=0 :
        if error =='raise': 
            v= smart_format(miss_items)
            verb = f"{ ' '+ v +' is' if len(miss_items)<2 else  's '+ v + 'are'}"
            raise ValueError (
                f"Item{verb} missing in the {type(o).__name__.lower()} {o}.")
            
       
    if return_diff : 
        # get difference 
        s = list(set(o).difference (s))  if len(o) > len( 
            s) else list(set(items).difference (s)) 
        # s = set(o).difference (s)  
    elif return_intersect: 
        s = list(set(o).intersection(s))  if len(o) > len( 
            items) else list(set(items).intersection (s))     
    
    s = None if len(s)==0 else list (s) 
    
    return s 

def add_noises_to(
    data,  
    noise=0.1, 
    seed=None, 
    gaussian_noise=False,
    cat_missing_value=pd.NA
    ):
    """
    Adds NaN or specified missing values to a pandas DataFrame.

    Parameters
    ----------
    data : pandas.DataFrame
        The DataFrame to which NaN values or specified missing 
        values will be added.

    noise : float, default=0.1
        The percentage of values to be replaced with NaN or the 
        specified missing value in each column. This must be a 
        number between 0 and 1. Default is 0.1 (10%).

        .. math:: \text{noise} = \frac{\text{number of replaced values}}{\text{total values in column}}

    seed : int, array-like, BitGenerator, np.random.RandomState, np.random.Generator, optional
        Seed for random number generator to ensure reproducibility. 
        If `seed` is an int, array-like, or BitGenerator, it will be 
        used to seed the random number generator. If `seed` is a 
        np.random.RandomState or np.random.Generator, it will be used 
        as given.

    gaussian_noise : bool, default=False
        If `True`, adds Gaussian noise to the data. Otherwise, replaces 
        values with NaN or the specified missing value.

    cat_missing_value : scalar, default=pd.NA
        The value to use for missing data in categorical columns. By 
        default, `pd.NA` is used.

    Returns
    -------
    pandas.DataFrame
        A DataFrame with NaN or specified missing values added.

    Notes
    -----
    The function modifies the DataFrame by either adding Gaussian noise 
    to numerical columns or replacing a percentage of values in each 
    column with NaN or a specified missing value.

    The Gaussian noise is added according to the formula:

    .. math:: \text{new_value} = \text{original_value} + \mathcal{N}(0, \text{noise})

    where :math:`\mathcal{N}(0, \text{noise})` represents a normal 
    distribution with mean 0 and standard deviation equal to `noise`.

    Examples
    --------
    >>> from gofast.tools.coreutils import add_noises_to
    >>> import pandas as pd
    >>> df = pd.DataFrame({'A': [1, 2, 3], 'B': ['x', 'y', 'z']})
    >>> new_df = add_noises_to(df, noise=0.2)
    >>> new_df
         A     B
    0  1.0  <NA>
    1  NaN     y
    2  3.0  <NA>

    >>> df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
    >>> new_df = add_noises_to(df, noise=0.1, gaussian_noise=True)
    >>> new_df
              A         B
    0  1.063292  3.986400
    1  2.103962  4.984292
    2  2.856601  6.017380

    See Also
    --------
    pandas.DataFrame : Two-dimensional, size-mutable, potentially 
        heterogeneous tabular data.
    numpy.random.normal : Draw random samples from a normal 
        (Gaussian) distribution.

    References
    ----------
    .. [1] Harris, C. R., Millman, K. J., van der Walt, S. J., et al. 
           (2020). Array programming with NumPy. Nature, 585(7825), 
           357-362.
    """
    
    is_frame = isinstance (data, pd.DataFrame ) 
    if not is_frame: 
        data = pd.DataFrame(data ) 
        
    np.random.seed(seed)
    if noise is None: 
        return data 
    noise, gaussian_noise  = _parse_gaussian_noise (noise )

    if gaussian_noise:
        # Add Gaussian noise to numerical columns only
        def add_gaussian_noise(column):
            if pd.api.types.is_numeric_dtype(column):
                return column + np.random.normal(0, noise, size=column.shape)
            return column
        
        noise_data = data.apply(add_gaussian_noise)
        
        if not is_frame: 
            noise_data = np.asarray(noise_data)
        return noise_data
    else:
        # Replace values with NaN or specified missing value
        df_with_nan = data.copy()
        nan_count_per_column = int(noise * len(df_with_nan))

        for column in df_with_nan.columns:
            nan_indices = random.sample(range(len(df_with_nan)), nan_count_per_column)
            if pd.api.types.is_numeric_dtype(df_with_nan[column]):
                df_with_nan.loc[nan_indices, column] = np.nan
            else:
                df_with_nan.loc[nan_indices, column] = cat_missing_value
                
        if not is_frame: 
            df_with_nan = df_with_nan.values 
            
        return df_with_nan
    
def _parse_gaussian_noise(noise):
    """
    Parses the noise parameter to determine if Gaussian noise should be used
    and extracts the noise level if specified.

    Parameters
    ----------
    noise : str, float, or None
        The noise parameter to be parsed. Can be a string specifying Gaussian
        noise with an optional noise level, a float, or None.

    Returns
    -------
    tuple
        A tuple containing:
        - float: The noise level.
        - bool: Whether Gaussian noise should be used.

    Examples
    --------
    >>> from gofast.tools.coreutils import _parse_gaussian_noise
    >>> _parse_gaussian_noise('0.1gaussian')
    (0.1, True)
    >>> _parse_gaussian_noise('gaussian0.1')
    (0.1, True)
    >>> _parse_gaussian_noise('gaussian_0.1')
    (0.1, True)
    >>> _parse_gaussian_noise('gaussian10%')
    (0.1, True)
    >>> _parse_gaussian_noise('gaussian 10 %')
    (0.1, True)
    >>> _parse_gaussian_noise(0.05)
    (0.05, False)
    >>> _parse_gaussian_noise(None)
    (0.1, False)
    >>> _parse_gaussian_noise('invalid')
    Traceback (most recent call last):
        ...
    ValueError: Invalid noise value: invalid
    """
    gaussian_noise = False
    default_noise = 0.1

    if isinstance(noise, str):
        orig_noise = noise 
        noise = noise.lower()
        gaussian_keywords = ["gaussian", "gauss"]

        if any(keyword in noise for keyword in gaussian_keywords):
            gaussian_noise = True
            noise = re.sub(r'[^\d.%]', '', noise)  # Remove non-numeric and non-'%' characters
            noise = re.sub(r'%', '', noise)  # Remove '%' if present

            try:
                noise_level = float(noise) / 100 if '%' in orig_noise else float(noise)
                noise = noise_level if noise_level else default_noise
            except ValueError:
                noise = default_noise

        else:
            try:
                noise = float(noise)
            except ValueError:
                raise ValueError(f"Invalid noise value: {noise}")
    elif noise is None:
        noise = default_noise
    
    noise = validate_noise (noise ) 
    
    return noise, gaussian_noise

def validate_noise(noise):
    """
    Validates the `noise` parameter and returns either the noise value
    as a float or the string 'gaussian'.

    Parameters
    ----------
    noise : str or float or None
        The noise parameter to be validated. It can be the string
        'gaussian', a float value, or None.

    Returns
    -------
    float or str
        The validated noise value as a float or the string 'gaussian'.

    Raises
    ------
    ValueError
        If the `noise` parameter is a string other than 'gaussian' or
        cannot be converted to a float.

    Examples
    --------
    >>> validate_noise('gaussian')
    'gaussian'
    >>> validate_noise(0.1)
    0.1
    >>> validate_noise(None)
    None
    >>> validate_noise('0.2')
    0.2

    """
    if isinstance(noise, str):
        if noise.lower() == 'gaussian':
            return 'gaussian'
        else:
            try:
                noise = float(noise)
            except ValueError:
                raise ValueError("The `noise` parameter accepts the string"
                                 " 'gaussian' or a float value.")
    elif noise is not None:
        noise = validate_ratio(noise, bounds=(0, 1), param_name='noise' )
        # try:
        # except ValueError:
        #     raise ValueError("The `noise` parameter must be convertible to a float.")
    return noise


def gen_X_y_batches(
    X, y, *,
    batch_size="auto",
    n_samples=None,
    min_batch_size=0,
    shuffle=True,
    random_state=None,
    return_batches=False,
    default_size=200,
):
    """
    Generate batches of data (`X`, `y`) for machine learning tasks such as 
    training or evaluation. This function slices the dataset into smaller 
    batches, optionally shuffles the data, and returns them as a list of 
    tuples or just the data batches.

    Parameters
    ----------
    X : ndarray of shape (n_samples, n_features)
        The input data matrix, where each row is a sample and each column 
        represents a feature.

    y : ndarray of shape (n_samples,) or (n_samples, n_targets)
        The target variable(s) corresponding to `X`. Can be a vector or 
        matrix depending on the problem (single or multi-output).

    batch_size : int, "auto", default="auto"
        The number of samples per batch. If set to `"auto"`, it uses the 
        minimum between `default_size` and the number of samples, `n_samples`.

    n_samples : int, optional, default=None
        The total number of samples to consider. If `None`, the function 
        defaults to using the number of samples in `X`.

    min_batch_size : int, default=0
        The minimum size for each batch. This parameter ensures that the 
        final batch contains at least `min_batch_size` samples. If the 
        last batch is smaller than `min_batch_size`, it will be excluded 
        from the result.

    shuffle : bool, default=True
        If `True`, the data is shuffled before batching. This helps avoid 
        bias when splitting data for training and validation.

    random_state : int, RandomState instance, or None, default=None
        The seed used by the random number generator for reproducibility. 
        If `None`, the random number generator uses the system time or 
        entropy source.

    return_batches : bool, default=False
        If `True`, the function returns both the data batches and the slice 
        objects used to index into `X` and `y`. If `False`, only the 
        data batches are returned.

    default_size : int, default=200
        The default batch size used when `batch_size="auto"` is selected.

    Returns
    -------
    Xy_batches : list of tuples
        A list of tuples where each tuple contains a batch of `X` and its 
        corresponding batch of `y`.

    batch_slices : list of slice objects, optional
        If `return_batches=True`, this list of `slice` objects is returned, 
        each representing the slice of `X` and `y` used for a specific batch.

    Notes
    -----
    - This function ensures that no empty batches are returned. If a batch 
      contains zero samples (either from improper slicing or due to 
      `min_batch_size`), it will be excluded.
    - The function performs shuffling using scikit-learn's `shuffle` function, 
      which is more stable and reduces memory usage by shuffling indices 
      rather than the whole dataset.
    - The function utilizes the `gen_batches` utility to divide the data into 
      batches.

    Examples
    --------
    >>> from gofast.tools.coreutils import gen_X_y_batches
    >>> X = np.random.rand(2000, 5)
    >>> y = np.random.randint(0, 2, size=(2000,))
    >>> batches = gen_X_y_batches(X, y, batch_size=500, shuffle=True)
    >>> len(batches)
    4

    >>> X = np.random.rand(2000, 5)
    >>> y = np.random.randint(0, 2, size=(2000,))
    >>> batches, slices = gen_X_y_batches(
    >>>     X, y, batch_size=500, shuffle=True, return_batches=True
    >>> )
    >>> len(batches)
    4
    >>> len(slices)
    4

    Notes
    ------
    Given a dataset of size `n_samples` and target `y`, we want to partition 
    the dataset into batches. The `batch_size` parameter defines the maximum 
    number of samples in each batch, and `min_batch_size` ensures that 
    the last batch has a minimum size if possible.

    For each batch, we perform the following steps:
    
    1. **Determine the batch size**:
       - If `batch_size` is "auto", we set:
       
       .. math::
           \text{batch\_size} = \min(\text{default\_size}, n_{\text{samples}})
       
    2. **Validate batch size**:
       - Ensure the batch size does not exceed the total number of samples. 
       If it does, we clip it:
       
       .. math::
           \text{batch\_size} = \min(\text{batch\_size}, n_{\text{samples}})
    
    3. **Generate batches**:
       - Use the `gen_batches` utility to create slice indices that partition 
       the dataset into batches:
       
       .. math::
           \text{batch\_slices} = \text{gen\_batches}(n_{\text{samples}}, 
           \text{batch\_size})
       
    4. **Shuffling** (if enabled):
       - If `shuffle=True`, shuffle the dataset's indices before splitting:
       
       .. math::
           \text{indices} = \text{shuffle}(0, 1, \dots, n_{\text{samples}} - 1)
    
    5. **Return Batches**:
       - After creating the batches, return them as tuples of `(X_batch, y_batch)`.

    See Also
    --------
    gen_batches : A utility function that generates slices of data.
    shuffle : A utility to shuffle data while keeping the data and labels in sync.

    References
    ----------
    [1] Scikit-learn. "sklearn.utils.shuffle". Available at 
    https://scikit-learn.org/stable/modules/generated/sklearn.utils.shuffle.html
    """
    from sklearn.utils import shuffle as sk_shuffle, _safe_indexing
    from sklearn.utils import gen_batches
    from .validator import check_X_y, validate_batch_size 
    
    X, y = check_X_y(X, y)

    # List to store the resulting batches
    Xy_batches = []
    batch_slices = []

    # Default to the number of samples in X if not provided
    if n_samples is None:
        n_samples = X.shape[0]

    # Determine and validate batch size
    if batch_size == "auto":
        batch_size = min(default_size, n_samples)
    else:
        
        if batch_size > n_samples:
            warnings.warn(
                "Got `batch_size` less than 1 or larger than "
                "sample size. It is going to be clipped."
            )
            batch_size = np.clip(batch_size, 1, n_samples)
    # Validate batch size
    batch_size = validate_batch_size( 
        batch_size, n_samples, min_batch_size=min_batch_size
    )
    
    # Generate batch slices
    batches = list(
        gen_batches(n_samples, batch_size, min_batch_size=min_batch_size)
    )

    # Generate an array of indices for shuffling
    indices = np.arange(X.shape[0])

    if shuffle:
        # Shuffle indices for stable randomization
        sample_idx = sk_shuffle(indices, random_state=random_state)

    for batch_idx, batch_slice in enumerate(batches):
        # Slice the training data to obtain the current batch
        if shuffle:
            X_batch = _safe_indexing(X, sample_idx[batch_slice])
            y_batch = y[sample_idx[batch_slice]]
        else:
            X_batch = X[batch_slice]
            y_batch = y[batch_slice]

        if y_batch.size == 0 or X_batch.size == 0:
            if shuffle: 
                X_batch, y_batch = ensure_non_empty_batch(
                    X, y, batch_slice, 
                    random_state=random_state, 
                    error = "warn", 
                ) 
            else:
                continue

        # Append valid batches to the results
        Xy_batches.append((X_batch, y_batch))
        batch_slices.append(batch_slice)

    if len(Xy_batches)==0: 
        # No batch found 
        Xy_batches.append ((X, y)) 
        
    return (Xy_batches, batch_slices) if return_batches else Xy_batches


def ensure_non_empty_batch(
    X, y, *, batch_slice, max_attempts=10, random_state=None,
    error ="raise", 
):
    """
    Shuffle the dataset (`X`, `y`) until the specified `batch_slice` yields 
    a non-empty batch. This function ensures that the batch extracted using 
    `batch_slice` contains at least one sample by repeatedly shuffling the 
    data and reapplying the slice.

    Parameters
    ----------
    X : ndarray of shape (n_samples, n_features)
        The input data matrix, where each row corresponds to a sample and 
        each column corresponds to a feature.

    y : ndarray of shape (n_samples,) or (n_samples, n_targets)
        The target variable(s) corresponding to `X`. It can be a one-dimensional 
        array for single-output tasks or a two-dimensional array for multi-output 
        tasks.

    batch_slice : slice
        A slice object representing the indices for the batch. For example, 
        `slice(0, 512)` would extract the first 512 samples from `X` and `y`.

    max_attempts : int, optional, default=10
        The maximum number of attempts to shuffle the data to obtain a non-empty 
        batch. If the batch remains empty after the specified number of attempts, 
        a `ValueError` is raised.

    random_state : int, RandomState instance, or None, default=None
        Controls the randomness of the shuffling. Pass an integer for reproducible 
        results across multiple function calls. If `None`, the random number 
        generator is the RandomState instance used by `np.random`.

    error: str, default ='raise' 
        Handle error status when empty batch is still present after 
        `max_attempts`. Expect ``{"raise", "warn" "ignore"} , if ``warn``, 
        error is converted in warning message. Any other value ignore the 
        error message. 
        
    Returns
    -------
    X_batch : ndarray of shape (batch_size, n_features)
        The batch of input data extracted using `batch_slice`. Ensures that 
        `X_batch` is not empty.

    y_batch : ndarray of shape (batch_size,) or (batch_size, n_targets)
        The batch of target data corresponding to `X_batch`, extracted using 
        `batch_slice`. Ensures that `y_batch` is not empty.

    Raises
    ------
    ValueError
        If a non-empty batch cannot be obtained after `max_attempts` shuffles.

    Examples
    --------
    >>> from gofast.tools.coreutils import ensure_non_empty_batch
    >>> import numpy as np
    >>> X = np.random.rand(2000, 5)
    >>> y = np.random.randint(0, 2, size=(2000,))
    >>> batch_slice = slice(0, 512)
    >>> X_batch, y_batch = ensure_non_empty_batch(X, y, batch_slice=batch_slice)
    >>> X_batch.shape
    (512, 5)
    >>> y_batch.shape
    (512,)

    >>> # Example where the batch might initially be empty
    >>> X_empty = np.empty((0, 5))
    >>> y_empty = np.empty((0,))
    >>> try:
    ...     ensure_non_empty_batch(X_empty, y_empty, batch_slice=slice(0, 512))
    ... except ValueError as e:
    ...     print(e)
    ...
    Unable to obtain a non-empty batch after 10 attempts.

    Notes
    -----
    Given a dataset with `n_samples` samples, the goal is to find a subset of 
    samples defined by the `batch_slice` such that:

    .. math::
        \text{batch\_size} = \text{len}(X[\text{batch\_slice}])

    The function ensures that:

    .. math::
        \text{batch\_size} > 0

    This is achieved by iteratively shuffling the dataset and reapplying the 
    `batch_slice` until the condition is satisfied or the maximum number of 
    attempts is reached.

    See Also
    --------
    gen_batches : Generate slice objects to divide data into batches.
    shuffle : Shuffle arrays or sparse matrices in a consistent way.

    References
    ----------
    .. [1] Pedregosa, F., Varoquaux, G., Gramfort, A., Michel, V., Thirion, 
       B., Grisel, O., ... & Duchesnay, E. (2011). Scikit-learn: Machine 
       learning in Python. *Journal of Machine Learning Research*, 12, 
       2825-2830.
    .. [2] NumPy Developers. (2023). NumPy Documentation. 
       https://numpy.org/doc/
    """
    from sklearn.utils import shuffle as sk_shuffle 
    
    attempts = 0

    while attempts < max_attempts:
        # Extract the batch using the provided slice
        X_batch = X[batch_slice]
        y_batch = y[batch_slice]

        # Check if both X_batch and y_batch are non-empty
        if X_batch.size > 0 and y_batch.size > 0:
            return X_batch, y_batch

        # Shuffle the dataset
        X, y = sk_shuffle(
            X, y, random_state=random_state
        )

        attempts += 1

    msg =  f"Unable to obtain a non-empty batch after {max_attempts} attempts."
    if error=="raise": 
        # If a non-empty batch is not found after max_attempts, raise an error
        raise ValueError(msg)
    elif error =='warn':
        warnings.warn( msg ) 
        
    return X, y 
    
  
def safe_slicing(slice_indexes, X):
    """
    Removes slices from the list `slice_indexes` that result in zero samples 
    when applied to the data `X`. The function checks each slice to ensure 
    it selects at least one sample, and discards any slices with no samples 
    selected.

    Parameters
    ----------
    slice_indexes : list of slice objects
        A list of slice objects, each representing a range of indices 
        that can be used to index into a dataset, typically for batch 
        processing.

    X : ndarray of shape (n_samples, n_features)
        The data array (or any other array-like structure) that the slices 
        will be applied to. The function assumes that each slice in 
        `slice_indexes` corresponds to a subset of rows (samples) in `X`.

    Returns
    -------
    valid_slices : list of slice objects
        A list of slice objects that correspond to valid (non-empty) 
        subsets of `X`. Slices with zero elements (e.g., when the 
        start index is equal to or greater than the end index) are removed.

    Examples
    --------
    # Example 1: Basic use case where the last slice is valid
    >>> X = np.random.rand(2000, 5)  # 2000 samples, 5 features
    >>> slice_indexes = [slice(0, 512), slice(512, 1024), slice(1024, 1536), 
                         slice(1536, 2000)]
    >>> safe_slicing(slice_indexes, X)
    [slice(0, 512, None), slice(512, 1024, None), slice(1024, 1536, None),
     slice(1536, 2000, None)]

    # Example 2: Case where the last slice has zero elements and is removed
    >>> slice_indexes = [slice(0, 512), slice(512, 1024), slice(1024, 1536),
                         slice(1536, 1500)]
    >>> safe_slicing(slice_indexes, X)
    [slice(0, 512, None), slice(512, 1024, None), slice(1024, 1536, None)]

    # Example 3: Empty slice case where all slices are removed
    >>> slice_indexes = [slice(0, 0), slice(1, 0)]
    >>> safe_slicing(slice_indexes, X)
    []

    Notes
    -----
    - This function is useful when handling slices generated for batch 
      processing in machine learning workflows, ensuring that only valid 
      batches are processed.
    - The function checks the start and stop indices of each slice and 
      ensures that `end > start` before including the slice in the 
      returned list.
    """
    
    valid_slices = []
    for slice_obj in slice_indexes:
        # Extract the slice range
        start, end = slice_obj.start, slice_obj.stop
        
        # Check if the slice has at least one sample
        if end > start:
            # Add to the valid_slices list only if there are samples
            valid_slices.append(slice_obj)
    
    return valid_slices

def str2columns (text,  regex=None , pattern = None): 
    """Split text from the non-alphanumeric markers using regular expression. 
    
    Remove all string non-alphanumeric and some operator indicators,  and 
    fetch attributes names. 
    
    Parameters 
    -----------
    text: str, 
        text litteral containing the columns the names to retrieve
        
    regex: `re` object,  
        Regular expresion object. the default is:: 
            
            >>> import re 
            >>> re.compile (r'[#&*@!_,;\s-]\s*', flags=re.IGNORECASE) 
    pattern: str, default = '[#&*@!_,;\s-]\s*'
        The base pattern to split the text into a columns
        
    Returns
    -------
    attr: List of attributes 
    
    Examples
    ---------
    >>> from gofast.tools.coreutils import str2columns 
    >>> text = ('this.is the text to split. It is an: example of; splitting str - to text.')
    >>> str2columns (text )  
    ... ['this',
         'is',
         'the',
         'text',
         'to',
         'split',
         'It',
         'is',
         'an:',
         'example',
         'of',
         'splitting',
         'str',
         'to',
         'text']

    """
    pattern = pattern or  r'[#&.*@!_,;\s-]\s*'
    regex = regex or re.compile (pattern, flags=re.IGNORECASE) 
    text= list(filter (None, regex.split(str(text))))
    return text 

def validate_ratio(
    value: float, 
    bounds: Optional[Tuple[float, float]] = None, 
    exclude: Optional[float] = None, 
    to_percent: bool = False, 
    param_name: str = 'value'
) -> float:
    """Validates and optionally converts a value to a percentage within 
    specified bounds, excluding specific values.

    Parameters:
    -----------
    value : float or str
        The value to validate and convert. If a string with a '%' sign, 
        conversion to percentage is attempted.
    bounds : tuple of float, optional
        A tuple specifying the lower and upper bounds (inclusive) for the value. 
        If None, no bounds are enforced.
    exclude : float, optional
        A specific value to exclude from the valid range. If the value matches 
        'exclude', a ValueError is raised.
    to_percent : bool, default=False
        If True, the value is converted to a percentage 
        (assumed to be in the range [0, 100]).
    param_name : str, default='value'
        The parameter name to use in error messages.

    Returns:
    --------
    float
        The validated (and possibly converted) value.

    Raises:
    ------
    ValueError
        If the value is outside the specified bounds, matches the 'exclude' 
        value, or cannot be converted as specified.
    """
    if isinstance(value, str) and '%' in value:
        to_percent = True
        value = value.replace('%', '')
    try:
        value = float(value)
    except ValueError:
        raise ValueError(f"Expected a float, got {type(value).__name__}: {value}")

    if to_percent and 0 < value <= 100:
        value /= 100

    if bounds:
        if not (bounds[0] <= value <= bounds[1]):
            raise ValueError(f"{param_name} must be between {bounds[0]}"
                             f" and {bounds[1]}, got: {value}")
    
    if exclude is not None and value == exclude:
        raise ValueError(f"{param_name} cannot be {exclude}")

    if to_percent and value > 1:
        raise ValueError(f"{param_name} converted to percent must"
                         f" not exceed 1, got: {value}")

    return value

def count_functions(
    module_name, 
    include_class=False, 
    return_counts=True, 
    include_private=False, 
    include_local=False
    ):
    """
    Count and list the number of functions and classes in a specified module.

    Parameters
    ----------
    module_name : str
        The name of the module to inspect, in the format `package.module`.
    include_class : bool, optional
        Whether to include classes in the count and listing. Default is 
        `False`.
    return_counts : bool, optional
        Whether to return only the count of functions and classes (if 
        ``include_class`` is `True`). If `False`, returns a list of functions 
        and classes in alphabetical order. Default is `True`.
    include_private : bool, optional
        Whether to include private functions and classes (those starting with 
        `_`). Default is `False`.
    include_local : bool, optional
        Whether to include local (nested) functions in the count and listing. 
        Default is `False`.

    Returns
    -------
    int or list
        If ``return_counts`` is `True`, returns the count of functions and 
        classes (if ``include_class`` is `True`). If ``return_counts`` is 
        `False`, returns a list of function and class names (if 
        ``include_class`` is `True`) in alphabetical order.

    Notes
    -----
    This function dynamically imports the specified module and analyzes its 
    Abstract Syntax Tree (AST) to count and list functions and classes. It 
    provides flexibility to include or exclude private and local functions 
    based on the parameters provided.

    The process can be summarized as:

    .. math::
        \text{total\_count} = 
        \text{len(functions)} + \text{len(classes)}

    where:

    - :math:`\text{functions}` is the list of functions found in the module.
    - :math:`\text{classes}` is the list of classes found in the module 
      (if ``include_class`` is `True`).

    Examples
    --------
    >>> from gofast.api.util import count_functions_classes
    >>> count_functions_classes('gofast.api.util', include_class=True,
                                return_counts=True)
    10

    >>> count_functions('gofast.api.util', include_class=True,
                                return_counts=False)
    ['ClassA', 'ClassB', 'func1', 'func2', 'func3']

    >>> count_functions('gofast.api.util', include_class=False, 
                                return_counts=True, include_private=True)
    15

    >>> count_functions('gofast.api.util', include_class=False, 
                                return_counts=False, include_private=True)
    ['_private_func1', '_private_func2', 'func1', 'func2']

    See Also
    --------
    ast : Abstract Syntax Tree (AST) module for parsing Python source code.

    References
    ----------
    .. [1] Python Software Foundation. Python Language Reference, version 3.9. 
       Available at http://www.python.org
    .. [2] Python `ast` module documentation. Available at 
       https://docs.python.org/3/library/ast.html
    """
 
    try:
        import ast
    except ImportError as e:  # Catch the specific ImportError exception
        raise ImportError(
            "The 'ast' module could not be imported. This module is essential"
            " for analyzing Python source code to count functions and classes."
            " Ensure that you are using a standard Python distribution, which"
            " includes the 'ast' module by default."
        ) from e

    import inspect
    import importlib
    # Import the module dynamically
    module = importlib.import_module(module_name)

    # Get the source code of the module
    source = inspect.getsource(module)

    # Parse the source code into an AST
    tree = ast.parse(source)

    # Initialize lists to store function and class names
    functions = []
    classes = []

    def is_local_function(node):
        """Determine if the function is local (nested)."""
        while node:
            if isinstance(node, ast.FunctionDef):
                return True
            node = getattr(node, 'parent', None)
        return False

    # Add parent references to each node
    for node in ast.walk(tree):
        for child in ast.iter_child_nodes(node):
            child.parent = node

    # Traverse the AST to find function and class definitions
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            if (include_private or not node.name.startswith('_')) and \
               (include_local or not is_local_function(node.parent)):
                functions.append(node.name)
        elif isinstance(node, ast.ClassDef) and include_class:
            if include_private or not node.name.startswith('_'):
                classes.append(node.name)

    # Combine and sort the lists if needed
    if include_class:
        result = sorted(functions + classes)
    else:
        result = sorted(functions)

    if return_counts:
        return len(result)
    else:
        return result

def smart_format(iter_obj: Any, choice: str = 'and') -> str:
    """
    Smartly formats an iterable object into a string with proper conjunction.

    Parameters
    ----------
    iter_obj : Any
        The iterable object to format. If not iterable, it is returned as a 
        string.
    choice : str, optional
        The conjunction to use ('and' or 'or'). Default is 'and'.

    Returns
    -------
    str
        A formatted string with elements separated by commas and the chosen 
        conjunction.

    Examples
    --------
    >>> from hwm.utils._core import smart_format
    >>> smart_format(['model', 'iter', 'mesh', 'data'])
    "'model', 'iter', 'mesh' and 'data'"
    >>> smart_format(['apple'])
    "'apple'"
    >>> smart_format('single_item')
    "'single_item'"
    >>> smart_format([])
    ''

    """
    # Check if the object is iterable but not a string or bytes
    if not isinstance(iter_obj, Iterable) or isinstance(iter_obj, (str, bytes)):
        return repr(iter_obj)

    # Create a list of string representations of the items
    items = [repr(item) for item in iter_obj]

    if not items:
        return ''

    if len(items) == 1:
        return items[0]

    # Join all items except the last with commas, then add the choice before 
    # the last item
    return ', '.join(items[:-1]) + f' {choice} {items[-1]}'

