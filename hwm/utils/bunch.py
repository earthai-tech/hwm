# -*- coding: utf-8 -*-
#   License: BSD-3-Clause
#   Author: LKouadio <etanoyau@gmail.com>

import numpy as np 
import pandas as pd 

__all__=["Boxspace"]

class Boxspace(dict):
    """
    A container object that extends dictionaries by enabling attribute-like 
    access to its items.
    
    `Boxspace` allows accessing values using the standard dictionary key 
    access method or directly as attributes. This feature provides a more 
    convenient and intuitive way to handle data, especially when dealing with 
    configurations or loosely structured objects.
    
    Examples
    --------
    >>> from hwm.utils.bunch import Boxspace
    >>> bs = Boxspace(pkg='hwm', objective='dynamic model', version='0.1.dev')
    >>> bs['pkg']
    'hwm'
    >>> bs.pkg
    'hwm'
    >>> bs.objective
    'dynamic model'
    >>> bs.version
    '0.1.dev'
    
    Notes
    -----
    While `Boxspace` provides a flexible way to access dictionary items, it's 
    important to ensure that key names do not conflict with the dictionary's 
    method names, as this could lead to unexpected behavior.
    """

    def __init__(self, **kwargs):
        """
        Initializes a Boxspace object with optional keyword arguments.
        
        Parameters
        ----------
        **kwargs : dict
            Arbitrary keyword arguments which are set as the initial 
            items of the dictionary.
        """
        super().__init__(**kwargs)

    def __getattr__(self, key):
        """
        Allows attribute-like access to dictionary items.
        
        Parameters
        ----------
        key : str
            The attribute name corresponding to the dictionary key.
        
        Returns
        -------
        The value associated with 'key' in the dictionary.
        
        Raises
        ------
        AttributeError
            If the key is not found in the dictionary.
        """
        try:
            return self[key]
        except KeyError:
            raise AttributeError(f"'Boxspace' object has no attribute '{key}'")

    def __setattr__(self, key, value):
        """
        Allows setting dictionary items as attributes.
        
        Parameters
        ----------
        key : str
            The attribute name to be added or updated in the dictionary.
        value : any
            The value to be associated with 'key'.
        """
        self[key] = value

    def __setstate__(self, state):
        """
        Overrides __setstate__ to ensure the object can be unpickled correctly.
        
        This method is a no-op, effectively ignoring the pickled __dict__, which is
        necessary because `Boxspace` objects use the dictionary itself for item storage.
        """
        pass

    def __dir__(self):
        """
        Ensures that autocompletion works in interactive environments.
        
        Returns
        -------
        list
            A list of keys in the dictionary, which are exposed as attributes.
        """
        return super().__dir__() + list(self.keys()) # self.keys()

    def __repr__(self):
        """
        Provides a detailed string representation of the Boxspace object.
        
        Returns
        -------
        str
            A string representation of the Boxspace object including its type 
            and key-value pairs.
        """
        keys = ', '.join(list(self.keys()))
        return f"<Bunch object with keys: {keys}>"

    def __str__(self):
        """
        Provides a user-friendly string representation of the Boxspace object.
        
        Returns
        -------
        str
            A string representation of the Boxspace object showing its key-value pairs.
        """
        dict_o = {k:v for k, v in self.items() if '__' not in str(k)}
        return format_dict_result(dict_o, dict_name="Bunch", include_message=True) 

        
def format_dict_result(
    dictionary, dict_name='Container', 
    max_char=50, 
    include_message=False):
    """
    Formats a dictionary into a string with specified formatting rules.

    Parameters
    ----------
    dictionary : dict
        The dictionary to format.
    dict_name : str, optional
        The name of the dictionary, by default 'Container'.
    max_char : int, optional
        The maximum number of characters for each value before truncating, 
        by default 50.
    include_message : bool, optional
        Whether to include a remainder message at the end, by default False.

    Returns
    -------
    str
        The formatted string representation of the dictionary.

    Examples
    --------
    >>> example_dict = {
    ...     'key1': 'short value',
    ...     'key2': 'a much longer value that should be truncated for readability purposes',
    ...     'key3': 'another short value',
    ...     'key4': 'value'
    ... }
    >>> print(format_dict_result(example_dict, dict_name='ExampleDict', max_char=30))
    ExampleDict({
        key1: short value,
        key2: a much longer value that s...,
        key3: another short value,
        key4: value,
    })

    Notes
    -----
    The function calculates the required indentation based on the length of the 
    dictionary name and the maximum key length. If a value exceeds the specified 
    maximum length, it truncates the value and appends an ellipsis ("...").
    """
    max_key_length = max(len(str(key)) for key in dictionary.keys())
    formatted_lines = [f"{dict_name}({{"]
    
    for key, value in dictionary.items():
        if (
                isinstance(value, value.__class__)
                and not hasattr(value, '__array__')
                and not isinstance(value, (str, list, tuple))
        ):
            try:
                formatted_value = value.__class__.__name__
            except:
                formatted_value = value.__name__
        else:
            formatted_value = format_iterable(value)
            
        if len(formatted_value) > max_char:
            formatted_value = formatted_value[:max_char - 3] + "..."
        formatted_lines.append(
            f"{' ' * (len(dict_name) + 2)}{key:{max_key_length}}: {formatted_value},")
    
    formatted_lines.append(" " * (len(dict_name) + 1) + "})")
    
    remainder = f"[Use <{dict_name}.key> to get the full value ...]"  
    
    return ( "\n".join(formatted_lines) + f"\n\n{remainder}" 
            if include_message else "\n".join(formatted_lines)
            )

def format_iterable(attr):
    """
    Formats an iterable with a string representation that includes
    statistical or structural information depending on the iterable's type.
    """
    def _numeric_stats(iterable):
        return {
            'min': round(np.min(iterable), 4),
            'max': round(np.max(iterable), 4),
            'mean': round(np.mean(iterable), 4),
            'len': len(iterable)
        }
    
    def _format_numeric_iterable(iterable):
        stats = _numeric_stats(iterable)
        return ( 
            f"{type(iterable).__name__} (min={stats['min']},"
            f" max={stats['max']}, mean={stats['mean']}, len={stats['len']})"
            )

    def _format_ndarray(array):
        stats = _numeric_stats(array.flat) if np.issubdtype(array.dtype, np.number) else {}
        details = ", ".join([f"{key}={value}" for key, value in stats.items()])
        return f"ndarray ({details}, shape={array.shape}, dtype={array.dtype})"
    
    def _format_pandas_object(obj):
        if isinstance(obj, pd.Series):
            stats = _numeric_stats(obj) if obj.dtype != 'object' else {}
            details = ", ".join([f"{key}={value}" for key, value in stats.items()])
            if details: 
                details +=', '
            return f"Series ({details}len={obj.size}, dtype={obj.dtype})"
        elif isinstance(obj, pd.DataFrame):
            numeric_cols = obj.select_dtypes(include=np.number).columns
            stats = _numeric_stats(obj[numeric_cols].values.flat) if not numeric_cols.empty else {}
            details = ", ".join([f"{key}={value}" for key, value in stats.items()])
            if details: 
                details +=', '
            return ( 
                f"DataFrame ({details}n_rows={obj.shape[0]},"
                f" n_cols={obj.shape[1]}, dtypes={obj.dtypes.unique()})"
                )
    
    if isinstance(attr, (list, tuple, set)) and all(
            isinstance(item, (int, float)) for item in attr):
        return _format_numeric_iterable(attr)
    elif isinstance(attr, np.ndarray):
        return _format_ndarray(attr)
    elif isinstance(attr, (pd.Series, pd.DataFrame)):
        return _format_pandas_object(attr)
    
    return str(attr)