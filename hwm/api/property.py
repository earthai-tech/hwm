# -*- coding: utf-8 -*-
#   License: BSD-3-Clause
#   Author: LKouadio <etanoyau@gmail.com>

from __future__ import annotations

import inspect 

from functools import wraps
from abc import ABCMeta

from types import FunctionType, MethodType # noqa 
# from typing import Any, Callable, Dict, Iterable, List, Tuple, Union, Optional 

__all__= ["HelpMeta"]

class DisplayStr(str):
    """
    A string subclass that displays its content without quotes when evaluated.

    This class is used to ensure that strings display their content directly
    when printed or evaluated in an interactive shell, without enclosing quotes.
    """

    def __repr__(self):
        return str(self)


class NoOutput:
    """
    A class that suppresses output when returned in an interactive shell.

    When an instance of this class is returned from a function, it ensures
    that no output is displayed in the interactive shell (e.g., IPython, Jupyter).
    """

    def __repr__(self):
        return ''

    def __str__(self):
        return ''


class HelpMeta(type):
    """
    Metaclass that adds `my_params` and `help` attributes to classes and methods.

    This metaclass enhances classes by automatically adding `my_params` and `help`
    attributes to the class itself and its methods. The `my_params` attribute
    provides a formatted string of the class or method parameters, excluding
    common parameters like `self`, `cls`, `*args`, and `**kwargs`. The `help`
    attribute provides a convenient way to display the documentation of the
    class or method.

    Parameters
    ----------
    name : str
        The name of the class being created.

    bases : tuple of type
        The base classes of the class being created.

    namespace : dict
        A dictionary containing the class's namespace.

    Class Attributes
    ----------------
    MAX_ITEMS_DISPLAY : int
        Default maximum number of parameters to display inline before switching
        to vertical formatting.

    Methods
    -------
    __new__(mcs, name, bases, namespace)
        Creates a new class with enhanced attributes.

    Examples
    --------
    >>> from hwm.api.property import HelpMeta
    >>> class Example(metaclass=HelpMeta):
    ...     \"\"\"
    ...     An example class to demonstrate HelpMeta functionality.
    ...
    ...     Parameters
    ...     ----------
    ...     a : int
    ...         First parameter.
    ...     b : int, optional
    ...         Second parameter, default is 2.
    ...     c : int, optional
    ...         Third parameter, default is 3.
    ...     \"\"\"
    ...     def __init__(self, a, b=2, c=3, d=4, e=5, f=6):
    ...         pass
    ...     def my_method(self, x, y=10):
    ...         \"\"\"A custom method.\"\"\"
    ...         pass
    ...     @staticmethod
    ...     def my_static_method(p, q=20):
    ...         \"\"\"A static method.\"\"\"
    ...         pass
    ...     @classmethod
    ...     def my_class_method(cls, s, t=30):
    ...         \"\"\"A class method.\"\"\"
    ...         pass
    ...
    >>> Example.my_params
    Example(
        a,
        b=2,
        c=3,
        d=4,
        e=5,
        f=6
    )
    >>> Example.help()
    Help on class Example in module __main__:
    <...help output...>
    >>> Example.my_method.my_params
    Example.my_method(x, y=10)
    >>> Example.my_method.help()
    Help on function my_method in module __main__:
    <...help output...>
    >>> Example.my_static_method.my_params
    Example.my_static_method(p, q=20)
    >>> Example.my_static_method.help()
    Help on function my_static_method in module __main__:
    <...help output...>
    >>> Example.my_class_method.my_params
    Example.my_class_method(s, t=30)
    >>> Example.my_class_method.help()
    Help on function my_class_method in module __main__:
    <...help output...>

    Notes
    -----
    The `HelpMeta` metaclass is designed to provide a user-friendly API by
    making parameter information and documentation easily accessible. It is
    particularly useful in interactive environments.

    See Also
    --------
    inspect.signature : Get a signature object for the callable.

    References
    ----------
    .. [1] Python documentation on metaclasses:
           https://docs.python.org/3/reference/datamodel.html#metaclasses
    """

    MAX_ITEMS_DISPLAY = 5  # Default maximum items to display inline

    def __new__(mcs, name, bases, namespace):

        cls = super(HelpMeta, mcs).__new__(mcs, name, bases, namespace)

        # Add 'my_params' attribute to the class
        cls.my_params = mcs._get_my_params(cls.__init__)
        cls.my_params = DisplayStr(cls.my_params)  # Ensure it displays nicely

        # Add 'help' method to the class
        cls.help = mcs._create_help(cls)

        # Decorate all methods to have 'my_params' and 'help'
        for attr_name, attr_value in namespace.items():
            if isinstance(attr_value, (FunctionType, staticmethod, classmethod)):
                decorated_method = mcs._decorate_method(attr_value)
                setattr(cls, attr_name, decorated_method)

        return cls

    @classmethod
    def _get_my_params(mcs, func):
        """
        Retrieves the parameters of the function and formats them.

        Parameters are displayed inline if their number is less than or equal
        to MAX_ITEMS_DISPLAY; otherwise, they are displayed vertically.

        Excludes 'self', 'cls', '*args', and '**kwargs' from the parameter list.
        """
        sig = inspect.signature(func)
        params = sig.parameters

        param_strings = []
        for name, param in params.items():
            # Exclude 'self', 'cls', '*args', and '**kwargs'
            if name in ('self', 'cls'):
                continue
            if param.kind in (
                    inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
                continue
            if param.default is inspect.Parameter.empty:
                param_strings.append(f"{name}")
            else:
                param_strings.append(f"{name}={param.default!r}")

        # Use the class name for '__init__', otherwise use the full function name
        if func.__name__ == '__init__':
            func_name = func.__qualname__.split('.')[0]
        else:
            func_name = func.__qualname__

        if len(param_strings) <= mcs.MAX_ITEMS_DISPLAY:
            # Inline display
            params_formatted = ", ".join(param_strings)
            return f"{func_name}({params_formatted})"
        else:
            # Vertical display
            params_formatted = ",\n    ".join(param_strings)
            return f"{func_name}(\n    {params_formatted}\n)"

    @staticmethod
    def _create_help(obj):
        """
        Creates a method that, when called, displays the help of the object.
        """
        def help_method(*args, **kwargs):
            help(obj)
            return NoOutput()  # Suppress 'None' output
        return help_method
    
    @classmethod
    def _decorate_method(mcs, method):
        """
        Decorator that adds 'my_params' and 'help' attributes to methods.
    
        This method decorates and wraps the original method to add `my_params` 
        and `help` attributes, which provide additional introspection 
        capabilities. It determines if the method is a `staticmethod`, 
        `classmethod`, or a regular instance method and applies the appropriate 
        decorator to preserve its behavior. The `my_params` attribute shows 
        details of the method's parameters, while the `help` attribute provides 
        a quick way to access the method's documentation.
    
        Parameters
        ----------
        method : function or method
            The original method or function that needs to be decorated with 
            `my_params` and `help` attributes.
    
        Returns
        -------
        decorated_method : function or method
            The wrapped method, now with `my_params` and `help` attributes, 
            either as a `staticmethod`, `classmethod`, or a regular method.
        """
        # Case 1: If method is a staticmethod
        if isinstance(method, staticmethod):
            # Retrieve the original function behind the staticmethod decorator
            original_func = method.__func__
    
            # Define a wrapper for the original function
            @wraps(original_func)
            def wrapper(*args, **kwargs):
                return original_func(*args, **kwargs)
    
            # Attach 'my_params' and 'help' to the wrapper
            wrapper.my_params = mcs._get_my_params(original_func)
            wrapper.my_params = DisplayStr(wrapper.my_params)
            wrapper.help = mcs._create_help(original_func)
            return staticmethod(wrapper)
    
        # Case 2: If method is a classmethod
        elif isinstance(method, classmethod):
            # Retrieve the original function behind the classmethod decorator
            original_func = method.__func__
    
            # Define a wrapper for the original function
            @wraps(original_func)
            def wrapper(cls, *args, **kwargs):
                return original_func(cls, *args, **kwargs)
    
            # Attach 'my_params' and 'help' to the wrapper
            wrapper.my_params = mcs._get_my_params(original_func)
            wrapper.my_params = DisplayStr(wrapper.my_params)
            wrapper.help = mcs._create_help(original_func)
            return classmethod(wrapper)
    
        # Case 3: If method is a regular instance method
        elif isinstance(method, FunctionType):
            # Define a wrapper for the regular function
            @wraps(method)
            def wrapper(self, *args, **kwargs):
                return method(self, *args, **kwargs)
    
            # Attach 'my_params' and 'help' to the wrapper
            wrapper.my_params = mcs._get_my_params(method)
            wrapper.my_params = DisplayStr(wrapper.my_params)
            wrapper.help = mcs._create_help(method)
            return wrapper
    
        # Case 4: If method is not recognized, return it unchanged
        else:
            return method
        
class LearnerMeta(ABCMeta, HelpMeta):
    """
    A metaclass that combines functionality from ABCMeta and HelpMeta.
    This allows classes using LearnerMeta to support abstract methods and
    to have enhanced introspection features from HelpMeta. 
    """
    pass 

