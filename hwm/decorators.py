# -*- coding: utf-8 -*-
# License: BSD-3-Clause
# Author: LKouadio <etanoyau@gmail.com>

import inspect 
import warnings 
from functools import wraps 

__all__=['copy_doc', 'append_inherited_doc', 
         'enable_specific_warnings']


def enable_specific_warnings(
    cls_or_func=None, *,
    categories=FutureWarning, 
    messages=None
):
    """
    Enable specific warnings for classes, functions, or methods.

    This decorator allows selective enabling of specified warning categories 
    and message patterns for the decorated class, function, or method.
    By default, it enables `FutureWarning`, but it can be customized to 
    target other warning types and specific warning messages.

    .. math::
        \text{Enable specified warnings only during the execution of the decorated
        object.}

    Parameters
    ----------
    cls_or_func : type or callable, optional
        The class, function, or method to decorate. If omitted, the decorator can be
        applied with or without parentheses.

    categories : Warning subclass or list/tuple of Warning subclasses,\
        default :class:`FutureWarning`
        The warning category or categories to enable. Accepts a single 
        warning category or a list/tuple of categories.

        Examples of warning categories include:

        - :class:`FutureWarning`
        - :class:`DeprecationWarning`
        - :class:`UserWarning`

    messages : str or list/tuple of str, optional
        Specific warning message patterns to enable. Accepts a single string 
        or a list/tuple of strings. Supports wildcard patterns using `*`.

        For example:

        - `"HammersteinWienerRegressor is deprecated*"`
        - `"old_function is deprecated*"`

    Returns
    -------
    type or callable
        The decorated class, function, or method with specific warnings enabled.

    Examples
    --------
    Enable `FutureWarning` for a deprecated class:

    >>> from hwm.decorators import enable_specific_warnings
    >>> 
    >>> @enable_specific_warnings(
    ...     categories=FutureWarning,
    ...     messages="HammersteinWienerRegressor is deprecated*"
    ... )
    ... class HammersteinWienerRegressor(HWRegressor):
    ...     def __init__(self, *args, **kwargs):
    ...         warnings.warn(
    ...             "HammersteinWienerRegressor is deprecated and will be removed "
    ...             "in version 1.2. Use HWRegressor instead.",
    ...             FutureWarning,
    ...             stacklevel=2
    ...         )
    ...         super().__init__(*args, **kwargs)
    ...
    
    Enable `DeprecationWarning` for a deprecated function:

    >>> from hwm.decorators import enable_specific_warnings
    >>> 
    >>> @enable_specific_warnings(
    ...     categories=DeprecationWarning,
    ...     messages="old_function is deprecated*"
    ... )
    ... def old_function():
    ...     warnings.warn(
    ...         "old_function is deprecated and will be removed in future versions.",
    ...         DeprecationWarning,
    ...         stacklevel=2
    ...     )
    ...     # Function implementation
    ...     pass
    ...
    
    Enable multiple warning categories for a class method:

    >>> from hwm.decorators import enable_specific_warnings
    >>> 
    >>> class ExampleClass:
    ...     @enable_specific_warnings(
    ...         categories=[UserWarning, DeprecationWarning],
    ...         messages=["deprecated_method is deprecated*", "old_behavior*"]
    ...     )
    ...     def deprecated_method(self):
    ...         warnings.warn(
    ...             "deprecated_method is deprecated and will be removed soon.",
    ...             UserWarning,
    ...             stacklevel=2
    ...         )
    ...         # Method implementation
    ...         pass
    ...

    Notes
    -----
    - The decorator temporarily modifies the warning filters within the scope of the
      decorated object's execution. This ensures that only the specified warnings are
      enabled, without affecting the global warning settings.
    
    - When decorating a class, the warning filters are applied during the instantiation
      of the class (i.e., within the `__init__` method).
    
    - When decorating a function or method, the warning filters are applied during the
      execution of the function or method.

    - The `messages` parameter supports wildcard patterns using `*` for flexible
      matching of warning messages.

    See Also
    --------
    warnings.simplefilter : Control the behavior of warnings.
    warnings.catch_warnings : Context manager to temporarily modify warning filters.

    References
    ----------
    .. [1] Python Documentation on `warnings` module:
       https://docs.python.org/3/library/warnings.html
    .. [2] Numpy Documentation on Docstrings:
       https://numpydoc.readthedocs.io/en/latest/format.html
    """
    def decorator(obj):
        if inspect.isclass(obj):
            obj.__init__ = _wrap_class_init(
                original_init=obj.__init__,
                categories=categories,
                messages=messages
            )
            return obj
        elif callable(obj):
            return _wrap_callable(
                func=obj,
                categories=categories,
                messages=messages
            )
        return obj

    if cls_or_func is None:
        return decorator
    else:
        return decorator(cls_or_func)

def _enable_warnings(categories, messages):
    if isinstance(categories, (list, tuple)):
        for category in categories:
            warnings.simplefilter("default", category)
    else:
        warnings.simplefilter("default", categories)
    
    if messages:
        if isinstance(messages, (list, tuple)):
            for msg in messages:
                warnings.filterwarnings(
                    "default",
                    message=msg
                )
        else:
            warnings.filterwarnings(
                "default",
                message=messages
            )

def _wrap_class_init(original_init, categories, messages):
    @wraps(original_init)
    def new_init(self, *args, **kwargs):
        with warnings.catch_warnings():
            _enable_warnings(categories, messages)
            original_init(self, *args, **kwargs)
    return new_init

def _wrap_callable(func, categories, messages):
    @wraps(func)
    def wrapper(*args, **kwargs):
        with warnings.catch_warnings():
            _enable_warnings(categories, messages)
            return func(*args, **kwargs)
    return wrapper


def copy_doc(
    source=None, 
    docstring=None, 
    replace=False, 
    copy_attrs=None
    ):
    """
    Class or function decorator to copy the docstring and specified attributes from a 
    source class or function to the decorated class or function.

    This decorator facilitates the transfer of documentation and attributes, ensuring 
    consistency and reducing redundancy. It is particularly useful when creating 
    aliases for deprecated classes or functions, allowing the new and deprecated 
    entities to share documentation seamlessly.

    .. math::
        \text{CopyDoc}(S, D) = 
        \begin{cases} 
            \text{Replace} & \text{if } \text{replace=True} \\
            \text{Append}  & \text{if } \text{replace=False}
        \end{cases}

    Parameters
    ----------
    source : class or function, optional
        The source class or function from which to copy the docstring and attributes.
        If provided, the decorator will copy the `__doc__` attribute and any attributes
        specified in `copy_attrs` from this source to the decorated object.

    docstring : str, optional
        An additional or alternative docstring to include in the decorated object.
        If `replace` is `False`, this docstring will be appended to the source's docstring.
        If `replace` is `True`, this docstring will replace the source's docstring.

    replace : bool, default=False
        Determines how the `docstring` parameter is applied.
        - If `True`, the existing docstring of the decorated object is replaced entirely 
          by the `docstring` parameter.
        - If `False`, the `docstring` parameter is appended to the existing docstring.

    copy_attrs : list of str, optional
        A list of attribute names to copy from the `source` to the decorated object.
        Only attributes listed in `copy_attrs` will be copied. If `None`, no additional
        attributes are copied beyond the docstring.

    Returns
    -------
    decorator : function
        The decorator function that applies the specified docstring and attribute 
        copying to the decorated class or function.

    Methods
    -------
    __call__(obj)
        Applies the decorator to the given class or function `obj`.

    Examples
    --------
    >>> from hwm.utils.decorators import copy_doc
    >>> from hwm.estimators import HWRegressor

    **Copying Docstring from a Class:**

    >>> @copy_doc(source=HWRegressor, docstring="Deprecated. Use HWRegressor instead.",
    ...              replace=False)
    ... class HammersteinWienerRegressor(HWRegressor):
    ...     def __init__(self, *args, **kwargs):
    ...         import warnings
    ...         warnings.warn(
    ...             "HammersteinWienerRegressor is deprecated and will be removed in version 1.1.3. "
    ...             "Use HWRegressor instead.",
    ...             DeprecationWarning,
    ...             stacklevel=2
    ...         )
    ...         super().__init__(*args, **kwargs)

    **Copying Docstring from a Function:**

    >>> def source_function():
    ...     '''Original source function docstring.'''
    ...     pass
    ...
    >>> @copy_doc(source=source_function, docstring="Additional information.", replace=False)
    ... def decorated_function():
    ...     pass
    ...
    >>> print(decorated_function.__doc__)
    Original source function docstring.

    Additional information.

    **Replacing Docstring Completely:**

    >>> @copy_doc(docstring="Completely new docstring.", replace=True)
    ... def new_function():
    ...     pass
    ...
    >>> print(new_function.__doc__)
    Completely new docstring.

    **Copying Specific Attributes:**

    >>> class Source:
    ...     attribute = "Copied attribute"
    ...
    >>> @copy_doc(source=Source, copy_attrs=["attribute"])
    ... class Decorated(Source):
    ...     pass
    ...
    >>> print(Decorated.attribute)
    Copied attribute

    Notes
    -----
    - The `copy_doc` decorator does not modify any methods or internal attributes 
      (those starting with an underscore) of the decorated class or function.
    - It is recommended to use this decorator primarily for creating aliases 
      for deprecated classes or functions to maintain documentation consistency.

    See Also
    --------
    warnings.warn : Issue warnings to users about deprecated features.

    References
    ----------
    .. [1] Python Documentation on [Decorators](https://docs.python.org/3/glossary.html#term-decorator).

    """
    def decorator(obj):
        """
        Apply the `copy_doc` decorator to the given object `obj`.

        Parameters
        ----------
        obj : class or function
            The class or function to which the docstring and attributes 
            will be copied.

        Returns
        -------
        obj : class or function
            The decorated object with updated docstring and attributes.
        """
        # Handle docstring copying
        if source:
            source_doc = source.__doc__ or ""
            if replace:
                combined_doc = docstring or ""
            else:
                combined_doc = f"{source_doc}\n\n{docstring}" if docstring else source_doc
        else:
            combined_doc = docstring or ""

        if combined_doc:
            obj.__doc__ = combined_doc

        # Handle attribute copying
        if source and copy_attrs:
            for attr in copy_attrs:
                if hasattr(source, attr):
                    setattr(obj, attr, getattr(source, attr))

        return obj

    return decorator


def append_inherited_doc(
    inherit_from=None,
    docstring=None,
    append=True,
    prepend=False,
    separator='\n\n',
    copy_attrs=None,
):
    """
    Decorator to inherit and combine the docstring from a base class or 
    a specified source with the decorated class's or function's docstring.

    .. math::
        \text{CombinedDoc}(C, S) = 
        \begin{cases} 
            S + \text{separator} + C & \text{if } \text{prepend=True} \\
            C + \text{separator} + S & \text{if } \text{append=True} \\
            C & \text{otherwise}
        \end{cases}

    Parameters
    ----------
    inherit_from : class or function, optional
        The source class or function from which to copy the docstring.
        If not provided, the decorator will use the first base class's docstring.
    docstring : str, optional
        An additional or alternative docstring to include in the decorated object.
        If `replace` is `False`, this docstring will be appended or prepended 
        based on the `append` and `prepend` flags.
        If `replace` is `True`, this docstring will replace the source's docstring.
    append : bool, default=True
        If `True`, append the source docstring to the decorated object's docstring.
    prepend : bool, default=False
        If `True`, prepend the source docstring to the decorated object's docstring.
    separator : str, default='\n\n'
        The string used to separate the source docstring and the decorated 
        object's docstring.
    copy_attrs : list of str, optional
        A list of attribute names to copy from the source to the decorated object.
        Only attributes listed in `copy_attrs` will be copied. If `None`, no 
        additional attributes are copied beyond the docstring.

    Returns
    -------
    decorator : function
        The decorator function that applies the specified docstring and attribute 
        copying to the decorated class or function.

    Examples
    --------
    **Using with Parentheses (Parameterized Decorator):**
    
    >>> from hwm.decorators import append_inherited_doc
    >>> from hwm.estimators import BaseHammersteinWiener, HWRegressor
    
    >>> _deprecated_docstring = '''
    ... .. deprecated:: 1.1.1
    ...     `HammersteinWienerRegressor` is deprecated and will be removed in 
    ...     version 1.1.3. Use `HWRegressor` instead.
    ... '''
    >>> 
    >>> @append_inherited_doc(
    ...     inherit_from=HWRegressor,
    ...     docstring=_deprecated_docstring.format(new_class="HWRegressor"),
    ...     append=True,
    ...     separator='\n\n',
    ...     copy_attrs=None
    ... )
    ... class HammersteinWienerRegressor(HWRegressor):
    ...     '''
    ...     HammersteinWienerRegressor specific documentation.
    ...     '''
    ...
    
    **Using without Parentheses (Default Parameters):**
    
    >>> from hwm.decorators import append_inherited_doc
    >>> from hwm.estimators import BaseHammersteinWiener, HWRegressor
    
    >>> @append_inherited_doc
    ... class HWRegressor(HammersteinWienerRegressor):
    ...     '''
    ...     HWRegressor specific documentation.
    ...     '''
    ...
    
    **Using as a Function Decorator:**
    
    >>> from hwm.utils.decorators import append_inherited_doc
    
    >>> def base_function():
    ...     '''Base function docstring.'''
    ...     pass
    ...
    >>> @append_inherited_doc(docstring="Decorated function additional info.")
    ... def decorated_function():
    ...     '''Decorated function docstring.'''
    ...     pass
    ...
    >>> print(decorated_function.__doc__)
    Base function docstring.
    
    Decorated function additional info.
    
    Notes
    -----
    - If both `append` and `prepend` are set to `True`, a `ValueError` is raised.
    - If `inherit_from` is not provided, the decorator uses the first base class's 
      docstring. Ensure that the class has at least one base class with a docstring.
    - The decorator does not modify any methods or internal attributes 
      (those starting with an underscore) of the decorated class or function.

    See Also
    --------
    warnings.warn : Issue warnings to users about deprecated features.

    References
    ----------
    .. [1] Python Documentation on [Decorators](https://docs.python.org/3/glossary.html#term-decorator).

    """
    def decorator(obj):
        """
        Apply the `append_inherited_doc` decorator to the given object `obj`.

        Parameters
        ----------
        obj : class or function
            The class or function to which the docstring and attributes 
            will be copied.

        Returns
        -------
        obj : class or function
            The decorated object with updated docstring and attributes.
        """
        # Validate parameters
        if append and prepend:
            raise ValueError("Cannot set both `append` and `prepend` to True.")

        # Determine the source docstring
        if inherit_from:
            source_doc = getattr(inherit_from, '__doc__', '') or ''
        else:
            # Attempt to get the first base class's docstring
            if hasattr(obj, '__bases__') and obj.__bases__:
                source_doc = obj.__bases__[0].__doc__ or ''
            else:
                source_doc = ''

        # Get the decorated object's docstring
        obj_doc = obj.__doc__ or ''

        # Combine docstrings based on append/prepend flags
        combined_doc = obj_doc
        if source_doc:
            if prepend:
                combined_doc = f"{source_doc}{separator}{obj_doc}"
            elif append:
                combined_doc = f"{obj_doc}{separator}{source_doc}"

        # If an additional docstring is provided, append or prepend it
        if docstring:
            if prepend:
                combined_doc = f"{docstring}{separator}{combined_doc}"
            elif append:
                combined_doc = f"{combined_doc}{separator}{docstring}"
            else:
                # If not appending or prepending, replace with the additional docstring
                combined_doc = docstring

        obj.__doc__ = combined_doc

        # Copy specified attributes from source to decorated object
        if copy_attrs:
            if inherit_from:
                source = inherit_from
            else:
                if hasattr(obj, '__bases__') and obj.__bases__:
                    source = obj.__bases__[0]
                else:
                    source = None

            if source:
                for attr in copy_attrs:
                    if hasattr(source, attr):
                        setattr(obj, attr, getattr(source, attr))

        return obj

    # Support usage without parentheses
    if callable(inherit_from):
        # The decorator was used without arguments
        obj = inherit_from
        inherit_from = None
        docstring = None
        append = False
        prepend = True
        separator = '\n\n'
        copy_attrs = None
        return decorator(obj)
    else:
        return decorator
