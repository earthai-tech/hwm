# -*- coding: utf-8 -*-


class NotFittedError(Exception):
    """
    Exception raised when a 'fit' method is called on an unfitted object.

    This is a common exception for classes in `gofast` that implement a 'fit'
    method for parameter initialization.
    """
    pass