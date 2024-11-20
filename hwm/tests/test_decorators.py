# -*- coding: utf-8 -*-
# tests/test_decorators.py

import warnings
import pytest
from hwm.estimators import HWRegressor, HammersteinWienerRegressor
from hwm.decorators import copy_doc, append_inherited_doc

def test_copy_doc_class():
    @copy_doc(
        source=HWRegressor,
        docstring="Additional info.",
        replace=False,
        copy_attrs=None
    )
    class TestRegressor(HWRegressor):
        pass

    assert "HWRegressor" in TestRegressor.__doc__
    assert "Additional info." in TestRegressor.__doc__


def test_copy_doc_function():
    def source_function():
        """Source function docstring."""
        pass

    @copy_doc(
        source=source_function,
        docstring="Decorator added info.",
        replace=False,
        copy_attrs=None
    )
    def decorated_function():
        pass

    assert "Source function docstring." in decorated_function.__doc__
    assert "Decorator added info." in decorated_function.__doc__


def test_copy_doc_replace():
    def source_function():
        """Source function docstring."""
        pass

    @copy_doc(
        docstring="Replaced docstring.",
        replace=True,
        copy_attrs=None
    )
    def decorated_function():
        pass

    assert decorated_function.__doc__ == "Replaced docstring."


def test_copy_attrs():
    class Source:
        attribute = "Copied attribute"

    @copy_doc(
        source=Source,
        copy_attrs=["attribute"],
        docstring=None,
        replace=False
    )
    class Decorated(Source):
        pass

    assert Decorated.attribute == "Copied attribute"


def test_deprecation_warning():
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        regressor = HammersteinWienerRegressor(batch_size=5000)
        assert len(w) == 1
        assert issubclass(w[-1].category, DeprecationWarning)
        assert "deprecated and will be removed" in str(w[-1].message)
        
        

def test_append_inherited_doc_append():
    @append_inherited_doc()
    class TestRegressor(HWRegressor):
        """
        TestRegressor specific documentation.
        """
        pass

    assert "HWRegressor" in TestRegressor.__doc__
    assert "TestRegressor specific documentation." in TestRegressor.__doc__


def test_append_inherited_doc_prepend():
    @append_inherited_doc(prepend=True, append=False, separator='\n\n')
    class TestRegressor(HWRegressor):
        """
        TestRegressor specific documentation.
        """
        pass

    assert "HWRegressor" in TestRegressor.__doc__
    assert "TestRegressor specific documentation." in TestRegressor.__doc__


def test_append_inherited_doc_specify_source():
    class SomeOtherClass:
        """SomeOtherClass's docstring."""

    @append_inherited_doc(inherit_from=SomeOtherClass())
    class TestRegressor(HWRegressor):
        """
        TestRegressor specific documentation.
        """
        pass

    assert "SomeOtherClass's docstring." in TestRegressor.__doc__
    assert "TestRegressor specific documentation." in TestRegressor.__doc__


def test_append_inherited_doc_copy_attrs():
    class Source:
        attribute = "Copied attribute"

    @append_inherited_doc(copy_attrs=["attribute"])
    class Decorated(Source):
        """
        Decorated class documentation.
        """
        pass

    assert Decorated.attribute == "Copied attribute"


def test_append_inherited_doc_replace():
    @append_inherited_doc(docstring="Replaced docstring.")
    class TestClass:
        """
        Original docstring.
        """
        pass

    assert "Replaced docstring." in TestClass.__doc__ 


def test_append_inherited_doc_error_both_append_prepend():
    with pytest.raises(ValueError, match="Cannot set both `append` and `prepend` to True."):
        @append_inherited_doc(append=True, prepend=True)
        class TestClass:
            """
            TestClass documentation.
            """
            pass

def test_deprecation_warning():
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        regressor = HammersteinWienerRegressor(batch_size=5000)
        assert len(w) == 1
        assert issubclass(w[-1].category, DeprecationWarning)
        assert "deprecated and will be removed" in str(w[-1].message)

if __name__=='__main__': 
    pytest.main([__file__])