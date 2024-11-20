# -*- coding: utf-8 -*-
# License: BSD-3-Clause
# Author: L. Kouadio <etanoyau@gmail.com>

"""
HWM: Adaptive Hammerstein-Wiener Modeling Toolkit
=================================================

`hwm` provides a flexible and modular toolkit for dynamic system modeling, 
nonlinear regression, and classification tasks. It is built to enhance 
productivity in time-series and structured data applications, 
with compatibility across popular Python data science libraries.

This package adheres to Scikit-learn conventions, offering easy integration 
with Scikit-learn models and APIs.
"""

import os
import logging
import warnings
import importlib

# Configure logging and suppress specific third-party library warnings
logging.basicConfig(level=logging.WARNING)
logging.getLogger('matplotlib.font_manager').disabled = True

# Helper function for lazy imports to improve load times
def _lazy_import(module_name, alias=None):
    """Lazily import a module to optimize initial package load time."""
    def _lazy_loader():
        return importlib.import_module(module_name)
    if alias:
        globals()[alias] = _lazy_loader
    else:
        globals()[module_name] = _lazy_loader

# Package version
try:
    from ._version import version as __version__
except ImportError:
    __version__ = "1.1.0"

# Dependency management
_required_dependencies = [
    ("numpy", None),
    ("pandas", None),
    ("scipy", None),
    ("sklearn", "scikit-learn"),
]

_missing_dependencies = []
for package, import_name in _required_dependencies:
    try:
        if import_name:
            _lazy_import(import_name, package)
        else:
            _lazy_import(package)
    except ImportError as e:
        _missing_dependencies.append(f"{package}: {str(e)}")

if _missing_dependencies:
    warnings.warn(
        "Some dependencies are missing. `hwm` may not function correctly:\n" +
        "\n".join(_missing_dependencies), ImportWarning
    )

# Suppression of non-critical warnings, adjustable by the user
_warnings_state = {"FutureWarning": "ignore"}
def suppress_warnings(suppress=True):
    """Enable or disable future/syntax warnings."""
    for warning, action in _warnings_state.items():
        if suppress:
            warnings.filterwarnings(action, category=FutureWarning)
            warnings.filterwarnings(action, category=SyntaxWarning)
        else:
            warnings.filterwarnings("default", category=FutureWarning)

suppress_warnings()

# Disable oneDNN custom operations for TensorFlow if necessary
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Additional documentation information
__doc__ += f"\nVersion: {__version__}\n"

# Public API for external access
__all__ = ["__version__"]
