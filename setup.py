# -*- coding: utf-8 -*-
from setuptools import setup, find_packages

# Determine package version
try:
    import hwm
    VERSION = hwm.__version__
except ImportError:
    VERSION = '1.1.0'


# Package metadata
DISTNAME = 'hwm'
DESCRIPTION = 'Adaptive Hammerstein-Wiener Modeling Toolkit'
with open('README.md', 'r', encoding='utf8') as f:
    LONG_DESCRIPTION = f.read()
MAINTAINER = "L.Kouadio"
MAINTAINER_EMAIL = 'etanoyau@gmail.com'
URL = "https://github.com/earthai-tech/hwm"
DOWNLOAD_URL = "https://pypi.org/project/hwm/#files"
LICENSE = "BSD-3-Clause"
PROJECT_URLS = {
    "API Documentation": "https://hwm.readthedocs.io/en/stable/api.html",
    "Home page": "https://hwm.readthedocs.io",
    "Bugs tracker": "https://github.com/earthai-tech/hwm/issues",
    "Installation guide": "https://hwm.readthedocs.io/en/stable/installation.html",
    "User guide": "https://hwm.readthedocs.io/en/lateststable/user_guide.html",
}
KEYWORDS = "machine learning, dynamic systems, algorithm, time series"

# Dependencies
INSTALL_REQUIRES = [
    "numpy<2",
    "scipy>=1.7.0",
    "scikit-learn>=1.2.0",
    "pandas>=1.3.0",
]

EXTRAS_REQUIRE = {
    "dev": [
        "pytest>=6.0.0",
        "sphinx>=4.0.0",
        "sphinx_rtd_theme",
        "tensorflow>=2.6.0",
    ],
    "examples": [
        "matplotlib>=3.4.0",
        "tensorflow>=2.6.0",
    ],
}

PYTHON_REQUIRES = '>=3.9'

CLASSIFIERS = [
    "Development Status :: 3 - Alpha",
    "Intended Audience :: Science/Research",
    "Intended Audience :: Developers",
    "Topic :: Software Development",
    "Topic :: Scientific/Engineering",
    "Programming Language :: Python",
    "Programming Language :: Python :: 3.9",
    "Programming Language :: Python :: 3.10",
    "Programming Language :: Python :: 3.11",
    "Programming Language :: Python :: Implementation :: CPython",
    "Operating System :: OS Independent",
]

setup(
    name=DISTNAME,
    version=VERSION,
    author=MAINTAINER,
    author_email=MAINTAINER_EMAIL,
    maintainer=MAINTAINER,
    maintainer_email=MAINTAINER_EMAIL,
    description=DESCRIPTION,
    long_description=LONG_DESCRIPTION,
    long_description_content_type="text/markdown",
    url=URL,
    download_url=DOWNLOAD_URL,
    project_urls=PROJECT_URLS,
    license=LICENSE,
    classifiers=CLASSIFIERS,
    keywords=KEYWORDS,
    packages=find_packages(),
    include_package_data=True,
    install_requires=INSTALL_REQUIRES,
    extras_require=EXTRAS_REQUIRE,
    python_requires=PYTHON_REQUIRES,
    zip_safe=False,
)
