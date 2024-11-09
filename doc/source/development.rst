
.. _development.rst 

=========================
Development Guide
=========================

The `hwm` package is designed for advanced dynamic system modeling and nonlinear 
regression, following the Scikit-learn API conventions to ensure consistency, 
interoperability, and ease of use. This guide provides detailed instructions 
on setting up a development environment, contributing to the codebase, and 
maintaining code quality and style.

Getting Started
-----------------
To contribute to the `hwm` package, clone the source code from GitHub, 
create a virtual environment, and install the package in editable mode 
with the required development dependencies.

1. **Clone the repository**:

   .. code-block:: bash

       git clone https://github.com/earthai-tech/hwm.git
       cd hwm

2. **Create and activate a virtual environment**:

   .. code-block:: bash

       python3 -m venv hwm_dev_env
       source hwm_dev_env/bin/activate  # On Windows: hwm_dev_env\Scripts\activate

3. **Install in editable mode with development dependencies**:

   .. code-block:: bash

       pip install -e ".[dev]"

This installs `hwm` in editable mode, allowing any changes made to the 
source code to be immediately reflected in the local environment.

Development Workflow
----------------------
The `hwm` development process follows the Scikit-learn API conventions, 
with a strong emphasis on consistency, modularity, and testing. All 
contributions should adhere to the following workflow to ensure a smooth 
integration:

1. **Feature Branches**: Create a new feature branch for each feature, 
   bug fix, or enhancement.

   .. code-block:: bash

       git checkout -b feature/new_feature

2. **Code Changes**: Make incremental changes, regularly committing 
   with meaningful messages. Use modular design principles and 
   maintain compatibility with the Scikit-learn API. Ensure all public 
   classes and functions use clear, intuitive parameter names, and 
   adopt `fit`, `predict`, and `transform` where applicable.

3. **Documentation**: Document all code thoroughly. The `hwm` package 
   uses reStructuredText (reST) for documentation. All classes and 
   functions should include detailed docstrings in the NumPy style, 
   with examples where possible.

4. **Testing**: Add tests for any new functionality to ensure code 
   reliability. The `hwm` package uses `pytest` for testing. Place 
   all test files in the `tests/` directory, with each module 
   having a corresponding test file. To run the test suite:

   .. code-block:: bash

       pytest tests/

5. **Code Quality**: Ensure code quality with `flake8` and formatting 
   standards using `black`:

   .. code-block:: bash

       flake8 hwm/
       black hwm/

6. **Commit and Push**: Once changes are finalized, commit and push 
   your changes.

   .. code-block:: bash

       git add .
       git commit -m "Added new feature"
       git push origin feature/new_feature

7. **Pull Request**: Open a pull request (PR) on GitHub, describing 
   the feature, fix, or enhancement. Link to any relevant issues and 
   request a code review.

API Conventions
-----------------
The `hwm` package adheres to the Scikit-learn API conventions to 
maintain consistency and usability. Key conventions include:

- **Estimators**: All estimators should implement `fit`, `predict`, 
  and, where appropriate, `transform`. Ensure that estimators use 
  `check_X_y` and `check_array` for input validation, and provide 
  support for multi-output data where applicable.

- **Parameters and Attributes**: Use meaningful parameter names and 
  follow Scikit-learn’s conventions for optional parameters, default 
  values, and attribute naming. For example, model parameters passed 
  to `fit` should generally be public attributes (e.g., `self.coef_`).

- **Docstring Conventions**: Follow the NumPy documentation style 
  with reST syntax. Each class and method should include:

  - **Parameters**: Document each parameter, specifying type, default 
    value, and a brief description.
  - **Attributes**: Document all public attributes with types and 
    descriptions.
  - **Examples**: Provide example usage for classes and complex 
    functions, using reStructuredText `.. code-block:: python` syntax.

Testing and Code Quality
--------------------------
To ensure the reliability and maintainability of `hwm`, all code 
must pass automated tests and adhere to high standards of quality:

- **Unit Tests**: Write unit tests for all new features and bug fixes, 
  ensuring thorough coverage of the codebase. Tests should include 
  both typical and edge cases.

- **Continuous Integration**: The repository includes CI configuration 
  (e.g., GitHub Actions) to automatically run tests on each pull 
  request. Ensure all tests pass before merging.

- **Code Style**: Adhere to PEP 8 style guidelines and use `flake8` 
  and `black` for code linting and formatting.

- **Type Checking**: Use type hints where possible and run `mypy` 
  for static type checking:

  .. code-block:: bash

      mypy hwm/

Branching Strategy
--------------------
The `hwm` project follows a branching strategy to streamline 
development:

- **main**: The stable branch for production-ready code. 
  Only merge PRs into `main` after thorough review and testing.
- **dev**: A development branch where new features and fixes are 
  integrated before merging into `main`.
- **feature/** or **bugfix/** branches: Each feature or bug fix 
  should have its own branch, which is merged into `dev` once 
  complete.

Contributing Documentation
----------------------------
High-quality documentation is critical. All new functionality should 
include complete documentation:

1. **Docstrings**: Write comprehensive docstrings for every class, 
   function, and method, using the NumPy style with reST syntax.

2. **User Guide**: Update the user guide or API reference as needed. 
   Major new features should be documented with examples in the 
   `doc/` directory.

3. **Building Documentation**: The `hwm` package uses Sphinx to 
   generate documentation. To build the docs locally:

   .. code-block:: bash

       cd doc/
       make html

   This generates HTML documentation in the `_build/html` directory, 
   which can be viewed in a browser.

4. **Review and Refine**: Proofread and refine documentation for 
   clarity, ensuring that examples are correct and all references 
   are accurate.

Submitting a Pull Request
---------------------------
After completing your feature or fix and writing tests and 
documentation:

1. **Run tests**: Confirm that all tests pass and code quality checks 
   are met:

   .. code-block:: bash

       pytest tests/
       flake8 hwm/
       black hwm/

2. **Push changes**: Push your feature or bugfix branch to GitHub.

3. **Create a pull request**: Open a pull request on GitHub, linking 
   to any relevant issues and summarizing the change.

4. **Address review feedback**: Be responsive to code reviews, 
   updating your PR as needed.

By following these guidelines, you help ensure that `hwm` maintains 
a high standard of quality, consistency, and usability for all users.

Thank you for contributing to `hwm`!
