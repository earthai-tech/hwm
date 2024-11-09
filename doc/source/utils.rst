
.. _utils:

=======
Utils
=======

This section provides documentation for utility functions in the
:mod:`hwm.utils` module. These functions offer essential support for
data manipulation, preprocessing, and other common tasks required
during model development and evaluation.

.. _activator:

activator
===========


**activator**
---------------

.. function:: activator(z, activation='sigmoid', alpha=1.0, clipping_threshold=250)

Apply the specified activation function to the input array `z` [1]_.

Parameters
------------
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
---------
activation_output : array-like
    The output array after applying the activation function.

Notes
-------
The available activation functions are defined as follows:

- **Sigmoid**:
  :math:`\sigma(z) = \frac{1}{1 + \exp(-z)}`

- **ReLU**:
  :math:`\text{ReLU}(z) = \max(0, z)`

- **Leaky ReLU**:
  :math:`\text{Leaky ReLU}(z) = \max(0.01z, z)`

- **Identity**:
  :math:`\text{Identity}(z) = z`

- **ELU**:
  :math:`\text{ELU}(z) = \begin{cases}
              z & \text{if } z > 0 \\
              \alpha (\exp(z) - 1) & \text{if } z \leq 0
            \end{cases}`

- **Tanh**:
  :math:`\tanh(z) = \frac{\exp(z) - \exp(-z)}{\exp(z) + \exp(-z)}`

- **Softmax**:
  :math:`\text{Softmax}(z)_i = \frac{\exp(z_i)}{\sum_{j} \exp(z_j)}`

Examples
----------
The following examples demonstrate how to use the `activator` function
with different activation functions and configurations.

**Basic Example:**

Applying ReLU activation to a simple array.

.. code-block:: python
    :linenos:

    import numpy as np
    from hwm.utils import activator

    # Input array
    z = np.array([1.0, 2.0, -1.0, -2.0])

    # Apply ReLU activation
    output = activator(z, activation='relu')
    print(output)
    # Output: [1. 2. 0. 0.]

**Advanced Example:**

Using softmax activation on a multi-class array.

.. code-block:: python
    :linenos:

    import numpy as np
    from hwm.utils import activator

    # Input array for softmax
    z = np.array([2.0, 1.0, 0.1])

    # Apply softmax activation
    output = activator(z, activation='softmax')
    print(output)
    # Output: [0.65900114 0.24243297 0.09856589]

**Custom Callable Example:**

Using a custom activation function.

.. code-block:: python
    :linenos:

    import numpy as np
    from hwm.utils import activator

    # Define a custom activation function
    def custom_activation(x):
        return np.sqrt(np.abs(x)) * np.sign(x)

    # Input array
    z = np.array([4, -9, 16, -25])

    # Apply custom activation
    output = activator(z, activation=custom_activation)
    print(output)
    # Output: [ 2. -3.  4. -5.]
    
.. _resample_data:

resample_data
===============

.. function:: resample_data(*data, samples=1, replace=False, random_state=None, shuffle=True)
    
Resample multiple data structures (arrays, sparse matrices, Series, 
DataFrames) based on specified sample size or ratio [4]_.

Parameters
------------

.. list-table::
   :widths: 25 75
   :header-rows: 1

   * - **data**
     - Variable number of array-like, sparse matrix, pandas Series, or 
       DataFrame objects to be resampled.
   * - **samples**
     - Specifies the number of items to sample from each data structure.
       - If an integer greater than 1, it is treated as the exact number 
         of items to sample.
       - If a float between 0 and 1, it is treated as a ratio of the 
         total number of rows to sample.
       - If a string containing a percentage (e.g., "50%"), it calculates 
         the sample size as a percentage of the total data length.
       Default is 1, meaning no resampling is performed unless a different 
       value is specified.
   * - **replace**
     - Determines if sampling with replacement is allowed, enabling the 
       same row to be sampled multiple times. Default is False.
   * - **random_state**
     - Sets the seed for the random number generator to ensure 
       reproducibility. If specified, repeated calls with the same 
       parameters will yield identical results. Default is None.
   * - **shuffle**
     - If True, shuffles the data before sampling. Otherwise, rows are 
       selected sequentially without shuffling. Default is True.

Returns
---------

List[Any]
    A list of resampled data structures, each in the original format 
    (e.g., numpy array, sparse matrix, pandas DataFrame) and with the 
    specified sample size.

Methods
---------

- **_determine_sample_size**: Calculates the sample size based on the 
  `samples` parameter.
- **_perform_sampling**: Conducts the sampling process based on the 
  calculated sample size, `replace`, and `shuffle` parameters.

Notes
-------

- If `samples` is given as a percentage string (e.g., "25%"), the 
  actual number of rows to sample, :math:`n`, is calculated as:
  
  .. math::
      n = \left(\frac{\text{percentage}}{100}\right) \times N

  where :math:`N` is the total number of rows in the data structure.

- Resampling supports both dense and sparse matrices. If the input 
  contains sparse matrices stored within numpy objects, the function 
  extracts and samples them directly.

Examples
----------

The following examples demonstrate how to use the `resample_data` function
to resample different data structures with various configurations.

**Basic Example:**

Resampling a NumPy array by selecting 10 items with replacement.

.. code-block:: python
    :linenos:

    from hwm.utils import resample_data
    import numpy as np

    # Original data array
    data = np.arange(100).reshape(20, 5)

    # Resample 10 items with replacement
    resampled_data = resample_data(data, samples=10, replace=True)
    print(resampled_data[0].shape)
    # Output: (10, 5)

**Resampling by Ratio:**

Resampling 50% of the rows from a NumPy array without replacement.

.. code-block:: python
    :linenos:

    from hwm.utils import resample_data
    import numpy as np

    # Original data array
    data = np.arange(100).reshape(20, 5)

    # Resample 50% of the rows
    resampled_data = resample_data(data, samples=0.5, random_state=42)
    print(resampled_data[0].shape)
    # Output: (10, 5)

**Resampling with Percentage:**

Resampling 25% of the rows from a NumPy array using a percentage string.

.. code-block:: python
    :linenos:

    from hwm.utils import resample_data
    import numpy as np

    # Original data array
    data = np.arange(100).reshape(20, 5)

    # Resample 25% of the rows
    resampled_data = resample_data(data, samples="25%", random_state=42)
    print(resampled_data[0].shape)
    # Output: (5, 5)

**Multiple Data Structures:**

Resampling multiple data structures simultaneously.

.. code-block:: python
    :linenos:

    from hwm.utils import resample_data
    import numpy as np
    import pandas as pd
    import scipy.sparse as sp

    # Original data structures
    array = np.arange(100).reshape(20, 5)
    dataframe = pd.DataFrame(array, columns=['A', 'B', 'C', 'D', 'E'])
    sparse_matrix = sp.csr_matrix(array)

    # Resample 10 items from each data structure
    resampled_array, resampled_df, resampled_sparse = resample_data(
        array, dataframe, sparse_matrix, samples=10, replace=True, random_state=42
    )

    print(resampled_array.shape)
    # Output: (10, 5)
    print(resampled_df.shape)
    # Output: (10, 5)
    print(resampled_sparse.shape)
    # Output: (10, 5)

See Also
----------

:func:`numpy.random.choice` : Selects random samples from an array.
:meth:`pandas.DataFrame.sample` : Randomly samples rows from a DataFrame.


add_noises_to
===============
.. _add_noises_to:

**add_noises_to**
-------------------
.. function:: add_noises_to(data, noise=0.1, seed=None, gaussian_noise=False, cat_missing_value=pd.NA)

Adds NaN or specified missing values to a pandas DataFrame [4]_.

Parameters
------------

.. list-table::
   :widths: 25 75
   :header-rows: 1

   * - **data**
     - The DataFrame to which NaN values or specified missing values will be added.
   * - **noise**
     - The percentage of values to be replaced with NaN or the specified missing
       value in each column. This must be a number between 0 and 1.
       Default is 0.1 (10%).
   * - **seed**
     - Seed for random number generator to ensure reproducibility.
       If `seed` is an int, array-like, or BitGenerator, it will be used to seed
       the random number generator. If `seed` is a np.random.RandomState or
       np.random.Generator, it will be used as given.
   * - **gaussian_noise**
     - If `True`, adds Gaussian noise to the data. Otherwise, replaces
       values with NaN or the specified missing value.
       Default is False.
   * - **cat_missing_value**
     - The value to use for missing data in categorical columns.
       By default, `pd.NA` is used.

Returns
---------

pandas.DataFrame
    A DataFrame with NaN or specified missing values added.

Notes
-------

The function modifies the DataFrame by either adding Gaussian noise
to numerical columns or replacing a percentage of values in each
column with NaN or a specified missing value.

The Gaussian noise is added according to the formula:

.. math::
    \text{new\_value} = \text{original\_value} + \mathcal{N}(0, \text{noise})

where :math:`\mathcal{N}(0, \text{noise})` represents a normal
distribution with mean 0 and standard deviation equal to `noise`.

Examples
----------

The following examples demonstrate how to use the `add_noises_to` function
to add missing values or Gaussian noise to a DataFrame.

**Adding Missing Values:**

.. code-block:: python
    :linenos:

    from hwm.utils import add_noises_to
    import pandas as pd

    # Original DataFrame
    df = pd.DataFrame({'A': [1, 2, 3], 'B': ['x', 'y', 'z']})

    # Add 20% missing values
    new_df = add_noises_to(df, noise=0.2)
    print(new_df)
    # Output:
    #      A     B
    # 0  1.0  <NA>
    # 1  NaN     y
    # 2  3.0  <NA>

**Adding Gaussian Noise:**

.. code-block:: python
    :linenos:

    from hwm.utils import add_noises_to
    import pandas as pd

    # Original DataFrame
    df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})

    # Add 10% Gaussian noise
    new_df = add_noises_to(df, noise=0.1, gaussian_noise=True)
    print(new_df)
    # Output:
    #           A         B
    # 0  1.063292  3.986400
    # 1  2.103962  4.984292
    # 2  2.856601  6.017380

See Also
----------

:class:`pandas.DataFrame` : Two-dimensional, size-mutable, potentially
    heterogeneous tabular data.
:func:`numpy.random.normal` : Draw random samples from a normal
    (Gaussian) distribution.

.. _gen_X_y_batches:

gen_X_y_batches
=================


.. function:: gen_X_y_batches(X, y, *, batch_size="auto", n_samples=None, min_batch_size=0, shuffle=True, random_state=None, return_batches=False, default_size=200)

Generate batches of data (`X`, `y`) for machine learning tasks such as
training or evaluation [2]_. This function slices the dataset into smaller
batches, optionally shuffles the data, and returns them as a list of
tuples or just the data batches [6]_.

Parameters
------------

.. list-table::
   :widths: 25 75
   :header-rows: 1

   * - **X**
     - The input data matrix, where each row is a sample and each column
       represents a feature. Must be an ndarray of shape (n_samples, n_features).
   * - **y**
     - The target variable(s) corresponding to `X`. Can be a vector or
       matrix depending on the problem (single or multi-output).
       Must be an ndarray of shape (n_samples,) or (n_samples, n_targets).
   * - **batch_size**
     - The number of samples per batch. If set to `"auto"`, it uses the
       minimum between `default_size` and the number of samples, `n_samples`.
       Default is `"auto"`.
   * - **n_samples**
     - The total number of samples to consider. If `None`, the function
       defaults to using the number of samples in `X`.
       Default is `None`.
   * - **min_batch_size**
     - The minimum size for each batch. This parameter ensures that the
       final batch contains at least `min_batch_size` samples. If the
       last batch is smaller than `min_batch_size`, it will be excluded
       from the result.
       Default is 0.
   * - **shuffle**
     - If `True`, the data is shuffled before batching. This helps avoid
       bias when splitting data for training and validation.
       Default is `True`.
   * - **random_state**
     - The seed used by the random number generator for reproducibility.
       If `None`, the random number generator uses the system time or
       entropy source.
       Default is `None`.
   * - **return_batches**
     - If `True`, the function returns both the data batches and the slice
       objects used to index into `X` and `y`. If `False`, only the
       data batches are returned.
       Default is `False`.
   * - **default_size**
     - The default batch size used when `batch_size="auto"` is selected.
       Default is 200.

Returns
---------

list of tuples
    A list of tuples where each tuple contains a batch of `X` and its
    corresponding batch of `y`.

list of slice objects, optional
    If `return_batches=True`, this list of `slice` objects is returned,
    each representing the slice of `X` and `y` used for a specific batch.

Notes
-------

- This function ensures that no empty batches are returned. If a batch
  contains zero samples (either from improper slicing or due to
  `min_batch_size`), it will be excluded.
- The function performs shuffling using scikit-learn's `shuffle` function,
  which is more stable and reduces memory usage by shuffling indices
  rather than the whole dataset.
- The function utilizes the `gen_batches` utility to divide the data into
  batches.

Examples
----------

The following examples demonstrate how to use the `gen_X_y_batches` function
to generate data batches for machine learning tasks.

**Basic Example:**

Generating batches of size 500 from random data.

.. code-block:: python
    :linenos:

    from hwm.utils import gen_X_y_batches
    import numpy as np

    # Generate random input data and binary targets
    X = np.random.rand(2000, 5)
    y = np.random.randint(0, 2, size=(2000,))

    # Create batches of size 500 with shuffling
    batches = gen_X_y_batches(X, y, batch_size=500, shuffle=True)
    print(len(batches))
    # Output: 4

**Returning Batch Slices:**

Generating batches and obtaining slice objects for indexing.

.. code-block:: python
    :linenos:

    from hwm.utils import gen_X_y_batches
    import numpy as np

    # Generate random input data and binary targets
    X = np.random.rand(2000, 5)
    y = np.random.randint(0, 2, size=(2000,))

    # Create batches of size 500 and return batch slices
    batches, slices = gen_X_y_batches(
        X, y, batch_size=500, shuffle=True, return_batches=True
    )
    print(len(batches))
    # Output: 4
    print(len(slices))
    # Output: 4

**Handling Minimum Batch Size:**

Ensuring that the final batch meets the minimum batch size requirement.

.. code-block:: python
    :linenos:

    from hwm.utils import gen_X_y_batches
    import numpy as np

    # Generate random input data and binary targets
    X = np.random.rand(1025, 5)
    y = np.random.randint(0, 2, size=(1025,))

    # Create batches with batch_size=500 and min_batch_size=25
    batches = gen_X_y_batches(
        X, y, batch_size=500, min_batch_size=25, shuffle=True
    )
    print(len(batches))
    # Output: 2
    for batch in batches:
        print(batch[0].shape, batch[1].shape)
    # Output:
    # (500, 5) (500,)
    # (525, 5) (525,)

See Also
----------

:func:`gen_batches` : A utility function that generates slices of data.
:func:`shuffle` : A utility to shuffle data while keeping the data and labels in sync.

.. _ensure_non_empty_batch:

ensure_non_empty_batch
=========================


.. function:: ensure_non_empty_batch(X, y, *, batch_slice, max_attempts=10, random_state=None, error="raise")

Shuffle the dataset (`X`, `y`) until the specified `batch_slice` yields
a non-empty batch. This function ensures that the batch extracted using
`batch_slice` contains at least one sample by repeatedly shuffling the
data and reapplying the slice [5]_.

Parameters
------------

.. list-table::
   :widths: 25 75
   :header-rows: 1

   * - **X**
     - The input data matrix, where each row corresponds to a sample and
       each column corresponds to a feature. Must be an ndarray of shape
       (n_samples, n_features).
   * - **y**
     - The target variable(s) corresponding to `X`. It can be a one-dimensional
       array for single-output tasks or a two-dimensional array for multi-output
       tasks. Must be an ndarray of shape (n_samples,) or (n_samples, n_targets).
   * - **batch_slice**
     - A slice object representing the indices for the batch. For example,
       `slice(0, 512)` would extract the first 512 samples from `X` and `y`.
   * - **max_attempts**
     - The maximum number of attempts to shuffle the data to obtain a non-empty
       batch. If the batch remains empty after the specified number of attempts,
       a `ValueError` is raised.
       Default is 10.
   * - **random_state**
     - Controls the randomness of the shuffling. Pass an integer for reproducible
       results across multiple function calls. If `None`, the random number
       generator is the RandomState instance used by `np.random`.
       Default is `None`.
   * - **error**
     - Handle error status when an empty batch is still present after
       `max_attempts`. Expected values are `"raise"`, `"warn"`, or `"ignore"`.
       If `"warn"`, the error is converted into a warning message.
       Any other value will ignore the error message.
       Default is `"raise"`.

Returns
---------

ndarray
    The batch of input data extracted using `batch_slice`. Ensures that
    `X_batch` is not empty.

ndarray
    The batch of target data corresponding to `X_batch`, extracted using
    `batch_slice`. Ensures that `y_batch` is not empty.

Raises
--------

ValueError
    If a non-empty batch cannot be obtained after `max_attempts` shuffles.

Examples
----------

The following examples demonstrate how to use the `ensure_non_empty_batch`
function to guarantee that a batch contains data.

**Basic Example:**

Ensuring a non-empty batch from random data.

.. code-block:: python
    :linenos:

    from hwm.utils import ensure_non_empty_batch
    import numpy as np

    # Generate random input data and binary targets
    X = np.random.rand(2000, 5)
    y = np.random.randint(0, 2, size=(2000,))
    batch_slice = slice(0, 512)

    # Ensure the batch is non-empty
    X_batch, y_batch = ensure_non_empty_batch(
        X, y, batch_slice=batch_slice
    )
    print(X_batch.shape)
    # Output: (512, 5)
    print(y_batch.shape)
    # Output: (512,)

**Handling Empty Batches:**

Attempting to extract a batch from empty data, which raises a ValueError.

.. code-block:: python
    :linenos:

    from hwm.utils import ensure_non_empty_batch
    import numpy as np

    # Empty input data
    X_empty = np.empty((0, 5))
    y_empty = np.empty((0,))
    batch_slice = slice(0, 512)

    # Attempt to ensure a non-empty batch
    try:
        X_batch, y_batch = ensure_non_empty_batch(
            X_empty, y_empty, batch_slice=batch_slice
        )
    except ValueError as e:
        print(e)
        # Output: Unable to obtain a non-empty batch after 10 attempts.

**Using with Different Error Handling:**

Suppressing the error and receiving the original data when a non-empty
batch cannot be obtained.

.. code-block:: python
    :linenos:

    from hwm.utils import ensure_non_empty_batch
    import numpy as np

    # Empty input data
    X_empty = np.empty((0, 5))
    y_empty = np.empty((0,))
    batch_slice = slice(0, 512)

    # Attempt to ensure a non-empty batch with warning
    X_batch, y_batch = ensure_non_empty_batch(
        X_empty, y_empty, batch_slice=batch_slice, error="warn"
    )
    print(X_batch.shape, y_batch.shape)
    # Output: (0, 5) (0,)

See Also
----------

:func:`gen_batches` : Generate slice objects to divide data into batches.
:func:`shuffle` : Shuffle arrays or sparse matrices in a consistent way.



References
------------
.. [1] Goodfellow, I., Bengio, Y., & Courville, A. (2016). *Deep Learning*.
       MIT Press. http://www.deeplearningbook.org

.. [2] Pedregosa, F., Varoquaux, G., Gramfort, A., Michel, V., Thirion,
       B., Grisel, O., ... & Duchesnay, E. (2011). Scikit-learn: Machine
       learning in Python. *Journal of Machine Learning Research*, 12,
       2825-2830.
       
.. [3] NumPy Developers. (2023). NumPy Documentation.
       https://numpy.org/doc/

.. [4] Fisher, R.A., "The Use of Multiple Measurements in Taxonomic 
       Problems", *Annals of Eugenics*, 1936.

.. [5] Harris, C. R., Millman, K. J., van der Walt, S. J., et al.
       (2020). Array programming with NumPy. *Nature*, 585(7825),
       357-362.

.. [6] Scikit-learn. "sklearn.utils.shuffle". Available at
       https://scikit-learn.org/stable/modules/generated/sklearn.utils.shuffle.html



