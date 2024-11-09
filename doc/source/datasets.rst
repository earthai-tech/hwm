.._datasets:

.. _datasets:

==========
Datasets
==========

This section contains descriptions of the available datasets in the
:mod:`hwm.datasets` module. These datasets are synthetic and designed for
use in financial and system modeling applications.

.. _make_financial_market_trends:

make_financial_market_trends
==============================

The `make_financial_market_trends` function generates a synthetic
dataset that simulates financial market trends, capturing dynamic
behaviors like price fluctuations, volatility, and market shifts.
It is particularly useful for supervised learning tasks such as
forecasting stock market trends, price prediction, and volatility
analysis.

Parameters
------------

.. list-table::
   :widths: 25 75
   :header-rows: 1

   * - **samples**
     - Number of data points (observations) in the dataset.
       Default is 1000.
   * - **trading_days**
     - The assumed number of trading days in a year. Default is 252.
   * - **start_date**
     - Start date for the dataset, e.g., "2023-01-01". Default is None.
   * - **end_date**
     - End date for the dataset. Must be used with `start_date`.
   * - **price_noise_level**
     - Standard deviation of noise added to the price. Default is 0.01.
   * - **volatility_level**
     - Volatility factor for simulating market fluctuations.
       Default is 0.02.
   * - **nonlinear_trend**
     - Whether to apply a nonlinear transformation to the trend.
       Default is True.
   * - **base_price**
     - Starting price for the asset. Default is 100.0.
   * - **trend_frequency**
     - Frequency of the sinusoidal trend. Default is 1/252.
   * - **market_sensitivity**
     - Sensitivity of the price to the general market trend.
       Default is 0.05.
   * - **trend_strength**
     - Strength of the price trend. Default is 0.03.
   * - **as_frame**
     - If True, returns the dataset as a `pandas.DataFrame`.
       Default is True.
   * - **return_X_y**
     - If True, returns the dataset without the target column.
       Default is False.
   * - **split_X_y**
     - If True, splits the dataset into training and test sets.
       Default is False.
   * - **target_names**
     - The name(s) of the target variables. Default is `["price_output"]`.
   * - **test_size**
     - The proportion of the dataset to include in the test set (0-1).
       Default is 0.3.
   * - **seed**
     - Random seed for reproducibility. Default is None.

Returns
---------

The function returns a structured dataset based on the parameters. The
format of the dataset depends on the `as_frame`, `return_X_y`, and
`split_X_y` arguments. Possible return formats include:

- **pandas.DataFrame**: If `as_frame=True`, a DataFrame with time-indexed
  records of simulated financial market features and target values.
- **tuple (X, y)**: If `return_X_y=True`, a tuple containing the features
  (`X`) and target (`y`).
- **dictionary**: A dictionary with the features and target.

The dataset includes features like the time variable, price trends,
market trends, daily returns, moving averages, volatility, and various
financial indicators such as the Relative Strength Index (RSI) and
Bollinger Bands.

Dataset Example
-----------------

Here’s how to generate and access the financial market trends dataset:

.. code-block:: python
    :linenos:

    from hwm.datasets import make_financial_market_trends

    # Generate a dataset with 1500 samples, using a start date
    data = make_financial_market_trends(
        samples=1500, trading_days=252,
        start_date="2023-01-01", seed=42
    )

    # Inspect the shape of the data
    print(data['data'].shape)
    # Output: (1500, 13)

    # View the first few rows of the dataset
    print(data['data'].head())

The dataset contains multiple features, including `price_trend`,
`market_trend`, and `price_response`.

Formulation
----------------

The dataset is generated based on a combination of linear and
nonlinear transformations:

- **Price Trend**: Models the general trend of the asset's price as a
  sinusoidal function with noise:

  .. math::

      \text{price\_trend} = \text{base\_price} \times \left(1 +
      \text{trend\_strength} \times \sin\left(2 \pi \times \text{trend\_frequency}
      \times \text{time}\right)\right) + \text{noise}

- **Market Trend**: Represents the market's general effect on the asset's
  price, defined by:

  .. math::

      \text{market\_trend} = \text{market\_sensitivity} \times \text{price\_trend}

- **Nonlinear Price Response**: Adds a nonlinear transformation to
  the market trend, influenced by market volatility:

  .. math::

      \text{price\_response} = \text{market\_trend} \times \left(1 +
      \tanh\left(\text{trend\_strength} \times \text{market\_trend}\right)\right)
      + \text{volatility\_noise}

- **Relative Strength Index (RSI)**: A momentum indicator that measures
  the price strength:

  .. math::

      \text{RSI} = 100 - \left(\frac{100}{1 + \frac{\text{Average Gain}}
      {\text{Average Loss}}}\right)

Methods
----------

- **manage_data**: Utility function for structuring and returning the dataset.
  Used internally to create datasets.

Examples
----------

The following examples demonstrate how to generate and utilize the
financial market trends dataset.

**Basic Example:**

In this example, we generate a simple financial market trends dataset to
understand its basic functionality.

.. code-block:: python
    :linenos:

    from hwm.datasets import make_financial_market_trends

    # Generate financial market data with default parameters
    data = make_financial_market_trends()

    # Output the first few rows of the dataset
    print(data['data'].head())
    # Output:
    #         time  price_trend  market_trend  price_response  \
    # 0  0.000000   100.000000     5.000000        5.000000   
    # 1  0.002008   100.001008     5.000050        5.000050   
    # 2  0.004016   100.002016     5.000100        5.000100   
    # 3  0.006024   100.003024     5.000150        5.000150   
    # 4  0.008032   100.004032     5.000200        5.000200   

       daily_return  moving_average  price_volatility  stability_metric  \
    0      0.000000       100.000000           0.000000          1.000000   
    1      0.010080       100.000504           0.010080          0.999899   
    2      0.010080       100.001008           0.010080          0.999800   
    3      0.010080       100.001512           0.010080          0.999700   
    4      0.010080       100.002016           0.010080          0.999600   

       relative_strength_index  exponential_moving_average  upper_band  \
    0                  50.00000                 100.000000   100.000000   
    1                  50.00000                 100.000252   100.010080   
    2                  50.00000                 100.000504   100.010080   
    3                  50.00000                 100.000756   100.010080   
    4                  50.00000                 100.001008   100.010080   

       lower_band  price_output  
    0   99.990000      5.000000  
    1   99.990000      5.000050  
    2   99.990000      5.000100  
    3   99.990000      5.000150  
    4   99.990000      5.000200  

**Complex Example:**

This example demonstrates the `make_financial_market_trends` function with
custom parameters to simulate a more realistic and nuanced financial market
scenario, including multiple outputs and sample weights.

.. code-block:: python
    :linenos:

    from hwm.datasets import make_financial_market_trends
    import numpy as np

    # Multi-output predictions over 6 time steps
    y_pred = np.array([
        [2.0, 3.0],
        [2.5, 3.5],
        [3.0, 4.0],
        [3.5, 4.5],
        [4.0, 5.0],
        [4.5, 5.5]
    ])

    # Sample weights for each prediction
    sample_weight = np.array([1, 2, 1, 2, 1, 2])

    # Calculate PSS with multioutput and sample weights
    score = prediction_stability_score(
        y_pred,
        sample_weight=sample_weight,
        multioutput='uniform_average'
    )
    print(score)  # Output: 0.75

    # Generate financial market data with custom parameters
    data = make_financial_market_trends(
        samples=2000,
        trading_days=252,
        start_date="2022-01-01",
        end_date="2026-12-31",
        price_noise_level=0.02,
        volatility_level=0.03,
        nonlinear_trend=False,
        base_price=150.0,
        trend_frequency=1/252,
        market_sensitivity=0.07,
        trend_strength=0.04,
        as_frame=True,
        return_X_y=False,
        split_X_y=False,
        target_names=["price_output"],
        test_size=0.25,
        seed=123
    )

    # Access specific features
    print(data[['price_trend', 'market_trend', 'RSI']].head())

Notes
-------

- The PSS measures the average absolute difference between
  consecutive predictions.
- A lower PSS indicates more stable predictions over time.
- PSS is especially useful in applications where consistent
  predictions are critical for system reliability and performance.

See Also
--------

- :func:`~hwm.metrics.twa_score` : Time-Weighted Accuracy for
  classification tasks.



make_system_dynamics
======================

.. _make_system_dynamics:

**make_system_dynamics**
--------------------------

The `make_system_dynamics` function generates a synthetic control
systems dataset with realistic features, modeling how a control system
responds to input signals, external disturbances, and nonlinear factors.
Designed for supervised learning tasks in control systems analysis, the
dataset includes both dynamic and performance-related features, making
it suitable for modeling system dynamics and behavior over time.

Parameters
------------

.. list-table::
   :widths: 25 75
   :header-rows: 1

   * - **samples**
     - Number of time points in the dataset, representing discrete
       observations of the control system over the specified duration.
       Default is 1000.
   * - **end_time**
     - Total duration of the simulation in seconds, defining the time
       range from 0 to `end_time` across the specified number of `samples`.
       Default is 10.
   * - **input_noise_level**
     - Standard deviation of Gaussian noise added to the input signal,
       simulating real-world input variability. Default is 0.05.
   * - **control_noise_level**
     - Standard deviation of Gaussian noise added to the control system's
       output, modeling external disturbances and control noise.
       Default is 0.02.
   * - **nonlinear_response**
     - Whether to apply a nonlinear transformation to the linear output using
       a hyperbolic tangent function (`tanh`). Set to `True` to simulate
       systems with nonlinear responses. Default is True.
   * - **input_amplitude**
     - Base amplitude of the input signal, defining its initial strength
       prior to modulation or noise addition. Default is 1.0.
   * - **input_frequency**
     - Frequency of the input signal in Hertz (Hz), determining the rate
       of oscillation in the sinusoidal input. Default is 0.5.
   * - **system_gain**
     - Gain applied to the input signal to simulate the linear response of the
       control system. Represents the system's linear amplification factor.
       Default is 0.9.
   * - **response_sensitivity**
     - Sensitivity applied in the nonlinear response calculation if
       `nonlinear_response` is `True`, controlling the strength of the
       nonlinear effect on the linear output. Default is 0.7.
   * - **as_frame**
     - If `True`, returns the dataset as a DataFrame; if `False`, returns it
       as a dictionary or another format based on additional arguments.
       Default is True.
   * - **return_X_y**
     - If `True`, returns feature data `X` and target `y` separately.
       Default is False.
   * - **split_X_y**
     - If `True`, splits data into training and test sets based on
       `test_size`. Default is False.
   * - **target_names**
     - Names of the target variable(s) to be returned in the dataset.
       Defaults to `["output"]`, representing the final output signal of the system.
   * - **test_size**
     - Proportion of the dataset to include in the test split when
       `split_X_y` is `True`. Default is 0.3.
   * - **seed**
     - Seed for random number generation to ensure reproducibility in noise
       addition and random operations. Default is None.

Returns
---------

The function returns a structured dataset based on the parameters. The
format of the dataset depends on the `as_frame`, `return_X_y`, and
`split_X_y` arguments. Possible return formats include:

- **pandas.DataFrame**: If `as_frame=True`, a DataFrame with time-indexed
  records of simulated control system features and target values.
- **tuple (X, y)**: If `return_X_y=True`, a tuple containing the features
  (`X`) and target (`y`).
- **dictionary**: A dictionary with the features and target.

The dataset includes features like the input signal, linear and nonlinear
outputs, control effort, error signals, power consumption, response rate,
stability metrics, and the final system output.

Formulation
-------------

The dataset is generated based on a combination of linear and
nonlinear transformations on the input signal. Several features capture
the control system’s behavior over time:

- **Input Signal**: The input is modeled as a sinusoidal wave with added
  Gaussian noise:

  .. math::

      \text{Input Signal} = A \cdot \sin(2 \pi f t) + \text{noise}

- **Linear Output**: Represents the system's linear response to the input
  after applying `system_gain`:

  .. math::

      \text{Linear Output} = \text{system\_gain} \cdot \text{Input Signal}

- **Nonlinear Response**: If `nonlinear_response` is `True`, applies a
  nonlinear function, controlled by `response_sensitivity`:

  .. math::

      \text{Response Output} = \tanh(\text{response\_sensitivity} \cdot \text{Linear Output})

- **Control Effort**: Estimated as the absolute value of the product of
  `system_gain` and `input_signal`, providing insight into the effort
  required to control the system.

- **Power Consumption**: Approximates the energy expenditure as a function
  of control effort:

  .. math::

      \text{Power Consumption} = \text{Control Effort}^2

- **Stability Metric**: Measures system stability by comparing the nonlinear
  response to the linear output:

  .. math::

      \text{Stability Metric} = 1 - \left| \text{Response Output} - \text{Linear Output} \right|

Methods
---------

- **manage_data**: Utility function for structuring and returning the dataset.
  Used internally to create datasets.

Examples
----------

The following examples demonstrate how to generate and utilize the
control systems dynamics dataset.

**Basic Example:**

In this example, we generate a simple control systems dataset to understand
its basic functionality.

.. code-block:: python

    from hwm.datasets import make_system_dynamics
    import pandas as pd

    # Generate a dataset with default parameters
    data = make_system_dynamics()
    print(data.head())
    # Output:
    #        time  input_signal  linear_output  response_output  control_effort  \
    # 0  0.000000       0.000000        0.000000         0.000000         0.000000   
    # 1  0.010101       0.031416        0.028275         0.026581         0.025383   
    # 2  0.020202       0.062523        0.056271         0.052927         0.050767   
    # 3  0.030303       0.093243        0.083918         0.075527         0.084237   
    # 4  0.040404       0.123544        0.111189         0.099343         0.111544   

       error_signal  power_consumption  response_rate  stability_metric     output  
    0       0.000000            0.000000        0.000000          1.000000  0.000000  
    1      -0.001694            0.000643        2.653534          0.983419  0.026581  
    2      -0.000927            0.002575        3.255982          0.947073  0.052927  
    3      -0.007391            0.007113        3.541963          0.915933  0.075527  
    4      -0.011201            0.012494        3.798764          0.905825  0.099343  

**Complex Example:**

This example demonstrates the `make_system_dynamics` function in a
multi-output control system scenario with sample weights.

.. code-block:: python

    from hwm.datasets import make_system_dynamics
    import pandas as pd
    import numpy as np

    # Generate control systems data with custom parameters
    data = make_system_dynamics(
        samples=1500,
        end_time=20,
        input_noise_level=0.1,
        control_noise_level=0.05,
        nonlinear_response=True,
        input_amplitude=2.0,
        input_frequency=1.0,
        system_gain=1.2,
        response_sensitivity=0.8,
        as_frame=True,
        return_X_y=False,
        split_X_y=False,
        target_names=["response_output"],
        test_size=0.25,
        seed=42
    )

    # Inspect the first few rows of the dataset
    print(data.head())

    # Access specific features
    print(data[['input_signal',  'power_consumption']].head())

**Explanation:**

- **Parameter Customization:**
  - Increased `samples` to 1500 and `end_time` to 20 seconds to simulate a longer duration.
  - Enhanced noise levels (`input_noise_level=0.1`, `control_noise_level=0.05`) to model more realistic variability.
  - Adjusted `input_amplitude` and `input_frequency` to change the input signal characteristics.
  - Modified `system_gain` and `response_sensitivity` to simulate different system dynamics.

- **Generated Features:**
  - **input_signal**: Enhanced amplitude and frequency with added noise.
  - **linear_output**: Scaled input signal reflecting system gain.
  - **response_output**: Nonlinear transformation applied to the linear output with added control noise.
  - **control_effort**: Increased due to higher system gain and input amplitude.
  - **power_consumption**: Reflects the squared control effort, indicating higher energy usage.
  - **stability_metric**: Measures the deviation between nonlinear response and linear output.

Notes
-------

- The `make_system_dynamics` dataset is ideal for training and testing models
  in control systems analysis, especially those focusing on system
  identification, dynamics, and response prediction in the presence of both
  linear and nonlinear behaviors.
- Proper selection of parameters like `alpha`, `system_gain`, and
  `response_sensitivity` is crucial for simulating realistic system
  dynamics.
- The dataset can be returned in various formats (DataFrame, tuple, dictionary)
  based on the user's needs, facilitating flexibility in downstream tasks.


References
------------

.. [1] Ogata, K. (2010). *Modern Control Engineering*. Prentice Hall.
.. [2] Dorf, R. C., & Bishop, R. H. (2017). *Modern Control Systems*. Pearson.

.. [3] Alexander, C. (2001). *Market Models: A Guide to Financial Data
      Analysis*. Wiley.
.. [4] Hull, J. C. (2014). *Options, Futures, and Other Derivatives*.
      Pearson.

See Also
----------
- :func:`hwm.utils.manage_data`: A utility for managing dataset structures.


