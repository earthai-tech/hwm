# -*- coding: utf-8 -*-
#   License: BSD-3-Clause
#   Author: LKouadio <etanoyau@gmail.com>

import numpy as np 
import pandas as pd 

from ..utils._core import manage_data 
from ..utils.validator import validate_dates, validate_positive_integer

__all__ = ["make_financial_market_trends", "make_system_dynamics"]

def make_financial_market_trends(
    *,
    samples=1000,
    trading_days=252,
    start_date=None,
    end_date=None,
    price_noise_level=0.01,
    volatility_level=0.02,
    nonlinear_trend=True,
    base_price=100.0,
    trend_frequency=1/252,
    market_sensitivity=0.05,
    trend_strength=0.03,
    as_frame=True,
    return_X_y=False,
    split_X_y=False,
    target_names=None,
    test_size=0.3,
    seed=None
):
    """
    Generate a synthetic dataset simulating financial market trends.

    This function creates a synthetic dataset that models dynamic behaviors
    in financial markets, capturing key market trends, price fluctuations,
    volatility, and other financial indicators over a specified period.
    It is designed for supervised learning tasks in financial modeling,
    providing a rich set of features suitable for analyzing and forecasting
    stock market trends, price prediction, volatility analysis, and risk
    assessment.

    Parameters
    ----------
    samples : int, default=1000
        Number of data points (observations) in the dataset, representing
        distinct trading days or other defined intervals within the simulated
        period.

    trading_days : int, default=252
        The assumed number of trading days in a year. If `start_date` and
        `end_date` are specified, `trading_days` is overridden and computed
        based on the actual dates provided.

    start_date : str, optional
        Start date of the simulation as a string (e.g., "2023-01-01").
        When provided, `end_date` can be specified to define the range.
        If only `start_date` is provided, the period runs from `start_date`
        to the period covering `trading_days`. If both are omitted, the
        simulation defaults to a period covering `trading_days`.

    end_date : str, optional
        End date of the simulation. Must be used in conjunction with
        `start_date`. If only `end_date` is provided, a `ValueError` is raised.

    price_noise_level : float, default=0.01
        Standard deviation of Gaussian noise added to the base price trend,
        simulating day-to-day price variability typical in financial markets.

    volatility_level : float, default=0.02
        Standard deviation of Gaussian noise added to simulate market
        volatility, affecting the `price_response` variable.

    nonlinear_trend : bool, default=True
        Whether to apply a nonlinear transformation to the market trend using
        a hyperbolic tangent function, modeling potential economic shifts
        or trends.

    base_price : float, default=100.0
        The starting base price of the stock or asset being simulated,
        representing the average market price prior to the application of
        trends and noise.

    trend_frequency : float, default=1/252
        The frequency of the sinusoidal trend applied to the base price.
        The default models a weekly frequency over a typical trading year.

    market_sensitivity : float, default=0.05
        Controls the linear component of the market trend, representing the
        asset's sensitivity to general market conditions.

    trend_strength : float, default=0.03
        Strength of the sinusoidal component applied to the price trend,
        affecting the amplitude of price oscillations.

    as_frame : bool, default=True
        If True, returns the dataset as a `pandas.DataFrame`. If False,
        the dataset is returned in the format specified by additional arguments.

    return_X_y : bool, default=False
        If True, returns a tuple `(X, y)` where `X` is the dataset without
        the target column, and `y` is the target column.

    split_X_y : bool, default=False
        If True, splits the dataset into training and test sets based on
        `test_size`.

    target_names : str or list of str, optional
        Names of the target variable(s) in the dataset. Defaults to
        `["price_output"]`, representing the final observed stock price
        after all transformations.

    test_size : float, default=0.3
        Proportion of the dataset to be included in the test split if
        `split_X_y` is True. Expressed as a decimal (e.g., 0.3 for 30%).

    seed : int, optional
        Seed for random number generation, ensuring reproducibility of random
        components like noise additions.

    Returns
    -------
    dataset : dict or tuple or pandas.DataFrame
        The structured dataset containing time-indexed records of simulated
        stock market features and target price values. Format depends on
        the arguments `as_frame`, `return_X_y`, and `split_X_y`.

    Notes
    -----
    The dataset models stock price behaviors over time through a combination
    of linear and nonlinear transformations and financial indicators:

    **Price Trend**: Models the general trend of the price as a sinusoidal
    function with noise:

    .. math::

        \\text{price\\_trend} = \\text{base\\_price} \\times
        \\left(1 + \\text{trend\\_strength} \\times
        \\sin(2 \\pi \\times \\text{trend\\_frequency} \\times \\text{time})\\right)
        + \\text{noise}

    **Market Trend**: Represents a linear transformation based on market
    sensitivity:

    .. math::

        \\text{market\\_trend} = \\text{market\\_sensitivity} \\times \\text{price\\_trend}

    **Nonlinear Price Response**: Adds a nonlinear transformation (tanh) to
    model significant market shifts:

    .. math::

        \\text{price\\_response} = \\text{market\\_trend} \\times
        \\left(1 + \\tanh(\\text{trend\\_strength} \\times \\text{market\\_trend})\\right)
        + \\text{volatility\\_noise}

    **Relative Strength Index (RSI)**: A momentum indicator measuring price
    strength:

    .. math::

        \\text{RSI} = 100 - \\left( \\frac{100}{1 + \\frac{\\text{Average Gain}}
        {\\text{Average Loss}}} \\right)

    Methods
    -------
    manage_data :
        Function for structuring and returning datasets.

    Examples
    --------
    >>> from hwm.datasets import make_financial_market_trends
    >>> data = make_financial_market_trends(samples=1500, trading_days=252,
    ...                                     start_date="2023-01-01", seed=42)
    >>> data['data'].shape
    (1500, 11)

    See Also
    --------
    manage_data : Function for structuring and returning datasets.

    References
    ----------
    .. [1] Alexander, C. (2001). *Market Models: A Guide to Financial Data
           Analysis*. Wiley.
    .. [2] Hull, J. C. (2014). *Options, Futures, and Other Derivatives*.
           Pearson.
    """
    # Set random seed for reproducibility
    np.random.seed(seed)
    
    # Validate trading time range
    trading_days = _validate_trading_days(trading_days, start_date, end_date)
    samples = validate_positive_integer(samples, "samples")
    
    # Generate time variable
    time = _generate_time_variable(trading_days, samples, start_date, end_date)

    # Generate primary price trends and responses
    price_trend = _generate_price_trend(
        time, base_price, trend_strength, trend_frequency, price_noise_level)
    market_trend = _generate_market_trend(price_trend, market_sensitivity)
    price_response = _generate_price_response(
        market_trend, trend_strength, volatility_level, nonlinear_trend)

    # Generate additional finance-specific features
    daily_return = _calculate_daily_return(price_response)
    moving_average = _calculate_moving_average(price_response)
    price_volatility = _calculate_volatility(price_response)
    stability_metric = 1 - np.abs(price_response - market_trend) / base_price

    # Additional financial indicators
    relative_strength_index = _calculate_rsi(price_response)
    exponential_moving_average = _calculate_ema(price_response)
    upper_band, lower_band = _calculate_bollinger_bands(moving_average, price_volatility)

    # Define target variable
    target_output = price_response

    # Create DataFrame for dataset with all features
    dataset = pd.DataFrame({
        "time": time,
        "price_trend": price_trend,
        "market_trend": market_trend,
        "price_response": price_response,
        "daily_return": daily_return,
        "moving_average": moving_average,
        "price_volatility": price_volatility,
        "stability_metric": stability_metric,
        "relative_strength_index": relative_strength_index,
        "exponential_moving_average": exponential_moving_average,
        "upper_band": upper_band,
        "lower_band": lower_band,
        "price_output": target_output
    })

    target_names = target_names or ["price_output"]

    # Return structured dataset using `manage_data`
    return manage_data(
        data=dataset, 
        as_frame=as_frame, 
        return_X_y=return_X_y, 
        split_X_y=split_X_y, 
        target_names=target_names, 
        test_size=test_size,
    )

def make_system_dynamics(
    *, samples=1000, 
    end_time=10, 
    input_noise_level=0.05, 
    control_noise_level=0.02,
    nonlinear_response=True,
    input_amplitude=1.0,  
    input_frequency=0.5,  
    system_gain=0.9,  
    response_sensitivity=0.7,  
    as_frame=True, 
    return_X_y= False, 
    split_X_y=False, 
    target_names=None,  
    test_size=.3, 
    seed=None
):
    """
    Generate a synthetic control systems dataset with realistic features, 
    modeling how a control system responds to input signals, external  
    disturbances, and nonlinear factors. Designed for supervised learning 
    tasks in control systems analysis, the dataset includes both dynamic and 
    performance-related features, making it suitable for modeling system 
    dynamics and behavior over time.
    
    Parameters
    ----------
    samples : int, default=1000
        Number of time points in the dataset, representing discrete observations of 
        the control system over the specified duration.
    
    end_time : float, default=10
        Total duration of the simulation in seconds, defining the time range from 
        0 to `end_time` across the specified number of `samples`.
    
    input_noise_level : float, default=0.05
        Standard deviation of Gaussian noise added to the input signal, simulating 
        real-world input variability.
    
    control_noise_level : float, default=0.02
        Standard deviation of Gaussian noise added to the control system's output, 
        modeling external disturbances and control noise.
    
    nonlinear_response : bool, default=True
        Whether to apply a nonlinear transformation to the linear output using a 
        hyperbolic tangent function (`tanh`). Set to `True` to simulate systems 
        with nonlinear responses.
    
    input_amplitude : float, default=1.0
        Base amplitude of the input signal, defining its initial strength prior 
        to modulation or noise addition.
    
    input_frequency : float, default=0.5
        Frequency of the input signal in Hertz (Hz), determining the rate of 
        oscillation in the sinusoidal input.
    
    system_gain : float, default=0.9
        Gain applied to the input signal to simulate the linear response of the 
        control system. Represents the system's linear amplification factor.
    
    response_sensitivity : float, default=0.7
        Sensitivity applied in the nonlinear response calculation if 
        `nonlinear_response` is `True`, controlling the strength of the nonlinear 
        effect on the linear output.
    
    as_frame : bool, default=True
        If `True`, returns the dataset as a DataFrame; if `False`, returns it as 
        a dictionary or another format based on additional arguments.
    
    return_X_y : bool, default=False
        If `True`, returns feature data `X` and target `y` separately.
    
    split_X_y : bool, default=False
        If `True`, splits data into training and test sets based on `test_size`.
    
    target_names : str or list of str, optional
        Names of the target variable(s) to be returned in the dataset. Defaults to 
        `["output"]`, representing the final output signal of the system.
    
    test_size : float, default=0.3
        Proportion of the dataset to include in the test split when `split_X_y` 
        is `True`.
    
    seed : int, optional
        Seed for random number generation to ensure reproducibility in noise 
        addition and random operations.
    
    Returns
    -------
    dataset : pd.DataFrame or dict
        The control systems dataset with columns representing features and target 
        variables, structured based on specified return options.
    
    Concept
    -------
    This dataset models control system dynamics through both linear and nonlinear 
    transformations on the input signal. Several features capture the control 
    system’s behavior over time:
    
    - **Input Signal**: The input is modeled as a sinusoidal wave with added 
      Gaussian noise:
    
      .. math::
          \\text{Input Signal} = A \\cdot \\sin(2 \\pi f t) + \\text{noise}
    
    - **Linear Output**: Represents the system's linear response to the input 
      after applying `system_gain`:
    
      .. math::
          \\text{Linear Output} = \\text{system\_gain} \\cdot \\text{Input Signal}
    
    - **Nonlinear Response**: If `nonlinear_response` is `True`, applies a 
      nonlinear function, controlled by `response_sensitivity`:
    
      .. math::
          \\text{Response Output} = \\tanh(\\text{response\_sensitivity} \\cdot \\text{Linear Output})
    
    - **Control Effort**: Estimated as the absolute value of the product of 
      `system_gain` and `input_signal`, providing insight into the effort 
      required to control the system.
    
    - **Power Consumption**: Approximates the energy expenditure as a function 
      of control effort:
    
      .. math::
          \\text{Power Consumption} = \\text{Control Effort}^2
    
    - **Stability Metric**: Measures system stability by comparing the nonlinear 
      response to the linear output:
    
      .. math::
          \\text{Stability Metric} = 1 - | \\text{Response Output} - \\text{Linear Output} |
    
    Notes
    -----
    This dataset is suitable for training and testing models for control systems 
    analysis, especially those focusing on system identification, dynamics, and 
    response prediction in the presence of both linear and nonlinear behaviors.
    
    Examples
    --------
    >>> from hwm.datasets import make_control_system_dynamics
    >>> data = make_control_system_dynamics(samples=1500, end_time=20)
    >>> data.head()
    
    See Also
    --------
    manage_data : Organizes and returns structured data based on specified options.
    
    References
    ----------
    .. [1] Ogata, K. (2010). Modern Control Engineering. Prentice Hall.
    .. [2] Dorf, R. C., & Bishop, R. H. (2017). Modern Control Systems. Pearson.
    """

    # Set random seed for reproducibility
    np.random.seed(seed)
    
    # Validate parameters
    end_time = validate_positive_integer(end_time, "end_time")
    samples = validate_positive_integer(samples, "samples")
    
    # Generate time variable from 0 to `end_time`
    time = np.linspace(0, end_time, samples)  
    
    # Generate system input signal as a sinusoidal function with input noise
    input_signal = input_amplitude * np.sin(2 * np.pi * input_frequency * time)
    input_signal += np.random.normal(0, input_noise_level, samples)

    # Apply linear transformation to simulate system gain on input signal
    linear_output = system_gain * input_signal  

    # Apply nonlinear transformation if specified
    if nonlinear_response:
        response_output = np.tanh(response_sensitivity * linear_output)  # Nonlinear response
    else:
        response_output = linear_output  # Purely linear response

    # Add control noise to simulate external disturbances
    control_noise = np.random.normal(0, control_noise_level, samples)
    response_output += control_noise

    # Generate additional system-specific features
    control_effort = system_gain * np.abs(input_signal)  # Control effort metric
    error_signal = input_signal - response_output  # Difference between input and output (error)
    power_consumption = control_effort ** 2  # Power consumption as function of control effort
    response_rate = np.gradient(response_output, time)  # Rate of change of the response
    stability_metric = 1 - np.abs(response_output - linear_output)  # Measure system stability

    # Define the target variable representing the final system output
    target_output = response_output

    # Create a DataFrame for the dataset with all features
    dataset = pd.DataFrame({
        "time": time,
        "input_signal": input_signal,
        "linear_output": linear_output,
        "response_output": response_output,
        "control_effort": control_effort,
        "error_signal": error_signal,
        "power_consumption": power_consumption,
        "response_rate": response_rate,
        "stability_metric": stability_metric,
        "output": target_output
    })

    target_names = target_names or ["output"]

    # Return the structured dataset using `manage_data`
    return manage_data(
        data=dataset, 
        as_frame=as_frame, 
        return_X_y= return_X_y, 
        split_X_y=split_X_y, 
        target_names=target_names, 
        test_size=test_size, 
    )


def _validate_trading_days(trading_days, start_date, end_date):
    # handle date comparisons
    from datetime import datetime
    # Checks if a start date is provided
    
    if start_date is not None:
        # If end date is not provided, defaults to the current date
        if end_date is None:
            end_date = datetime.now()
        # Validates that both start and end dates are proper date formats
        start_date, end_date = validate_dates(start_date, end_date)
        # Computes trading days as the difference between the start and end dates
        trading_days = (end_date - start_date).days
        
    # If neither trading days nor dates are provided, raises an error
    elif trading_days is None:
        raise ValueError("Provide `trading_days`, `start_date`, or `end_date`.")
    # Returns the computed or validated trading days
    return trading_days

def _generate_time_variable(trading_days, samples, start_date, end_date):
    # If a start date is given, generates dates using business day frequency
    if start_date is not None:
        return pd.date_range(start=start_date, periods=samples, freq="B")
    
    # Otherwise, generates a continuous time variable up to trading days
    return np.linspace(0, trading_days, samples)

def _generate_price_trend(
        time, base_price, trend_strength, trend_frequency, noise_level):
    # Models the price trend as a sinusoidal function, adding specified noise
    price_trend = base_price * (1 + trend_strength * np.sin(
        2 * np.pi * trend_frequency * time))
    # Adds Gaussian noise to simulate realistic price fluctuations
    return price_trend + np.random.normal(0, noise_level, len(time))

def _generate_market_trend(price_trend, sensitivity):
    # Generates the market trend by applying a linear
    # sensitivity factor to the price trend
    return sensitivity * price_trend

def _generate_price_response(
        market_trend, trend_strength, volatility_level, nonlinear_trend):
    # Applies a nonlinear transformation if specified, simulating economic shifts
    if nonlinear_trend:
        price_response = market_trend * (
            1 + np.tanh(trend_strength * market_trend))
        
    # If no nonlinear trend is specified, keeps the response linear
    else:
        price_response = market_trend
    # Adds volatility noise to the price response
    return price_response + np.random.normal(
        0, volatility_level, len(market_trend))

def _calculate_daily_return(price_response):
    # Calculates the daily return rate as the gradient of the price response
    return np.gradient(price_response) / price_response

def _calculate_moving_average(
        price_response, window=20):
    # Computes the moving average over a specified window,
    # filling missing values by backfilling
    return pd.Series(price_response).rolling(
        window=window).mean().fillna(method='bfill')

def _calculate_volatility(price_response, window=20):
    # Computes rolling standard deviation 
    # to estimate volatility over a specified window
    return pd.Series(price_response).rolling(
        window=window).std().fillna(method='bfill')

def _calculate_rsi(price_response, period=14):
    # Computes the Relative Strength Index (RSI), 
    # a momentum indicator for price changes
    delta = pd.Series(price_response).diff()
    # Calculates average gains and losses over the specified period
    gain = delta.apply(lambda x: max(x, 0)).rolling(period).mean()
    loss = -delta.apply(lambda x: min(x, 0)).rolling(period).mean()
    # Computes the RSI using the ratio of average gains to losses
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def _calculate_ema(price_response, span=20):
    # Calculates the Exponential Moving Average (EMA) with a specified span
    return pd.Series(price_response).ewm(span=span, adjust=False).mean()

def _calculate_bollinger_bands(moving_average, volatility, n_std=2):
    # Calculates Bollinger Bands by adding/subtracting 
    # n standard deviations from the moving average
    upper_band = moving_average + n_std * volatility
    lower_band = moving_average - n_std * volatility
    # Returns both upper and lower bands
    return upper_band, lower_band
