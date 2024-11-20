# -*- coding: utf-8 -*-
# License: BSD-3-Clause
# Author: LKouadio <etanoyau@gmail.com>

from abc import abstractmethod
from numbers import Integral, Real
import warnings
from typing import Optional, Union, Tuple

import numpy as np
import pandas as pd 
from sklearn.base import BaseEstimator
from sklearn.model_selection import train_test_split
from sklearn.utils._param_validation import HasMethods, StrOptions

from ..api.property import LearnerMeta 
from ..utils.validator import check_is_fitted

class BaseHammersteinWiener(BaseEstimator, metaclass=LearnerMeta):
    """
    Base class for Hammerstein-Wiener models.
    
    The Hammerstein-Wiener (HW) model is a type of block-structured
    nonlinear model that consists of three main components:
    a nonlinear input block, a linear dynamic block, and a nonlinear
    output block. This structure allows the HW model to capture complex
    nonlinear relationships in data while maintaining interpretability
    and computational efficiency.

    The `BaseHammersteinWiener` class serves as an abstract base class
    for implementing specific Hammerstein-Wiener models. It provides
    the foundational structure and parameter management required for
    training and prediction. Subclasses should implement the abstract
    methods to define the specific behaviors of the input and output
    nonlinearities.

    .. math::
        \mathbf{y} = f_{\text{output}}\\
            \left( \mathbf{H} f_{\text{input}} \left( \mathbf{X} \right) \right)

    where:
    :math:`f_{\text{input}}` is the nonlinear input estimator,
    :math:`\mathbf{H}` represents the linear dynamic block (e.g., regression
    coefficients), and
    :math:`f_{\text{output}}` is the nonlinear output estimator.

    Parameters
    ----------
    nonlinear_input_estimator : estimator, default=None
        The estimator to model the nonlinear relationship at the input.
        It must implement the methods `fit` and either `transform` or
        `predict`. If `None`, no nonlinear transformation is applied
        to the input data.
    
    nonlinear_output_estimator : estimator, default=None
        The estimator to model the nonlinear relationship at the output.
        It must implement the methods `fit` and either `transform` or
        `predict`. If `None`, no nonlinear transformation is applied
        to the output data.
    
    p : int, default=1
        The number of lagged observations to include in the model. This
        determines the number of past time steps used to predict the
        current output.
    
    feature_engineering : str, default='auto'
        Method for feature engineering. Currently supports only `'auto'`,
        which enables automatic feature creation based on the number of
        lagged observations.
    
    optimizer : str, default='adam'
        Optimization algorithm to use for training the linear dynamic
        block. Supported options are:
        
        - `'sgd'`: Stochastic Gradient Descent
        - `'adam'`: Adaptive Moment Estimation
        - `'adagrad'`: Adaptive Gradient Algorithm
        
    learning_rate : float, default=0.001
        The initial learning rate for the optimizer. Controls the step size
        during gradient descent updates.
    
    batch_size : int or str, default='auto'
        The number of samples per gradient update. If set to `'auto'`,
        the batch size is determined automatically based on the dataset
        size.
    
    max_iter : int, default=100
        Maximum number of iterations (epochs) for training the linear
        dynamic block.
    
    tol : float, default=1e-3
        Tolerance for the optimization. Training stops when the loss
        improvement is below this threshold.
    
    early_stopping : bool, default=False
        Whether to stop training early if the validation loss does not
        improve after a certain number of iterations.
    
    validation_fraction : float, default=0.1
        The proportion of the training data to set aside as validation data
        for early stopping.
    
    n_iter_no_change : int, default=5
        Number of iterations with no improvement to wait before stopping
        training early.
    
    shuffle : bool, default=True
        Whether to shuffle the training data before each epoch.
    
    epsilon : float, default=1e-15
        A small constant added to avoid division by zero during scaling.
    
    time_weighting : str or None, default=None
        Method for applying time-based weights to the loss function.
        Supported options are:
        
        - `'linear'`: Linearly increasing weights over time.
        - `'exponential'`: Exponentially increasing weights over time.
        - `'inverse'`: Inversely proportional weights over time.
        - `None`: No time-based weighting (equal weights).
        
    random_state : int, RandomState instance, default=None
        Determines random number generation for weights and bias
        Pass an int for reproducible results across multiple function calls.
    
    n_jobs : int or None, default=None
        Number of CPU cores to use during training. `-1` means using all
        available cores. If `None`, the number of jobs is determined
        automatically.
    
    verbose : int, default=0
        Controls the verbosity of the training process. Higher values
        result in more detailed logs.

    Attributes
    ----------
    linear_model_ : SGDRegressor
        The linear dynamic block trained using stochastic gradient descent.
    
    best_loss_ : float or None
        The best validation loss observed during training. Used for early
        stopping.
    
    initial_loss_ : float
        The loss computed on the entire dataset after initial training.
    
    is_fitted_ : bool
        Indicates whether the model has been fitted.
    
    Examples
    --------
    >>> from hwm.estimators._dynamic_system import BaseHammersteinWiener
    >>> from hwm.estimators.dynamic_system import HammersteinWienerRegressor
    >>> from sklearn.preprocessing import StandardScaler
    >>> from sklearn.linear_model import SGDRegressor
    >>> # Initialize the Hammerstein-Wiener regressor with a linear scaler
    >>> hw_regressor = HammersteinWienerRegressor(
    ...     nonlinear_input_estimator=StandardScaler(),
    ...     nonlinear_output_estimator=StandardScaler(),
    ...     p=2,
    ...     optimizer='adam',
    ...     learning_rate=0.01,
    ...     batch_size=64,
    ...     max_iter=500,
    ...     tol=1e-4,
    ...     early_stopping=True,
    ...     validation_fraction=0.2,
    ...     n_iter_no_change=10,
    ...     shuffle=True,
    ...     epsilon=1e-10,
    ...     time_weighting='linear',
    ...     n_jobs=-1,
    ...     verbose=1
    ... )
    >>> # Fit the model on training data
    >>> hw_regressor.fit(X_train, y_train)
    >>> # Make predictions on new data
    >>> predictions = hw_regressor.predict(X_test)

    Notes
    -----
    - The Hammerstein-Wiener model is particularly effective for systems
      where the input-output relationship can be decomposed into distinct
      nonlinear and linear components.
    - Proper selection of the number of lagged observations (`p`) is
      crucial for capturing the temporal dependencies in the data.
    - Time-based weighting can be used to emphasize recent observations
      more than older ones, which is useful in time series forecasting.

    See Also
    --------
    scikit-learn :py:mod:`sklearn.base.BaseEstimator`  
        The base class for all estimators in scikit-learn, providing
        basic parameter management and utility methods.
    
    HammersteinModel :class:`~gofast.estimators.HammersteinModel`  
        A concrete implementation of the Hammerstein-Wiener model.

    References
    ----------
    .. [1] Hammerstein, W. (1950). "Beiträge zum Problem der adaptiven
       Regelung". *Zeitschrift für angewandte Mathematik und Mechanik*,
       30(3), 345-367.
    .. [2] Wiener, N. (1949). "Extrapolation, Interpolation, and Smoothing
       of Stationary Time Series". *The MIT Press*.
    .. [3] Ljung, L. (1999). *System Identification: Theory for the
       User*. Prentice Hall.

    """

    _parameter_constraints: dict = {
        "nonlinear_input_estimator": [
            HasMethods(['fit', 'transform']),
            HasMethods(['fit', 'predict']),
            None
        ],
        "nonlinear_output_estimator": [
            HasMethods(['fit', 'transform']),
            HasMethods(['fit', 'predict']),
            None
        ],
        "p": [Integral],
        "feature_engineering": [StrOptions({'auto'})],
        "optimizer": [StrOptions({"sgd", "adam", "adagrad"})],
        "learning_rate": [Real],
        "batch_size": [Integral, StrOptions({'auto'})],
        "max_iter": [Integral],
        "tol": [Real],
        "early_stopping": [bool],
        "validation_fraction": [Real],
        "n_iter_no_change": [Integral],
        "shuffle": [bool],
        "epsilon": [Real],
        "time_weighting": [StrOptions({"linear", "exponential",
                                       "inverse"}), None],
        "n_jobs": [Integral, None],
        "verbose": [Integral],
    }
    
    DEFAULT_BATCH_SIZE=512 
    
    @abstractmethod 
    def __init__(
        self,
        nonlinear_input_estimator=None,
        nonlinear_output_estimator=None,
        p=1,
        feature_engineering='auto',
        optimizer='adam',
        learning_rate=0.001,
        batch_size='auto',
        max_iter=100,
        tol=1e-3,
        early_stopping=False,
        validation_fraction=0.1,
        n_iter_no_change=5,
        shuffle=True,
        epsilon=1e-15,
        time_weighting=None,
        random_state=None, 
        n_jobs=None,
        verbose=1,
    ):

        self.nonlinear_input_estimator = nonlinear_input_estimator
        self.nonlinear_output_estimator = nonlinear_output_estimator
        self.p = p
        self.feature_engineering = feature_engineering
        self.optimizer = optimizer
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.max_iter = max_iter
        self.tol = tol
        self.early_stopping = early_stopping
        self.validation_fraction = validation_fraction
        self.n_iter_no_change = n_iter_no_change
        self.shuffle = shuffle
        self.epsilon = epsilon
        self.time_weighting = time_weighting
        self.random_state=random_state 
        self.n_jobs = n_jobs
        self.verbose = verbose


    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray, **fit_params) -> None:
        """
        Fit the model to data.
    
        This method must be implemented in subclasses to train the model
        using the provided input features and target values.
    
        Parameters
        ----------
        X : np.ndarray
            The input features, shape (n_samples, n_features).
        y : np.ndarray
            The target values, shape (n_samples,).
        fit_params : keyword arguments
            Additional parameters for fitting the model.
        """
        pass
    
    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict using the model.
    
        This method must be implemented in subclasses to generate predictions
        based on the input features.
    
        Parameters
        ----------
        X : np.ndarray
            The input features, shape (n_samples, n_features).
    
        Returns
        -------
        np.ndarray
            The predicted values, shape (n_samples,).
        """
        pass

    def _validate_input_data(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        y: Union[np.ndarray, pd.Series, pd.DataFrame]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Validate input data and handle DataFrames or Series.
    
        This method checks the types of input features and targets,
        converting DataFrames and Series to NumPy arrays while warning
        about the loss of feature and target names.
    
        Parameters
        ----------
        X : Union[np.ndarray, pd.DataFrame]
            The input features to validate.
        y : Union[np.ndarray, pd.Series, pd.DataFrame]
            The target values to validate.
    
        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            The validated input features and target values as NumPy arrays.
        """
        # Check if X is a DataFrame
        if hasattr(X, "columns"):
            self.feature_names_in_ = list(X.columns)
            X = np.asarray(X)
            warnings.warn(
                "Input features detected as a DataFrame. Feature names will "
                "not be preserved in the processing."
            )
    
        # Check if y is a DataFrame or Series
        if hasattr(y, "columns"):
            self.target_names_in_ = list(y.columns)
            # squeeze data if dimension (n_samples, 1)
            y= y.squeeze() 
            y = np.asarray(y)
            warnings.warn(
                "Input targets detected as a DataFrame. Target names will "
                "not be preserved during training."
            )
        elif hasattr(y, "name"):
            self.target_name_ = y.name
            y = np.asarray(y)
            warnings.warn(
                "Input target detected as a Series. The target name will "
                "not be preserved during training."
            )
    
        return X, y

    def _apply_nonlinear_input(
        self,
        X: np.ndarray,
        y: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Apply the nonlinear input transformation.
    
        This method uses a fitted nonlinear input estimator to transform
        the input features. If no estimator is provided, the original
        input features are returned.
    
        Parameters
        ----------
        X : np.ndarray
            The input features to transform, shape (n_samples, n_features).
        y : Optional[np.ndarray], default=None
            The target values, shape (n_samples,).
    
        Returns
        -------
        np.ndarray
            The transformed input features.
        """

        if self.verbose > 1:
            print("Applying nonlinear input transformation.")
    
        if self.nonlinear_input_estimator is not None:
            try:
                check_is_fitted(self.nonlinear_input_estimator)
            except:
                if self.verbose > 1:
                    print("Fitting nonlinear input estimator.")
                self.nonlinear_input_estimator.fit(
                    X, y) if y is not None else self.nonlinear_input_estimator.fit(X)
            
            if hasattr(self.nonlinear_input_estimator, 'transform'):
                X_transformed = self.nonlinear_input_estimator.transform(X)
            elif hasattr(self.nonlinear_input_estimator, 'predict'):
                X_transformed = self.nonlinear_input_estimator.predict(X)
                if X_transformed.ndim == 1:
                    X_transformed = X_transformed.reshape(-1, 1)
            else:
                raise ValueError(
                    "The nonlinear_input_estimator must have either a"
                    " 'transform' or 'predict' method.")
        else:
            X_transformed = X
            if self.verbose > 1:
                print("No nonlinear input estimator provided; using original X.")
        
        return X_transformed.astype(np.float32)

    def _create_lagged_features(
        self,
        X_transformed: np.ndarray
    ) -> np.ndarray:
        """
        Create lagged features for the linear dynamic block.
        
        This method generates lagged features from the transformed input
        data, where each feature is shifted by a specified number of
        time steps. Lagged features are useful for capturing temporal
        dependencies in time-series data.
        
        Parameters
        ----------
        X_transformed : np.ndarray
            The transformed input data, shape (n_samples, n_features).
        
        Returns
        -------
        np.ndarray
            An array of lagged features with shape
            (n_samples, n_features * p), where p is the number of lags.
        """

        if self.verbose > 1:
            print("Creating lagged features.")
    
        n_samples, n_features = X_transformed.shape
        lagged_features = np.zeros((n_samples, n_features * self.p), dtype=np.float32)
    
        for lag in range(self.p):
            if self.verbose > 1:
                print(f"Creating lag {lag + 1} features.")
            lagged = np.roll(X_transformed, shift=lag, axis=0)
            lagged[:lag, :] = 0
            lagged_features[:, lag * n_features:(lag + 1) * n_features] = lagged
    
        if self.verbose > 1:
            print(f"Lagged features shape: {lagged_features.shape}")
    
        return lagged_features


    def _apply_linear_dynamic_block(
        self,
        X_lagged: np.ndarray
    ) -> np.ndarray:
        """
        Apply the linear dynamic block.
        
        This method utilizes the fitted linear model to compute the
        decision function values for the provided lagged features.
        
        Parameters
        ----------
        X_lagged : np.ndarray
            The input array of lagged features, shape (n_samples, n_features).
        
        Returns
        -------
        np.ndarray
            The decision function values computed by the linear model.
        """
        if self.verbose > 1:
            print("Applying linear dynamic block.")
    
        return self.linear_model_.decision_function(X_lagged)
    
    def _apply_nonlinear_output(
        self,
        y_linear: np.ndarray,
        y: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Apply the nonlinear output transformation.
        
        This method transforms the linear output using a fitted nonlinear
        output estimator if provided. If no estimator is given, it uses
        the linear output directly.
        
        Parameters
        ----------
        y_linear : np.ndarray
            The linear output values, shape (n_samples,).
        y : Optional[np.ndarray], default=None
            The true target values, shape (n_samples,).
        
        Returns
        -------
        np.ndarray
            The transformed output values.
        """

        if self.verbose > 1:
            print("Applying nonlinear output transformation.")
    
        if self.nonlinear_output_estimator is not None:
            # Check if the estimator is fitted
            try:
                check_is_fitted(self.nonlinear_output_estimator)
            except:
                if self.verbose > 1:
                    print("Fitting nonlinear output estimator.")
                if y is not None:
                    self.nonlinear_output_estimator.fit(
                        y_linear.reshape(-1, 1), y
                        )
                else:
                    self.nonlinear_output_estimator.fit(
                        y_linear.reshape(-1, 1), y_linear
                        )
    
            # Use transform or predict based on estimator type
            if hasattr(self.nonlinear_output_estimator, 'transform'):
                y_pred = self.nonlinear_output_estimator.transform(
                    y_linear.reshape(-1, 1)
                    )
            elif hasattr(self.nonlinear_output_estimator, 'predict'):
                y_pred = self.nonlinear_output_estimator.predict(
                    y_linear.reshape(-1, 1)
                    )
            else:
                raise ValueError("The nonlinear_output_estimator must have"
                                 " either a 'transform' or 'predict' method.")
            
            # Ensure the output is 2D
            if y_pred.ndim == 1:
                y_pred = y_pred.reshape(-1, 1)
        else:
            # No nonlinear output estimator provided; use linear output
            y_pred = y_linear.reshape(-1, 1)
            if self.verbose > 1:
                print(
                    "No nonlinear output estimator provided; using linear output.")
        
        return y_pred.astype(np.float32)
    
    def _split_data(
        self,
        X: np.ndarray,
        y: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, Optional[np.ndarray],
               Optional[np.ndarray]]:
        """
        Split data into training and validation sets.
        
        This method divides the dataset into training and validation subsets based
        on the specified validation fraction. If the validation fraction is set
        between 0.0 and 1.0, it performs a stratified split to maintain the class
        distribution. Otherwise, it returns the original dataset without splitting.
        
        Parameters
        ----------
        X : np.ndarray
            Feature matrix.
        y : np.ndarray
            Target vector.
        
        Returns
        -------
        tuple[np.ndarray, np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]
            Returns a tuple containing:
            - X_train: Training feature matrix.
            - X_val: Validation feature matrix (None if not split).
            - y_train: Training target vector.
            - y_val: Validation target vector (None if not split).
        """
        # Check if validation_fraction is within the valid range
        stratify = None  if self._estimator_type =='regressor' else y 
        if 0.0 < self.validation_fraction < 1.0:
            # Perform stratified train-test split to maintain class distribution
            X_train, X_val, y_train, y_val = train_test_split(
                X,
                y,
                test_size=self.validation_fraction,
                random_state=42,
                stratify=stratify
            )
            return X_train, X_val, y_train, y_val
        else:
            # If validation_fraction is not set, return the entire dataset
            return X, None, y,  None
    

    def _compute_time_weights(
        self,
        n_samples: int
    ) -> np.ndarray:
        """
        Compute time weights based on the selected weighting scheme.
        
        This method generates an array of weights to be applied to samples
        based on the specified `time_weighting` strategy. The available
        strategies are "linear", "exponential", and "inverse". If an
        unrecognized scheme is provided, it defaults to equal weights.
        
        Parameters
        ----------
        n_samples : int
            The number of samples to compute weights for.
        
        Returns
        -------
        np.ndarray
            An array of computed weights with shape (n_samples,).
        """
        if self.verbose > 1:
            # Log the selected time weighting method
            print(
                f"Computing time weights using "
                f"{self.time_weighting} method."
            )
        
        # Compute weights based on the selected method
        if self.time_weighting == "linear":
            # Linear weighting: weights increase uniformly from 0.1 to 1.0
            weights = np.linspace(0.1, 1.0, n_samples)
        elif self.time_weighting == "exponential":
            # Exponential weighting: weights increase exponentially
            weights = np.exp(np.linspace(0, 1, n_samples)) - 1
            weights /= weights.max()  # Normalize weights to [0, 1]
        elif self.time_weighting == "inverse":
            # Inverse weighting: weights decrease with sample index
            weights = 1 / np.arange(1, n_samples + 1)
            weights /= weights.max()  # Normalize weights to [0, 1]
        else:
            # Default weighting: equal weights for all samples
            weights = np.ones(n_samples)
        
        if self.verbose > 1:
            # Log the computed weights
            print(f"Time weights: {weights}")
        
        return weights

    def _handle_early_stopping(
        self,
        val_loss: float
    ) -> None:
        """
        Manage early stopping logic based on validation loss.
        
        This method updates the count of consecutive epochs without
        improvement in validation loss. If the validation loss improves
        by more than the specified tolerance, the no improvement counter
        is reset. Otherwise, the counter is incremented. When the counter
        reaches `n_iter_no_change`, early stopping is triggered.
        
        Parameters
        ----------
        val_loss : float
            The current epoch's validation loss.
        
        Returns
        -------
        None
            Updates internal counters and best loss in place.
        """
        if val_loss < (self.best_loss_ - self.tol):
            # Improvement detected: reset no improvement counter
            self._no_improvement_count = 0
            # Update the best observed validation loss
            self.best_loss_ = val_loss
            if self.verbose > 1:
                print(
                    f"Validation loss improved to {self.best_loss_:.6f}. "
                    "Resetting no improvement counter."
                )
        else:
            # No significant improvement: increment counter
            self._no_improvement_count += 1
            if self.verbose > 1:
                print(
                    f"No improvement in validation loss for "
                    f"{self._no_improvement_count} consecutive epochs."
                )
            
            # Check if early stopping criterion is met
            if self._no_improvement_count >= self.n_iter_no_change:
                if self.verbose > 1:
                    print(
                        f"Early stopping triggered after "
                        f"{self._no_improvement_count} epochs without "
                        "improvement."
                    )
                # Implement early stopping logic, e.g., set a flag or raise
                # an exception to halt training externally
                self.early_stopping_triggered_ = True

    def _train_epoch(
        self,
        y_train,
        X_y_batches, 
        X_val,
        y_val,
        metrics,
        epoch,
        bar=None,
        epoch_metrics= None
    ) -> None:
        """
        Train the model for one epoch.
        
        This method performs training over all batches in the training data for a 
        single epoch. It handles shuffling, batching, model updates via partial_fit, 
        metric evaluations, and progress bar updates.
        
        Parameters
        ----------
        y_train : np.ndarray
            Training labels.
        X_y_batches: List of Tuple ( np.ndarray, np.ndarray)
           List of batches in pair (X_batch, y_batch) sampling for 
           training as a couple of Training features and labels.
        X_val : Optional[np.ndarray]
            Validation features.
        y_val : Optional[np.ndarray]
            Validation labels.
        n_batches : int
            Number of batches per epoch.
        metrics : dict[str, float]
            Dictionary to store aggregated metrics.
        epoch : int
            Current epoch number (0-based indexing).
        bar : Optional[TrainingProgressBar], optional
            Progress bar instance for visual feedback, by default None.
        epoch_metrics : Optional[dict[str, list[float]]], optional
            Dictionary to collect metrics for the current epoch, by default None.
        
        Returns
        -------
        None
            Updates metrics and best_metrics in place.
        """
        if epoch_metrics is None:
            # Initialize epoch_metrics if not provided
            epoch_metrics = {}
        
        for batch_idx, ( X_batch, y_batch) in enumerate (X_y_batches): 

            # Initialize step_metrics to collect metrics across the batch
            step_metrics: dict[str, float] = {}
        
            if self.verbose > 1:
                # Print batch details if verbosity level is high
                msg =f"Batch {batch_idx + 1}/{len(X_y_batches)}:"
                print(msg)
                if self.verbose > 2: 
                    print(
                        f"{msg} X_batch shape {X_batch.shape}, "
                        f"y_batch shape {y_batch.shape}"
                    )
            if epoch == 0 and batch_idx == 0: 
                # For the first batch, provide classes if classifier
                if self._estimator_type=='classifier':
                    # Initialize partial_fit with classes for classifiers
                    self.linear_model_.partial_fit(
                        X_batch, y_batch, classes=np.unique(y_train)
                    )
                else:
                    # Initialize partial_fit without classes for regressors
                    self.linear_model_.partial_fit(X_batch, y_batch)
            else:
                # Perform partial_fit on subsequent batches
                self.linear_model_.partial_fit(X_batch, y_batch)
            
            if batch_idx == len(X_y_batches) -1:
                # Evaluate the current batch and update metrics
                step_metrics, epoch_metrics = self._evaluate_batch(
                    X_batch=X_batch,
                    y_batch=y_batch,
                    X_val=X_val,
                    y_val=y_val,
                    metrics=metrics,
                    batch_idx=batch_idx,
                    step_metrics=step_metrics,
                    epoch_metrics=epoch_metrics
                )
                
            if bar is not None:
                # Update the progress bar with current batch metrics
                bar.update(
                    step=batch_idx + 1,
                    epoch=epoch,
                    step_metrics=step_metrics, 
                    is_epoch_over= (batch_idx + 1) ==len(X_y_batches)
                )
            
            # Reinitialize step_metrics for the next batch
            step_metrics.clear()
        
        # After all batches, update best_metrics with epoch_metrics
        self._update_metrics_from_epoch(
            epoch=epoch,
            epoch_metrics=epoch_metrics,
            best_metrics=metrics
        )

    def _update_metrics_from_epoch(
        self,
        epoch: int,
        epoch_metrics: dict[str, list[float]],
        best_metrics: dict[str, float]
    ) -> None:
        """
        Update the best_metrics dictionary with the best values from the 
        current epoch.
    
        This method iterates over each metric collected during the epoch, determines
        whether the metric should be minimized or maximized based on its name, and
        updates the best_metrics accordingly. After updating, it clears the
        epoch_metrics for the next epoch.
    
        Parameters
        ----------
        epoch : int
            The current epoch number (0-based indexing).
        epoch_metrics : dict[str, list[float]]
            A dictionary where keys are metric names and values are lists 
            of metric values collected during the epoch.
        best_metrics : dict[str, float]
            A dictionary storing the best (minimum or maximum) values observed for
            each metric across all epochs.
    
        Returns
        -------
        None
            The method updates the best_metrics in place and clears epoch_metrics.
        """
        # Iterate over each metric and its corresponding list of values
        # collected during the epoch
        for metric_name, metric_values in epoch_metrics.items():
            # Determine if the metric should be minimized
            # (e.g., 'loss', 'pss') or maximized
            is_minimization_metric = any(
                keyword in metric_name.lower() for keyword in ["loss", "pss"]
            )
    
            if is_minimization_metric:
                # For metrics to be minimized, find the minimum value
                epoch_best = min(metric_values)
                if epoch == 0:
                    # Initialize best_metrics with the first epoch's best
                    best_metrics[metric_name] = round(epoch_best, 4)
                else:
                    # Update best_metrics if the current epoch's best is better
                    best_metrics[metric_name] = min(
                        best_metrics[metric_name],
                        epoch_best
                    )
            else:
                # For metrics to be maximized, find the maximum value
                epoch_best = max(metric_values)
                if epoch == 0:
                    # Initialize best_metrics with the first epoch's best
                    best_metrics[metric_name] = epoch_best
                else:
                    # Update best_metrics if the current epoch's best is better
                    best_metrics[metric_name] = max(
                        best_metrics[metric_name],
                        epoch_best
                    )
        # Clear the epoch_metrics dictionary to prepare for the next epoch
        epoch_metrics.clear()