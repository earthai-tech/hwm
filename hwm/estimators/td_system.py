# -*- coding: utf-8 -*-
"""
Created on Sun Feb 16 15:47:49 2025

@author: Daniel
"""

import numpy as np
import warnings 

from sklearn.base import RegressorMixin
from sklearn.base import ClassifierMixin
from sklearn.utils.validation import check_is_fitted, check_array

from ._dynamic_system import BaseHammersteinWiener 

from keras.layers import LSTM, Dense, Dropout
from keras.models import Sequential
from keras.optimizers import Adam

class TDHWRegressor(BaseHammersteinWiener, RegressorMixin):
    def __init__(
        self,
        nonlinear_input_estimator=None,
        nonlinear_output_estimator=None,
        p=1,
        time_weighting="linear",
        batch_size="auto",
        optimizer='adam',
        loss='mean_squared_error', 
        learning_rate=0.001,
        max_iter=100,
        tol=1e-3,
        early_stopping=False,
        validation_fraction=0.1,
        n_iter_no_change=5,
        random_state=None,
        n_jobs=None, 
        memory_units=50,
        dropout_rate=0.2,
        memory_layers=1, 
        verbose=1,
    ):
        super().__init__(
            nonlinear_input_estimator=nonlinear_input_estimator,
            nonlinear_output_estimator=nonlinear_output_estimator,
            p=p,
            time_weighting=time_weighting,
            batch_size=batch_size,
            optimizer=optimizer,
            learning_rate=learning_rate,
            max_iter=max_iter,
            tol=tol,
            early_stopping=early_stopping,
            validation_fraction=validation_fraction,
            n_iter_no_change=n_iter_no_change,
            random_state=random_state,
            n_jobs=n_jobs,
            verbose=verbose
        )
        self.memory_units = memory_units
        self.dropout_rate = dropout_rate
        self.memory_layers = memory_layers
        self.loss= loss 

    def _initialize_model(self) -> None:
        """
        Initialize the LSTM-based model with Hammerstein-Wiener.
        """
        self.model = Sequential()

        # Adding LSTM layers
        for _ in range(self.lstm_layers):
            self.model.add(
                LSTM(
                    self.memory_units,
                    return_sequences=True if _ < self.memory_layers - 1 else False)
                )
            self.model.add(Dropout(self.dropout_rate))

        # Output layer (linear output followed by nonlinear transformation)
        self.model.add(Dense(1))

        # Compile model with Adam optimizer
        if self.optimizer=='auto': 
            self.optimizer=Adam(lr=self.learning_rate)
            
        self.model.compile(
            optimizer=self.optimizer, 
            loss=self.loss, 
        )

    def fit(
        self, X, y, **fit_params
        ):
        """
        Fit the model to data using LSTM to learn temporal dependencies.
        """
        X, y = self._validate_input_data(X, y)
        X_transformed = self._apply_nonlinear_input(X, y)
        X_lagged = self._create_lagged_features(X_transformed)

        # Reshape X_lagged to fit LSTM input format (samples, timesteps, features)
        X_lagged = X_lagged.reshape(X_lagged.shape[0], 1, X_lagged.shape[1])

        self._initialize_model()

        callbacks = fit_params.pop ("callbacks", None)
        
        self.history= self.model.fit(
            X_lagged, y, 
            batch_size=self.batch_size,
            epochs=self.max_iter, 
            verbose=self.verbose, 
            callbacks= callbacks, 
            **fit_params
            )
        self.history_= self.history.history 
        
        return self

    def predict(self, X) -> np.ndarray:
        """
        Predict using the fitted LSTM-based model.
        """
        check_is_fitted(self, 'model')
        X = check_array(X)

        # Prepare input data
        X_transformed = self._apply_nonlinear_input(X)
        X_lagged = self._create_lagged_features(X_transformed)

        # Reshape data for LSTM (samples, timesteps, features)
        X_lagged = X_lagged.reshape(X_lagged.shape[0], 1, X_lagged.shape[1])

        # Make predictions
        y_pred_linear = self.model.predict(X_lagged)

        return self._apply_nonlinear_output(y_pred_linear)

    def _compute_loss(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return np.mean((y_true - y_pred) ** 2)


class TDHWClassifier(BaseHammersteinWiener, ClassifierMixin):
    def __init__(
        self,
        nonlinear_input_estimator=None,
        nonlinear_output_estimator=None,
        p=1,
        time_weighting="linear",
        batch_size="auto",
        optimizer='adam',
        loss='binary_crossentropy',
        activation='auto',  
        learning_rate=0.001,
        max_iter=100,
        tol=1e-3,
        early_stopping=False,
        validation_fraction=0.1,
        n_iter_no_change=5,
        random_state=None,
        n_jobs=None,
        memory_units=50,
        dropout_rate=0.2,
        epsilon=1e-15,
        memory_layers=1,
        verbose=1,
    ):
        super().__init__(
            nonlinear_input_estimator=nonlinear_input_estimator,
            nonlinear_output_estimator=nonlinear_output_estimator,
            p=p,
            time_weighting=time_weighting,
            batch_size=batch_size,
            optimizer=optimizer,
            learning_rate=learning_rate,
            max_iter=max_iter,
            tol=tol,
            early_stopping=early_stopping,
            validation_fraction=validation_fraction,
            n_iter_no_change=n_iter_no_change,
            random_state=random_state,
            n_jobs=n_jobs,
            verbose=verbose
        )

        self.memory_units = memory_units
        self.dropout_rate = dropout_rate
        self.memory_layers = memory_layers
        self.loss = loss
        self.activation = activation
        self.epsilon = epsilon

    def _initialize_model(self) -> None:
        """
        Initialize the LSTM-based model with Hammerstein-Wiener for classification.
        
        The model is constructed using an LSTM-based architecture. The output layer 
        activation function is determined based on the loss function. If the task is 
        binary classification, 'sigmoid' is used, and for multiclass classification, 
        'softmax' is applied.
        """
        self.model = Sequential()

        # Add LSTM layers
        for _ in range(self.memory_layers):
            self.model.add(
                LSTM(
                    self.memory_units,
                    return_sequences=True
                    if _ < self.memory_layers - 1
                    else False
                )
            )
            self.model.add(Dropout(self.dropout_rate))

        # Handle auto activation selection
        if self.activation == 'auto':
            if self.loss == 'binary_crossentropy':
                self.activation = 'sigmoid'
            elif self.loss == 'categorical_crossentropy':
                self.activation = 'softmax'
            else:
                raise ValueError(
                    "Invalid loss function for 'auto' activation"
                )

        # Output layer
        if self.loss == 'binary_crossentropy':
            self.model.add(Dense(1, activation=self.activation))
        elif self.loss == 'categorical_crossentropy':
            self.model.add(
                Dense(self.num_classes_, activation=self.activation)
            )
        else:
            raise ValueError("Unsupported loss function")

        # Compile model
        if self.optimizer =='auto': 
            self.optimizer = Adam(lr= self.learning_rate)
        self.model.compile(
            optimizer=self.optimizer, 
            loss=self.loss,
            metrics=['accuracy']
        )

    def fit(self, X, y, **fit_params):
        """
        Fit the LSTM model to the training data, capturing temporal dependencies.
        
        This method transforms the input features and labels, applies lagging,
        and trains the model using the specified number of epochs.
        """
        callbacks = fit_params.pop("callbacks", None)

        # Validate and preprocess input data
        X, y = self._validate_input_data(X, y)
        X_transformed = self._apply_nonlinear_input(X, y)
        X_lagged = self._create_lagged_features(X_transformed)

        # Auto-detect binary or multiclass and adjust activation/loss accordingly
        self.num_classes_ = np.unique(y)
        if self.activation == 'auto':
            if self.num_classes_.size <= 2:
                self.activation = 'sigmoid'
                if self.loss != 'binary_crossentropy':
                    warnings.warn(
                        "Loss function 'binary_crossentropy' is recommended for "
                        "binary classification. Switching to binary_crossentropy."
                    )
                    self.loss = 'binary_crossentropy'
            else:
                self.activation = 'softmax'
                if self.loss != 'categorical_crossentropy':
                    warnings.warn(
                        "Loss function 'categorical_crossentropy' is"
                        " recommended for multiclass classification."
                        " Switching to categorical_crossentropy."
                    )
                    self.loss = 'categorical_crossentropy'

        # Reshape for LSTM input
        X_lagged = X_lagged.reshape(X_lagged.shape[0], 1, X_lagged.shape[1])

        # Initialize and train model
        self._initialize_model()
        self.history = self.model.fit(
            X_lagged, y, 
            batch_size=self.batch_size,
            epochs=self.max_iter, 
            verbose=self.verbose,
            callbacks=callbacks, 
            **fit_params
        )

        self.history_ = self.history.history
        
        return self

    def predict(self, X):
        """
        Predict class labels using the fitted LSTM model.
        
        This method applies the trained model to the input data and returns 
        predicted class labels based on the learned temporal dependencies.
        """
        check_is_fitted(self, 'model')
        X = check_array(X)

        # Apply nonlinear transformations and create lagged features
        X_transformed = self._apply_nonlinear_input(X)
        X_lagged = self._create_lagged_features(X_transformed)

        # Reshape for LSTM input
        X_lagged = X_lagged.reshape(X_lagged.shape[0], 1, X_lagged.shape[1])

        # Make predictions
        y_pred_linear = self.model.predict(X_lagged)

        # Handle binary or multiclass classification
        if self.loss == 'binary_crossentropy':
            return (y_pred_linear > 0.5).astype(int)
        elif self.loss == 'categorical_crossentropy':
            
            return np.argmax(y_pred_linear, axis=1)

    def predict_proba(self, X) -> np.ndarray:
        """
        Predict class probabilities for input samples.
        
        This method generates probability estimates for each class for the
        input samples. It uses the fitted model and applies appropriate
        transformations to return the probabilities.
        """
        check_is_fitted(self, 'model')
        X = check_array(X)

        # Prepare input data
        X_transformed = self._apply_nonlinear_input(X)
        X_lagged = self._create_lagged_features(X_transformed)

        # Reshape for LSTM input
        X_lagged = X_lagged.reshape(X_lagged.shape[0], 1, X_lagged.shape[1])

        # Get class probabilities
        y_pred_proba = self.model.predict(X_lagged)
        return y_pred_proba

    def _compute_loss(self, y_true, y_pred) -> float:
        """
        Compute the loss based on the specified loss function.
        
        This method calculates the loss between the true labels and predicted
        probabilities. Supports binary cross-entropy and categorical cross-entropy.
        """
        y_pred = np.clip(y_pred, self.epsilon, 1. - self.epsilon)

        if self.loss == 'binary_crossentropy':
            loss = -np.mean(
                y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred)
            )
        elif self.loss == 'categorical_crossentropy':
            loss = -np.mean(np.sum(y_true * np.log(y_pred), axis=1))
        else:
            raise ValueError("Unsupported loss function")

        return loss
