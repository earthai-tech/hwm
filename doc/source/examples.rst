.. _examples: 

=========================
Examples and Tutorials
=========================

The `hwm` package includes various tools and models for advanced dynamic 
system modeling, particularly suited for nonlinear and time-dependent 
data. This section provides practical examples to demonstrate the 
capabilities of the `hwm` package. The examples range from simple 
dataset generation to complex applications using real-world datasets, 
such as the KDD Cup 1999 dataset.

Example 1: Using the Hammerstein-Wiener Classifier on the KDD Cup 1999 Dataset
-----------------------------------------------------------------------------

The KDD Cup 1999 dataset, a well-known benchmark in intelligent network 
modeling, is used here to demonstrate the `HammersteinWienerClassifier` 
in detecting network intrusions. This example preprocesses data, applies 
a custom ReLU transformer for nonlinear transformations, and compares 
the performance of the `HammersteinWienerClassifier` with an LSTM 
model for anomaly detection.

.. code-block:: python

    # KDD Cup 1999 Example for Intelligent Network Modeling
    # This example demonstrates the use of the Hammerstein-Wiener model
    # on the KDD Cup dataset, highlighting intelligent network anomaly 
    # detection using a structured and flexible modeling approach.
    # https://kdd.ics.uci.edu/databases/kddcup99/kddcup99.html

    import os 
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from sklearn.preprocessing import StandardScaler, OneHotEncoder
    from sklearn.compose import ColumnTransformer
    from sklearn.model_selection import train_test_split, RandomizedSearchCV
    from sklearn.metrics import accuracy_score, roc_curve, auc, ConfusionMatrixDisplay
    from hwm.estimators import HammersteinWienerClassifier
    from hwm.metrics import prediction_stability_score, twa_score
    from hwm.utils import resample_data 
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense
    from tensorflow.keras.callbacks import EarlyStopping

    # Load and sample the KDD Cup 1999 Dataset
    data_path = r'F:\repositories'
    column_names = [ ... ]  # List of columns from the dataset
    continuous_features = [ ... ]  # List of continuous feature names
    categorical_features = [ ... ]  # List of categorical feature names

    data = pd.read_csv(os.path.join(data_path, 'kddcup.data_10_percent_corrected'),
                       names=column_names, header=None)
    data = resample_data(data, samples=100000, random_state=42)

    # Data Preprocessing
    data['label'] = data['label'].apply(lambda x: 0 if x == 'normal.' else 1)
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), continuous_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ]
    )
    X = data.drop('label', axis=1)
    y = data['label']
    X_processed = preprocessor.fit_transform(X)

    # Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X_processed, y, test_size=0.2, random_state=42, stratify=y
    )

    # Custom ReLU Transformer
    from sklearn.base import BaseEstimator, TransformerMixin

    class ReLUTransformer(BaseEstimator, TransformerMixin):
        """Applies the ReLU activation function for nonlinear transformations."""
        def fit(self, X, y=None):
            return self
        def transform(self, X):
            return np.maximum(0, X)

    # Define and Train the Hammerstein-Wiener Classifier
    model = HammersteinWienerClassifier(
        nonlinear_input_estimator=ReLUTransformer(),
        nonlinear_output_estimator=ReLUTransformer(),
        p=9,
        loss="cross_entropy",
        time_weighting="linear",
        optimizer='sgd',
        learning_rate=0.001,
        max_iter=173, 
        early_stopping=True,
        verbose=1
    )
    model.fit(X_train, y_train)

    # Evaluate the Hammerstein-Wiener Classifier
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    pss = prediction_stability_score(y_pred_proba)
    twa = twa_score(y_test, y_pred, alpha=0.9)

    # Plot Results for Hammerstein-Wiener
    def plot_results(y_true, y_pred, y_pred_proba, title):
        ConfusionMatrixDisplay.from_predictions(y_true, y_pred)
        plt.title(f'Confusion Matrix - {title}')
        plt.show()
        fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
        plt.plot(fpr, tpr, label=f'{title} (AUC = {auc(fpr, tpr):.4f})')
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.legend(loc='lower right')
        plt.show()

    plot_results(y_test, y_pred, y_pred_proba, 'Hammerstein-Wiener Classifier')

    # Train and Evaluate an LSTM Model for Comparison
    n_features = X_processed.shape[1]
    X_train_lstm, X_test_lstm = X_train.reshape(-1, 10, n_features), X_test.reshape(-1, 10, n_features)

    lstm_model = Sequential([
        LSTM(64, input_shape=(10, n_features)),
        Dense(1, activation='sigmoid')
    ])
    lstm_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
    lstm_model.fit(X_train_lstm, y_train, epochs=10, batch_size=64, validation_split=0.1, callbacks=[early_stopping])
    y_pred_lstm_proba = lstm_model.predict(X_test_lstm).flatten()
    y_pred_lstm = (y_pred_lstm_proba >= 0.5).astype(int)

    # Plot Results for LSTM
    plot_results(y_test, y_pred_lstm, y_pred_lstm_proba, 'LSTM Model')

    # Print Comparison Summary
    print("Hammerstein-Wiener Classifier Accuracy:", accuracy)
    print("Hammerstein-Wiener PSS:", pss)
    print("Hammerstein-Wiener TWA:", twa)

.. note::

    The KDD Cup dataset is well-suited for evaluating network intrusion 
    models. The `HammersteinWienerClassifier` offers flexibility in 
    handling complex time-based dependencies with customizable lagged 
    features. This example showcases how intelligent network monitoring 
    can benefit from both traditional machine learning and advanced 
    dynamic system models.

Other Examples
----------------
Explore the `examples/` directory for additional use cases, including:

- **Dynamic System Regression**: Using the `HammersteinWienerRegressor` 
  to predict time-dependent targets.
- **Financial Trend Forecasting**: Applying `hwm` models on synthetic 
  financial datasets to predict market trends.
- **Evaluation with Custom Metrics**: Calculating `prediction_stability_score` 
  and `twa_score` for assessing time-series model performance.

Each example provides code comments and explanations to facilitate 
learning and experimentation. We recommend following the provided 
examples step-by-step to familiarize yourself with the `hwm` API.

.. seealso::
    - :ref:`User Guide <user_guide>` for a more comprehensive overview 
      of package features and usage.
    - :ref:`API Reference <api_ref>` for detailed documentation on each 
      module, class, and function in `hwm`.

