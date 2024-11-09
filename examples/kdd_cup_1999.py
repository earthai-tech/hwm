# -*- coding: utf-8 -*-
# 
"""
Created on Fri Nov  1 17:36:16 2024
@author: Daniel
[ write a wonderfull docstring here a preatical test for intelligent networks 
#  in this example using KDD Cup dataset. ...
# explain more details 
 come back to new line after 70 characters]

"""
import os 
from scipy.stats import randint, uniform
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import RandomizedSearchCV

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_curve, auc, ConfusionMatrixDisplay
from sklearn.base import BaseEstimator, TransformerMixin
import matplotlib.pyplot as plt
# from sklearn.neural_network import MLPClassifier
# import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping

from hwm.estimators import HammersteinWienerClassifier
from hwm.metrics import prediction_stability_score, twa_score
from hwm.utils import resample_data 

data_path =r'F:\repositories'
# %% 
# Load the KDD Cup 1999 Dataset
column_names = [
    'duration', 'protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes',
    'land', 'wrong_fragment', 'urgent', 'hot', 'num_failed_logins',
    'logged_in', 'num_compromised', 'root_shell', 'su_attempted',
    'num_root', 'num_file_creations', 'num_shells', 'num_access_files',
    'num_outbound_cmds', 'is_host_login', 'is_guest_login', 'count',
    'srv_count', 'serror_rate', 'srv_serror_rate', 'rerror_rate',
    'srv_rerror_rate', 'same_srv_rate', 'diff_srv_rate', 'srv_diff_host_rate',
    'dst_host_count', 'dst_host_srv_count', 'dst_host_same_srv_rate',
    'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate',
    'dst_host_srv_diff_host_rate', 'dst_host_serror_rate',
    'dst_host_srv_serror_rate', 'dst_host_rerror_rate',
    'dst_host_srv_rerror_rate', 'label'
]
continuous_features = [
    'duration', 'src_bytes', 'dst_bytes', 'wrong_fragment', 'urgent', 'hot',
    'num_failed_logins', 'num_compromised', 'num_root', 'num_file_creations',
    'num_shells', 'num_access_files', 'count', 'srv_count', 'serror_rate',
    'srv_serror_rate', 'rerror_rate', 'srv_rerror_rate', 'same_srv_rate',
    'diff_srv_rate', 'srv_diff_host_rate', 'dst_host_count',
    'dst_host_srv_count', 'dst_host_same_srv_rate', 'dst_host_diff_srv_rate',
    'dst_host_same_src_port_rate', 'dst_host_srv_diff_host_rate',
    'dst_host_serror_rate', 'dst_host_srv_serror_rate',
    'dst_host_rerror_rate', 'dst_host_srv_rerror_rate'
]

categorical_features = ['protocol_type', 'service', 'flag', 'land', 'logged_in',
                        'is_host_login', 'is_guest_login', 'root_shell', 'su_attempted',
                        'num_outbound_cmds']

data = pd.read_csv(os.path.join(data_path, 'kddcup.data_10_percent_corrected'),
                   names=column_names, header=None)
# %% samples 100 000 samples of data 
data  = resample_data(data, samples= 100000, random_state= 42 )
# %%
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
    
    
#%%
# No need lag will inner be created via the parameter p

# Define the function to create lagged features
def create_lagged_features(X, p):
    # Create lagged DataFrame for each lag from 1 to p
    lagged_features = [X.shift(i).add_suffix(f'_lag_{i}') for i in range(1, p + 1)]
    # Concatenate lagged features along the column axis
    X_lagged = pd.concat(lagged_features, axis=1)
    return X_lagged

p = 5
#%%
# X_processed_df = pd.DataFrame(X_processed)  # Assuming X_processed is already defined
# X_lagged = create_lagged_features(X_processed_df, p)

# # Concatenate the original features and lagged features, then drop rows with NaNs
# X_combined = pd.concat([X_processed_df, X_lagged], axis=1).dropna().reset_index(drop=True)

# # Adjust target variable `y` to match the lagged features
# y = y.iloc[p:].reset_index(drop=True)

# # Split the Data into Training and Testing Sets

# X_train, X_test, y_train, y_test = train_test_split(
#     X_combined, y, test_size=0.2, random_state=42, stratify=y)

#%%
# you can go straight for to the package 
X_train, X_test, y_train, y_test = train_test_split(
    X_processed, y.values, test_size=0.2, random_state=42, stratify=y)
#%%
# Since we want to use ReLU activation functions for the nonlinear input and 
# output transformations, we'll define custom transformers.

class ReLUTransformer(BaseEstimator, TransformerMixin):
    """Custom transformer that applies the ReLU activation function."""
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        return np.maximum(0, X)
#%%
# Define the Hammerstein-Wiener Classifier with new parameters
model = HammersteinWienerClassifier(
    nonlinear_input_estimator=ReLUTransformer(),
    nonlinear_output_estimator=ReLUTransformer(),
    p=9,
    loss="cross_entropy",
    time_weighting="linear",
    batch_size="auto",
    optimizer='sgd',
    learning_rate=0.001,
    max_iter=173, 
    early_stopping=True,
    verbose=1, 
   
)
#batch_size=84, learning_rate=0.005967511656638482, max_iter=173, optimizer=sgd, p=9
# Train the Model
model.fit(X_train, y_train)

# %% You can train the HWM to Find the best parameters using randomsearch : 

# Define the parameter grid for RandomizedSearchCV
param_distributions = {
    'p': randint(1, 10),  # Dependency order from 1 to 10
    'batch_size': randint(32, 128),  # Batch size between 32 and 128
    'optimizer': ['sgd', 'adam', 'adagrad'],  # Optimizers to choose from
    'learning_rate': uniform(0.0001, 0.01),  # Learning rate from 0.0001 to 0.01
    'max_iter': randint(50, 200)  # Max iterations between 50 and 200
}

# Initialize the Hammerstein-Wiener Classifier with fixed components
model = HammersteinWienerClassifier(
    nonlinear_input_estimator=ReLUTransformer(),
    nonlinear_output_estimator=ReLUTransformer(),
    loss="cross_entropy",
    time_weighting="linear",
    verbose=0, 
    batch_size = 200, 
    early_stopping=True, 
)

# Initialize RandomizedSearchCV with the model and parameter distributions
random_search = RandomizedSearchCV(
    estimator=model,
    param_distributions=param_distributions,
    n_iter=20,  # Number of parameter settings sampled
    scoring='accuracy',  # Evaluation metric
    cv=3,  # 3-fold cross-validation
    verbose=0,
    random_state=42,
    n_jobs=-1  # Use all available cores

)

# Fit the RandomizedSearchCV to find the best parameters
random_search.fit(X_train, y_train)

# Display the best parameters and the corresponding score
print("Best Parameters:", random_search.best_params_)
print("Best Score:", random_search.best_score_)

# Use the best estimator to make predictions
best_model = random_search.best_estimator_
y_pred = best_model.predict(X_test)

# Evaluate the model with the test data
accuracy = accuracy_score(y_test, y_pred)
print(f"Test Accuracy: {accuracy:.4f}")
    
#%%
# Evaluate the Model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
y_pred_proba = model.predict_proba(X_test)[:, 1]
pss = prediction_stability_score(y_pred_proba)
twa = twa_score(y_test, y_pred, alpha=0.9)
#%%
# Plot Results
def plot_results(y_true, y_pred, y_pred_proba, title):
    # Confusion Matrix
    ConfusionMatrixDisplay.from_predictions(y_true, y_pred)
    plt.title(f'Confusion Matrix - {title}')
    plt.show()

    # ROC Curve
    fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, label=f'{title} (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.title('ROC Curve')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc='lower right')
    plt.show()

# Plot Hammerstein-Wiener Results
plot_results(y_test, y_pred, y_pred_proba, 'Hammerstein-Wiener Classifier')

# Define and Train the LSTM Model
n_features = X_processed.shape[1]
try: 
    X_train_lstm = X_train.values.reshape(-1, p + 1, n_features).values
    X_test_lstm = X_test.values.reshape(-1, p + 1, n_features).values
    y_test_lstm = y_test.values
    input_shape = p + 1
except:
    X_train_lstm = X_train
    X_test_lstm = X_test
    y_test_lstm = y_test
    input_shape = X_train_lstm.shape[0]
    

lstm_model = Sequential([
    LSTM(64, input_shape=(p + 1, n_features)),
    Dense(1, activation='sigmoid')
])
lstm_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

# Train the LSTM Model
lstm_history = lstm_model.fit(
    X_train_lstm, y_train,
    epochs=10,
    batch_size=64,
    validation_split=0.1,
    callbacks=[early_stopping]
)

# Evaluate the LSTM Model
lstm_loss, lstm_accuracy = lstm_model.evaluate(X_test_lstm, y_test.values)
y_pred_lstm_proba = lstm_model.predict(X_test_lstm).flatten()
y_pred_lstm = (y_pred_lstm_proba >= 0.5).astype(int)
pss_lstm = prediction_stability_score(y_pred_lstm_proba)
twa_lstm = twa_score(y_test.values, y_pred_lstm, alpha=0.9)

# Plot LSTM Results
plot_results(y_test.values, y_pred_lstm, y_pred_lstm_proba, 'LSTM Model')

# Compare ROC Curves
# Compute ROC curve

fpr_lstm, tpr_lstm, thresholds_lstm = roc_curve(y_test_lstm, y_pred_lstm_proba)
roc_auc_lstm = auc(fpr_lstm, tpr_lstm)

fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, label=f'Hammerstein-Wiener (AUC = {auc(fpr, tpr):.4f})')
plt.plot(fpr_lstm, tpr_lstm, label=f'LSTM (AUC = {auc(fpr_lstm, tpr_lstm):.4f})')
plt.plot([0, 1], [0, 1], 'k--')
plt.title('ROC Curve Comparison')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc='lower right')
plt.show()

# Print Summary of Results
print("Summary of Results:")
print(f"Hammerstein-Wiener Classifier Accuracy: {accuracy:.4f}")
print(f"Hammerstein-Wiener Classifier PSS: {pss:.4f}")
print(f"Hammerstein-Wiener Classifier TWA: {twa:.4f}")
print(f"LSTM Accuracy: {lstm_accuracy:.4f}")
print(f"LSTM PSS: {pss_lstm:.4f}")
print(f"LSTM TWA: {twa_lstm:.4f}")

#
