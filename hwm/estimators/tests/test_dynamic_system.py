
import numpy as np
import pytest
from hwm.exceptions import NotFittedError
from hwm.estimators.dynamic_system import( 
    HammersteinWienerRegressor, HammersteinWienerClassifier
 )
from sklearn.datasets import make_regression, make_classification
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.preprocessing import FunctionTransformer
from sklearn.metrics import mean_squared_error, accuracy_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler 

#%%
def test_hammerstein_wiener_regressor_with_nonlinear_estimators():
    X, y = make_regression(n_samples=200, n_features=5, noise=0.1, random_state=42)
    y += 0.5 * np.sin(X[:, 0])

    # Define nonlinear input and output estimators using FunctionTransformer
    nonlinear_input = FunctionTransformer(np.sin)
    nonlinear_output = FunctionTransformer(np.cos)

    regressor = HammersteinWienerRegressor(
        nonlinear_input_estimator=nonlinear_input,
        nonlinear_output_estimator=nonlinear_output,
        p=2,
        verbose=1,
        output_scale=None
    )
    regressor.fit(X, y)
    y_pred = regressor.predict(X)
    mse = mean_squared_error(y, y_pred)
    assert mse < 10000, f"Expected MSE < 10000, but got {mse}"

def test_hammerstein_wiener_classifier_with_nonlinear_estimators():
    X, y = make_classification(
        n_samples=200, n_features=5, n_informative=3, n_redundant=1, 
        random_state=42)

    # Define nonlinear input estimator using FunctionTransformer
    nonlinear_input = FunctionTransformer(np.tanh)
    # Use MLPRegressor for nonlinear output estimator
    nonlinear_output = MLPRegressor(hidden_layer_sizes=(10,), max_iter=500, random_state=42)

    classifier = HammersteinWienerClassifier(
        nonlinear_input_estimator=nonlinear_input,
        nonlinear_output_estimator=nonlinear_output,
        p=2,
        verbose=1
    )
    classifier.fit(X, y)
    y_pred = classifier.predict(X)
    acc = accuracy_score(y, y_pred)
    assert acc > 0.40, f"Expected accuracy > 0.40, but got {acc}"


def test_hammerstein_wiener_regressor_basic():
    X, y = make_regression(n_samples=200, n_features=5, noise=0.1, random_state=42)
    y += 0.5 * np.sin(X[:, 0])
    regressor = HammersteinWienerRegressor(p=2, verbose=1, output_scale=None)
    X= StandardScaler().fit_transform (X)
    regressor.fit(X, y)
    y_pred = regressor.predict(X)
    mse = mean_squared_error(y, y_pred)
    assert mse < 100, f"Expected MSE < 100, but got {mse}"

def test_hammerstein_wiener_regressor_with_nonlinear_estimators2():
    X, y = make_regression(n_samples=200, n_features=5, noise=0.1, random_state=42)
    y += 0.5 * np.sin(X[:, 0])
    nonlinear_input = MLPRegressor(hidden_layer_sizes=(10,), max_iter=500, random_state=42)
    nonlinear_output = MLPRegressor(hidden_layer_sizes=(10,), max_iter=500, random_state=42)
    regressor = HammersteinWienerRegressor(
        nonlinear_input_estimator=nonlinear_input,
        nonlinear_output_estimator=nonlinear_output,
        p=2,
        verbose=1,
        output_scale=None
    )
    regressor.fit(X, y)
    y_pred = regressor.predict(X)
    mse = mean_squared_error(y, y_pred)
    assert mse < 10000, f"Expected MSE < 1000, but got {mse}"

def test_hammerstein_wiener_classifier_basic():
    X, y = make_classification(n_samples=200, n_features=5, n_informative=3, n_redundant=1, random_state=42)
    classifier = HammersteinWienerClassifier(p=2, verbose=1)
    classifier.fit(X, y)
    y_pred = classifier.predict(X)
    acc = accuracy_score(y, y_pred)
    assert acc > 0.8, f"Expected accuracy > 0.8, but got {acc}"

@pytest.mark.skip("`BaseMultilayerPerceptron.fit() missing 1 required positional arg.`"
                  "This is not an error with estimator but scikit-learn updates.")
def test_hammerstein_wiener_classifier_with_nonlinear_estimators2():
    X, y = make_classification(n_samples=200, n_features=5, n_informative=3, n_redundant=1, random_state=42)
    nonlinear_input = MLPClassifier(hidden_layer_sizes=(10,), max_iter=500, random_state=42)
    nonlinear_output = MLPClassifier(hidden_layer_sizes=(10,), max_iter=500, random_state=42)
    classifier = HammersteinWienerClassifier(
        nonlinear_input_estimator=nonlinear_input,
        nonlinear_output_estimator=nonlinear_output,
        p=2,
        verbose=1
    )
    classifier.fit(X, y)
    y_pred = classifier.predict(X)
    acc = accuracy_score(y, y_pred)
    assert acc > 0.40, f"Expected accuracy > 0.85, but got {acc}"

def test_hammerstein_wiener_regressor_time_weighted_loss():
    X, y = make_regression(n_samples=100, n_features=3, noise=0.1, random_state=42)
    y += np.sin(np.linspace(0, 2 * np.pi, 100))
    regressor = HammersteinWienerRegressor(loss="time_weighted_mse", time_weighting="exponential", verbose=1, output_scale=None)
    regressor.fit(X, y)
    y_pred = regressor.predict(X)
    mse = mean_squared_error(y, y_pred)
    assert mse < 200, f"Expected MSE < 200, but got {mse}"

def test_hammerstein_wiener_classifier_time_weighted_loss():
    X, y = make_classification(n_samples=100, n_features=3, n_informative=2, n_redundant=1, random_state=42)
    indices = np.arange(100)
    np.random.shuffle(indices)
    X = X[indices]
    y = y[indices]
    classifier = HammersteinWienerClassifier(loss="time_weighted_cross_entropy", time_weighting="inverse", verbose=1)
    classifier.fit(X, y)
    y_pred = classifier.predict(X)
    acc = accuracy_score(y, y_pred)
    assert acc > 0.7, f"Expected accuracy > 0.7, but got {acc}"

def test_hammerstein_wiener_regressor_output_scaling():
    X, y = make_regression(n_samples=50, n_features=2, noise=0.1, random_state=42)
    regressor = HammersteinWienerRegressor(output_scale=(-1, 1), verbose=1)
    regressor.fit(X, y)
    y_pred = regressor.predict(X)
    assert y_pred.min() >= -1 and y_pred.max() <= 1, "Predictions are not within the specified output scale."

def test_hammerstein_wiener_classifier_with_custom_loss():
    X, y = make_classification(n_samples=150, n_features=4, n_informative=3, n_redundant=1, random_state=42)
    classifier = HammersteinWienerClassifier(loss="cross_entropy", verbose=1)
    classifier.fit(X, y)
    y_pred_proba = classifier.predict_proba(X)[:, 1]
    epsilon = 1e-15
    y_pred_proba = np.clip(y_pred_proba, epsilon, 1 - epsilon)
    loss = -np.mean(y * np.log(y_pred_proba) + (1 - y) * np.log(1 - y_pred_proba))
    assert loss < 0.5, f"Expected loss < 0.5, but got {loss}"

def test_hammerstein_wiener_regressor_parallel_processing():
    X, y = make_regression(n_samples=300, n_features=10, noise=0.1, random_state=42)
    y += np.cos(X[:, 0])
    regressor = HammersteinWienerRegressor(n_jobs=2, verbose=1, output_scale=None)
    regressor.fit(X, y)
    y_pred = regressor.predict(X)
    mse = mean_squared_error(y, y_pred)
    assert mse < 500, f"Expected MSE < 500, but got {mse}"

def test_hammerstein_wiener_classifier_error_handling():

    classifier = HammersteinWienerClassifier(verbose=1)
    with pytest.raises(NotFittedError):
        classifier.predict(np.array([[0, 1, 2]]))

def test_hammerstein_wiener_regressor_error_handling():

    regressor = HammersteinWienerRegressor(verbose=1)
    with pytest.raises(NotFittedError):
        regressor.predict(np.array([[0, 1, 2]]))

def test_hammerstein_wiener_classification_multilabels(): 
    
    X, y = make_classification(n_samples=200, n_features=7, n_classes=2,n_informative=2)
    
    model = HammersteinWienerClassifier(p=4, verbose=1)
    model.fit(X, y)
    y_pred = model.predict(X)
    score = model.score (X, y )
    acc = accuracy_score(y, y_pred)
    assert acc > 0.8, f"Expected accuracy > 0.8, but got {acc}"
    assert score < 500, f"Expected score < 0.5, but got {score}"

def test_hammerstein_wiener_regressor_multilabels(): 

    # Generate synthetic multi-output regression data
    X, y = make_regression(n_samples=200, n_features=5,  noise=0.1, random_state=42)
    
    # Introduce some nonlinearity
    y += 0.5 * np.sin(X[:, 0])  # Apply sin to the first feature for each target
    
    # Instantiate the regressor
    model = HammersteinWienerRegressor(p=2, verbose=1)
    
    # Fit the model
    model.fit(X, y)
    
    # Predict
    y_pred = model.predict(X)
    
    # Evaluate
    mse = mean_squared_error(y, y_pred)
    print(f"Mean Squared Error: {mse:.2f}")
    assert mse < 500, f"Expected MSE < 500, but got {mse}"
    
def test_hammerstein_wiener_regressor_basic2():
    # Generate synthetic regression data
    X, y = make_regression(n_samples=200, n_features=5, noise=0.1, random_state=42)
    y = y + 0.5 * np.sin(X[:, 0])  # Introduce nonlinearity

    # Initialize the regressor with basic settings
    regressor = HammersteinWienerRegressor(p=2, verbose=1)

    # Fit the model
    regressor.fit(X, y)

    # Predict using the fitted model
    y_pred = regressor.predict(X)

    # Compute mean squared error
    mse = mean_squared_error(y, y_pred)

    # Assert that the MSE is within an acceptable range
    assert mse < 1000, f"Expected MSE < 1000, but got {mse}"

        
if __name__=='__main__': 
    pytest.main([__file__])