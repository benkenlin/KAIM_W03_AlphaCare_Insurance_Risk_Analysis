# src/models.py

import os
import pandas as pd
import numpy as np
import joblib # For saving/loading models

# Scikit-learn models
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    mean_squared_error, r2_score, classification_report, confusion_matrix
)
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

# XGBoost
import xgboost as xgb

# Import constants from config
from src.config import RANDOM_STATE, MODEL_DIR

# --- Model Building Functions ---

def build_classification_model(model_name: str, params: dict = None):
    """
    Builds and returns an initialized classification model.
    
    Args:
        model_name (str): The name of the model ('RandomForest', 'XGBoost').
        params (dict, optional): Dictionary of hyperparameters for the model.
                                 Defaults to None, in which case default params or
                                 predefined robust params are used.

    Returns:
        sklearn.base.Estimator: An initialized model instance.
    """
    if params is None:
        params = {} # Start with empty dict if no params provided

    if model_name == 'RandomForest':
        # Default/robust parameters for RandomForestClassifier
        default_params = {
            'random_state': RANDOM_STATE,
            'n_estimators': 200,
            'class_weight': 'balanced', # Important for imbalanced classification
            'n_jobs': -1 # Use all available cores
        }
        # Merge provided params over defaults
        final_params = {**default_params, **params}
        return RandomForestClassifier(**final_params)
    
    elif model_name == 'XGBoost':
        # Default/robust parameters for XGBClassifier
        default_params = {
            'random_state': RANDOM_STATE,
            'n_estimators': 200,
            'eval_metric': 'logloss', # Common metric for binary classification
            'use_label_encoder': False, # Suppress warning, not needed for binary targets
            'n_jobs': -1
        }
        # Merge provided params over defaults
        final_params = {**default_params, **params}
        return xgb.XGBClassifier(**final_params)
    
    elif model_name == 'LogisticRegression': # Adding Logistic Regression as a baseline
        from sklearn.linear_model import LogisticRegression
        default_params = {
            'random_state': RANDOM_STATE,
            'solver': 'liblinear', # Good for small datasets, and l1/l2 regularization
            'penalty': 'l2',
            'max_iter': 1000,
            'n_jobs': -1
        }
        final_params = {**default_params, **params}
        return LogisticRegression(**final_params)

    else:
        raise ValueError(f"Unknown classification model: {model_name}")

def build_regression_model(model_name: str, params: dict = None):
    """
    Builds and returns an initialized regression model.
    
    Args:
        model_name (str): The name of the model ('LinearRegression', 'RandomForest', 'XGBoost').
        params (dict, optional): Dictionary of hyperparameters for the model.

    Returns:
        sklearn.base.Estimator: An initialized model instance.
    """
    if params is None:
        params = {} # Start with empty dict if no params provided

    if model_name == 'LinearRegression':
        default_params = {
            'n_jobs': -1
        }
        final_params = {**default_params, **params}
        return LinearRegression(**final_params)
        
    elif model_name == 'RandomForest':
        default_params = {
            'random_state': RANDOM_STATE,
            'n_estimators': 200,
            'n_jobs': -1
        }
        final_params = {**default_params, **params}
        return RandomForestRegressor(**final_params)
    
    elif model_name == 'XGBoost':
        default_params = {
            'random_state': RANDOM_STATE,
            'n_estimators': 200,
            'eval_metric': 'rmse', # Root Mean Squared Error for regression
            'n_jobs': -1
        }
        final_params = {**default_params, **params}
        return xgb.XGBRegressor(**final_params)
    
    else:
        raise ValueError(f"Unknown regression model: {model_name}")


# --- Model Evaluation Functions ---

def evaluate_classification_model(y_true: pd.Series, y_pred: np.ndarray, y_proba: np.ndarray, model_name: str = "Model"):
    """
    Evaluates and prints classification metrics.

    Args:
        y_true (pd.Series): True labels.
        y_pred (np.ndarray): Predicted labels.
        y_proba (np.ndarray): Predicted probabilities for the positive class.
        model_name (str): Name of the model for printing.
    """
    print(f"\n--- {model_name} Classification Model Evaluation ---")
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    
    try:
        roc_auc = roc_auc_score(y_true, y_proba)
    except ValueError:
        roc_auc = np.nan # Handle case where only one class is present in y_true or y_proba
        print("Warning: ROC AUC cannot be calculated due to only one class present in y_true or y_proba.")

    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")
    print(f"ROC AUC: {roc_auc:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, zero_division=0))
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_true, y_pred))
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'roc_auc': roc_auc
    }

def evaluate_regression_model(y_true: pd.Series, y_pred: np.ndarray, model_name: str = "Model"):
    """
    Evaluates and prints regression metrics.

    Args:
        y_true (pd.Series): True target values.
        y_pred (np.ndarray): Predicted target values.
        model_name (str): Name of the model for printing.
    """
    print(f"\n--- {model_name} Regression Model Evaluation ---")
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)

    print(f"Mean Squared Error (MSE): {mse:.4f}")
    print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
    print(f"R-squared (R2): {r2:.4f}")
    
    return {
        'mse': mse,
        'rmse': rmse,
        'r2': r2
    }

# --- Model Saving/Loading ---

def save_model(model, filename: str):
    """Saves a trained model to the MODEL_DIR."""
    filepath = os.path.join(MODEL_DIR, filename)
    joblib.dump(model, filepath)
    print(f"Model saved to: {filepath}")

def load_model(filename: str):
    """Loads a trained model from the MODEL_DIR."""
    filepath = os.path.join(MODEL_DIR, filename)
    if os.path.exists(filepath):
        print(f"Loading model from: {filepath}")
        return joblib.load(filepath)
    else:
        print(f"Error: Model file not found at {filepath}")
        return None

# --- Hyperparameter Tuning (Optional, for later advanced tasks) ---
def tune_model(model, param_grid, X_train, y_train, cv=3, scoring='roc_auc', model_type='classification'):
    """
    Performs hyperparameter tuning using GridSearchCV or RandomizedSearchCV.
    (This is a placeholder for potential future use in advanced model development)
    """
    print(f"\n--- Starting Hyperparameter Tuning for {model.__class__.__name__} ---")
    # For simplicity, using GridSearchCV, but RandomizedSearchCV is faster for large spaces
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=cv, scoring=scoring, n_jobs=-1, verbose=2)
    grid_search.fit(X_train, y_train)

    print(f"Best parameters found: {grid_search.best_params_}")
    print(f"Best cross-validation score: {grid_search.best_score_:.4f}")

    return grid_search.best_estimator_


if __name__ == "__main__":
    print("This is the models.py module. It defines functions for model building and evaluation.")
    print("It should not be run directly. Please use scripts/training_models.py to train models.")

