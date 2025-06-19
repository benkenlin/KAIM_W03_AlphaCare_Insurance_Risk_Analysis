# scripts/training_models.py

import os
import sys
import pandas as pd
import numpy as np
import joblib
import shap # For model interpretability
import matplotlib.pyplot as plt # For SHAP plots
import warnings

# Suppress specific warnings from scikit-learn or other libraries
warnings.filterwarnings("ignore", category=UserWarning, module='sklearn')
warnings.filterwarnings("ignore", category=FutureWarning)

# --- Add the project root to the Python path ---
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, '..'))

if project_root not in sys.path:
    sys.path.insert(0, project_root)
    print(f"Added project root '{project_root}' to sys.path.")
else:
    print(f"Project root '{project_root}' was already in sys.path.")

# Import modules from src
from src.data_preparation import load_raw_data, prepare_data_for_modeling, split_data, get_feature_names_after_preprocessing
from src.models import build_classification_model, build_regression_model, evaluate_classification_model, evaluate_regression_model, save_model
from src.config import (
    RAW_DATA_PATH, RANDOM_STATE, MODEL_DIR,
    NUMERICAL_FEATURES_CLASSIFICATION, CATEGORICAL_FEATURES_CLASSIFICATION, DATE_FEATURES_CLASSIFICATION, COLS_TO_DROP_CLASSIFICATION,
    NUMERICAL_FEATURES_REGRESSION, CATEGORICAL_FEATURES_REGRESSION, DATE_FEATURES_REGRESSION, COLS_TO_DROP_REGRESSION
)


def run_shap_analysis(model, X_data, feature_names, title="SHAP Summary Plot"):
    """Runs SHAP analysis and generates a summary plot."""
    print(f"\n--- Running SHAP Analysis for {title} ---")
    try:
        # For tree-based models, use TreeExplainer
        if "XGB" in model.__class__.__name__ or "RandomForest" in model.__class__.__name__:
            explainer = shap.TreeExplainer(model)
        else:
            # Fallback for other model types, or if TreeExplainer fails
            explainer = shap.KernelExplainer(model.predict_proba if hasattr(model, 'predict_proba') else model.predict, X_data)
        
        shap_values = explainer.shap_values(X_data)

        # For classification, shap_values might be a list (one array per class). Take the positive class.
        if isinstance(shap_values, list):
            # For binary classification, typically class 1 (positive outcome) is of interest
            # Ensure the correct array is picked for shap_values (e.g., shap_values[1] for binary class 1)
            if len(shap_values) == 2: # Binary classification
                shap_values_to_plot = shap_values[1]
            else: # Multi-class or unusual case, default to first class
                shap_values_to_plot = shap_values[0] 
        else:
            shap_values_to_plot = shap_values

        # Ensure X_data is a DataFrame with feature names for plotting
        if not isinstance(X_data, pd.DataFrame):
            X_data_df = pd.DataFrame(X_data, columns=feature_names)
        else:
            X_data_df = X_data

        plt.figure(figsize=(10, 6))
        shap.summary_plot(shap_values_to_plot, X_data_df, plot_type="bar", show=False, max_display=10, color='skyblue')
        plt.title(f"SHAP Feature Importance: {title}")
        plt.tight_layout()
        plt.show()

        plt.figure(figsize=(10, 6))
        shap.summary_plot(shap_values_to_plot, X_data_df, show=False, max_display=10) # Dot plot
        plt.title(f"SHAP Summary (Impact & Direction): {title}")
        plt.tight_layout()
        plt.show()

    except Exception as e:
        print(f"Error during SHAP analysis for {title}: {e}")
        print("SHAP analysis might require model-specific explainers or data formats.")


def main():
    """
    Main function to orchestrate the entire data processing and model training pipeline.
    """
    print("\n--- Starting Model Training Pipeline ---")

    # 1. Load Raw Data
    df_raw = load_raw_data(RAW_DATA_PATH)
    if df_raw.empty:
        print("Raw data could not be loaded. Exiting pipeline.")
        return

    # --- PART 1: CLAIM PROBABILITY (CLASSIFICATION) ---
    print("\n--- Preparing data for Claim Probability (Classification) ---")
    X_clf_prepared, y_clf, preprocessor_clf = prepare_data_for_modeling(
        df_raw.copy(), # Pass raw data for internal preprocessing
        NUMERICAL_FEATURES_CLASSIFICATION,
        CATEGORICAL_FEATURES_CLASSIFICATION,
        DATE_FEATURES_CLASSIFICATION,
        COLS_TO_DROP_CLASSIFICATION,
        problem_type='classification'
    )
    
    if X_clf_prepared.empty or y_clf is None or y_clf.empty:
        print("Classification data preparation resulted in empty data. Skipping classification models.")
    else:
        print(f"X_clf_prepared shape: {X_clf_prepared.shape}, y_clf shape: {y_clf.shape}")
        
        # Fit and transform the classification features using the preprocessor
        X_clf_transformed = preprocessor_clf.fit_transform(X_clf_prepared)
        
        # Get feature names after preprocessing using the now fitted preprocessor
        feature_names_clf = get_feature_names_after_preprocessing(preprocessor_clf)
        
        # Convert transformed data back to DataFrame for better usability (essential for SHAP)
        X_clf_transformed_df = pd.DataFrame(X_clf_transformed, columns=feature_names_clf, index=X_clf_prepared.index)
        print(f"X_clf_transformed_df shape: {X_clf_transformed_df.shape}")
        print("Sample of transformed classification features:")
        print(X_clf_transformed_df.head())

        # Split data for classification (using the preprocessed data)
        X_train_clf, X_test_clf, y_train_clf, y_test_clf = split_data(
            X_clf_transformed_df, y_clf, test_size=0.2, random_state=RANDOM_STATE, stratify_y=True
        )
        print(f"Classification Train/Test Split: X_train_clf={X_train_clf.shape}, X_test_clf={X_test_clf.shape}")

        # --- Training Classification Models ---
        print("\n--- Training Classification Models ---")
        
        # RandomForestClassifier
        print("\n--- Training RandomForestClassifier ---")
        rf_clf_model = build_classification_model('RandomForest')
        rf_clf_model.fit(X_train_clf, y_train_clf)
        print("RandomForestClassifier training complete.")
        y_rf_clf_pred = rf_clf_model.predict(X_test_clf)
        y_rf_clf_proba = rf_clf_model.predict_proba(X_test_clf)[:, 1]
        evaluate_classification_model(y_test_clf, y_rf_clf_pred, y_rf_clf_proba, "RandomForestClassifier")
        
        # XGBClassifier
        print("\n--- Training XGBClassifier ---")
        xgb_clf_model = build_classification_model('XGBoost')
        xgb_clf_model.fit(X_train_clf, y_train_clf)
        print("XGBClassifier training complete.")
        y_xgb_clf_pred = xgb_clf_model.predict(X_test_clf)
        y_xgb_clf_proba = xgb_clf_model.predict_proba(X_test_clf)[:, 1]
        evaluate_classification_model(y_test_clf, y_xgb_clf_pred, y_xgb_clf_proba, "XGBClassifier")

        # --- Model Interpretability: SHAP for XGBClassifier ---
        run_shap_analysis(xgb_clf_model, X_test_clf, feature_names_clf, "Claim Probability XGBoost Model")

        # --- Save Classification Models and Preprocessor ---
        save_model(rf_clf_model, 'rf_classification_model.joblib')
        save_model(xgb_clf_model, 'xgb_classification_model.joblib')
        save_model(preprocessor_clf, 'preprocessor_clf.joblib')
        print(f"\nClassification models and preprocessor saved to: {MODEL_DIR}")


    # --- PART 2: CLAIM SEVERITY (REGRESSION) ---
    print("\n--- Preparing data for Claim Severity (Regression) ---")
    X_reg_prepared, y_reg, preprocessor_reg = prepare_data_for_modeling(
        df_raw.copy(), # Use a fresh copy of the raw data
        NUMERICAL_FEATURES_REGRESSION,
        CATEGORICAL_FEATURES_REGRESSION,
        DATE_FEATURES_REGRESSION,
        COLS_TO_DROP_REGRESSION,
        problem_type='regression' # Filtered for claims > 0 internally
    )
    
    if X_reg_prepared.empty or y_reg is None or y_reg.empty:
        print("Regression data preparation resulted in empty data or no claims. Skipping regression models.")
    else:
        print(f"X_reg_prepared shape: {X_reg_prepared.shape}, y_reg shape: {y_reg.shape}")

        # Fit and transform the regression features
        X_reg_transformed = preprocessor_reg.fit_transform(X_reg_prepared)
        
        # Get feature names for regression
        feature_names_reg = get_feature_names_after_preprocessing(preprocessor_reg)
        
        X_reg_transformed_df = pd.DataFrame(X_reg_transformed, columns=feature_names_reg, index=X_reg_prepared.index)
        print(f"X_reg_transformed_df shape: {X_reg_transformed_df.shape}")
        print("Sample of transformed regression features:")
        print(X_reg_transformed_df.head())

        # Split data for regression
        X_train_reg, X_test_reg, y_train_reg, y_test_reg = split_data(
            X_reg_transformed_df, y_reg, test_size=0.2, random_state=RANDOM_STATE, stratify_y=False # No stratification for regression
        )
        print(f"Regression Train/Test Split: X_train_reg={X_train_reg.shape}, X_test_reg={X_test_reg.shape}")

        # --- Training Regression Models ---
        print("\n--- Training Regression Models ---")

        # Linear Regression
        print("\n--- Training LinearRegression ---")
        lr_reg_model = build_regression_model('LinearRegression')
        lr_reg_model.fit(X_train_reg, y_train_reg)
        print("LinearRegression training complete.")
        y_lr_reg_pred = lr_reg_model.predict(X_test_reg)
        evaluate_regression_model(y_test_reg, y_lr_reg_pred, "LinearRegression")

        # RandomForestRegressor
        print("\n--- Training RandomForestRegressor ---")
        rf_reg_model = build_regression_model('RandomForest')
        rf_reg_model.fit(X_train_reg, y_train_reg)
        print("RandomForestRegressor training complete.")
        y_rf_reg_pred = rf_reg_model.predict(X_test_reg)
        evaluate_regression_model(y_test_reg, y_rf_reg_pred, "RandomForestRegressor")

        # XGBRegressor
        print("\n--- Training XGBRegressor ---")
        xgb_reg_model = build_regression_model('XGBoost')
        xgb_reg_model.fit(X_train_reg, y_train_reg)
        print("XGBRegressor training complete.")
        y_xgb_reg_pred = xgb_reg_model.predict(X_test_reg)
        evaluate_regression_model(y_test_reg, y_xgb_reg_pred, "XGBRegressor")

        # --- Model Interpretability: SHAP for XGBRegressor ---
        run_shap_analysis(xgb_reg_model, X_test_reg, feature_names_reg, "Claim Severity XGBoost Model")

        # --- Save Regression Models and Preprocessor ---
        save_model(lr_reg_model, 'lr_regression_model.joblib')
        save_model(rf_reg_model, 'rf_regression_model.joblib')
        save_model(xgb_reg_model, 'xgb_regression_model.joblib')
        save_model(preprocessor_reg, 'preprocessor_reg.joblib')
        print(f"\nRegression models and preprocessor saved to: {MODEL_DIR}")


    print("\n--- Full Model Training Pipeline Execution Complete ---")

if __name__ == "__main__":
    main()

