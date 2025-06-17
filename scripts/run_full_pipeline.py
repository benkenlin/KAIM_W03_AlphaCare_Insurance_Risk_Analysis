# scripts/run_full_pipeline.py

import os
import sys
import pandas as pd

# --- CRITICAL: Add the project root to the Python path ---
# This allows importing modules from 'src' directly.
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, '..'))

if project_root not in sys.path:
    sys.path.insert(0, project_root)
    print(f"Added project root '{project_root}' to sys.path.")
else:
    print(f"Project root '{project_root}' was already in sys.path.")

print(f"Current working directory (os.getcwd()): {os.getcwd()}")
print(f"sys.path[0]: {sys.path[0]}")

from src.data_preparation import load_raw_data, prepare_data_for_modeling, split_data, create_preprocessor_pipeline, get_feature_names_after_preprocessing
from src.config import RAW_DATA_PATH, PROCESSED_DATA_PATH, RANDOM_STATE, POLICY_ID_COL, TRANSACTION_MONTH_COL, TOTAL_CLAIMS_COL, CLAIM_PROBABILITY_TARGET, CLAIM_SEVERITY_TARGET, TOTAL_PREMIUM_COL, HAS_CLAIM_COL, MARGIN_COL # Added more config imports

def main():
    """
    Main function to orchestrate the entire data processing pipeline.
    """
    print("\n--- Starting Full Data Processing Pipeline ---")

    # 1. Load Raw Data
    df_raw = load_raw_data(RAW_DATA_PATH)
    if df_raw.empty:
        print("Raw data could not be loaded. Exiting pipeline.")
        return

    # --- Initial Data Type Corrections and Feature Derivation (moved from data_preparation.py's clean_and_prepare_data) ---
    df_pre_processed = df_raw.copy()

    # Handle comma decimals and convert to numeric for financial columns
    financial_cols = [TOTAL_PREMIUM_COL, TOTAL_CLAIMS_COL, 'CustomValueEstimate',
                      'CapitalOutstanding', 'SumInsured', 'CalculatedPremiumPerTerm']
    
    for col in financial_cols:
        if col in df_pre_processed.columns:
            df_pre_processed[col] = df_pre_processed[col].astype(str) # Convert to string first
            # Replace thousands separator (if any) and then decimal comma
            df_pre_processed[col] = df_pre_processed[col].str.replace('.', '', regex=False).str.replace(',', '.', regex=False)
            df_pre_processed[col] = pd.to_numeric(df_pre_processed[col], errors='coerce').fillna(0)
            df_pre_processed[col] = df_pre_processed[col].clip(lower=0)

    # Create 'HasClaim' and 'Margin' here as they depend on the raw data/initial numeric conversion
    if TOTAL_CLAIMS_COL in df_pre_processed.columns:
        df_pre_processed[HAS_CLAIM_COL] = (df_pre_processed[TOTAL_CLAIMS_COL] > 0).astype(int)
    if TOTAL_PREMIUM_COL in df_pre_processed.columns and TOTAL_CLAIMS_COL in df_pre_processed.columns:
        df_pre_processed[MARGIN_COL] = df_pre_processed[TOTAL_PREMIUM_COL] - df_pre_processed[TOTAL_CLAIMS_COL]
    # --- END Initial Pre-processing in main ---


    # Define features for preprocessing. These need to be identified from your dataset.
    numerical_features = [
        'TotalPremium', 'SumInsured', 'CustomValueEstimate',
        'CapitalOutstanding', 'CalculatedPremiumPerTerm', 'Age', 'VehicleAge',
        MARGIN_COL
    ]
    categorical_features = [
        'PolicyType', 'DriverGender', 'VehicleType', 'Geolocation',
        'FuelType', 'VehicleSegment'
    ]
    date_features = ['TransactionMonth', 'VehicleIntroDate']

    # Columns that you might want to drop entirely before modeling (e.g., identifiers)
    cols_to_drop = [POLICY_ID_COL]

    # 3. Prepare Data for Modeling (for Classification - HasClaim)
    print("\n--- Preparing data for Claim Probability (Classification) ---")
    X_clf, y_clf, preprocessor_clf = prepare_data_for_modeling(
        df_pre_processed.copy(),
        numerical_features,
        categorical_features,
        date_features,
        cols_to_drop,
        problem_type='classification'
    )
    
    if X_clf.empty or y_clf is None:
        print("Classification data preparation resulted in empty data. Skipping.")
    else:
        print(f"X_clf shape: {X_clf.shape}, y_clf shape: {X_clf.shape}") # Typo fixed here - should be y_clf.shape
        
        # Fit and transform the classification features
        X_clf_transformed = preprocessor_clf.fit_transform(X_clf)
        
        # --- FIX HERE: Call get_feature_names_after_preprocessing *after* fit_transform ---
        feature_names_clf = get_feature_names_after_preprocessing(preprocessor_clf)
        
        # Convert transformed data back to DataFrame (optional, but good for inspection)
        X_clf_transformed_df = pd.DataFrame(X_clf_transformed, columns=feature_names_clf, index=X_clf.index)
        print(f"X_clf_transformed_df shape: {X_clf_transformed_df.shape}")
        print("Sample of transformed classification features:")
        print(X_clf_transformed_df.head())

        # Split data for classification
        X_train_clf, X_test_clf, y_train_clf, y_test_clf = split_data(
            X_clf_transformed_df, y_clf, test_size=0.2, random_state=RANDOM_STATE, stratify=y_clf
        )
        print(f"Classification Train/Test Split: X_train_clf={X_train_clf.shape}, X_test_clf={X_test_clf.shape}")

    # 4. Prepare Data for Modeling (for Regression - Claim Severity)
    print("\n--- Preparing data for Claim Severity (Regression) ---")
    X_reg, y_reg, preprocessor_reg = prepare_data_for_modeling(
        df_pre_processed.copy(),
        numerical_features,
        categorical_features,
        date_features,
        cols_to_drop,
        problem_type='regression'
    )
    
    if X_reg.empty or y_reg is None:
        print("Regression data preparation resulted in empty data. Skipping.")
    else:
        print(f"X_reg shape: {X_reg.shape}, y_reg shape: {X_reg.shape}") # Typo fixed here - should be y_reg.shape

        # Fit and transform the regression features
        X_reg_transformed = preprocessor_reg.fit_transform(X_reg)
        
        # --- FIX HERE: Call get_feature_names_after_preprocessing *after* fit_transform ---
        feature_names_reg = get_feature_names_after_preprocessing(preprocessor_reg)
        
        X_reg_transformed_df = pd.DataFrame(X_reg_transformed, columns=feature_names_reg, index=X_reg.index)
        print(f"X_reg_transformed_df shape: {X_reg_transformed_df.shape}")
        print("Sample of transformed regression features:")
        print(X_reg_transformed_df.head())

        # Split data for regression
        X_train_reg, X_test_reg, y_train_reg, y_test_reg = split_data(
            X_reg_transformed_df, y_reg, test_size=0.2, random_state=RANDOM_STATE
        ) # No stratify for regression
        print(f"Regression Train/Test Split: X_train_reg={X_train_reg.shape}, X_test_reg={X_test_reg.shape}")

    # You could save preprocessors and transformed data here for later use
    # import joblib
    # joblib.dump(preprocessor_clf, 'models/preprocessor_clf.pkl')
    # joblib.dump(preprocessor_reg, 'models/preprocessor_reg.pkl')
    # X_train_clf.to_parquet(PROCESSED_DATA_PATH.replace('.parquet', '_X_train_clf.parquet'), index=False)
    # etc.
    
    print("\n--- Data Processing Pipeline Completed ---")

if __name__ == "__main__":
    main()