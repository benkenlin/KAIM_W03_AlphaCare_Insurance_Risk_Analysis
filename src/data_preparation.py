# src/data_preparation.py

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.base import clone

# Import engineer_features from the new feature_engineering module
from src.feature_engineering import engineer_features

# Imports from config
from src.config import (
    RANDOM_STATE, CLAIM_PROBABILITY_TARGET, CLAIM_SEVERITY_TARGET, TOTAL_CLAIMS_COL,
    RAW_DATA_PATH, POLICY_ID_COL, TRANSACTION_MONTH_COL, TOTAL_PREMIUM_COL,
    HAS_CLAIM_COL, MARGIN_COL, TEST_SIZE
)

def load_raw_data(file_path: str) -> pd.DataFrame:
    """
    Loads raw data from a specified text file.
    Assumes '|' as the delimiter.
    """
    try:
        df = pd.read_csv(file_path, delimiter='|')
        print(f"Raw data loaded successfully from {file_path}. Shape: {df.shape}")
        df.columns = df.columns.str.strip()
        return df
    except FileNotFoundError:
        print(f"Error: The file {file_path} was not found.")
        return pd.DataFrame()
    except Exception as e:
        print(f"An error occurred while loading data: {e}")
        return pd.DataFrame()

def _convert_comma_to_dot_and_numeric(df: pd.DataFrame, cols: list) -> pd.DataFrame:
    """
    Converts specified columns to numeric (float).
    Handles empty strings/whitespace to NaN.
    """
    df_copy = df.copy()
    for col in cols:
        if col in df_copy.columns:
            if pd.api.types.is_numeric_dtype(df_copy[col]):
                continue
            
            df_copy[col] = df_copy[col].astype(str).replace(r'^\s*$', np.nan, regex=True)
            df_copy[col] = pd.to_numeric(df_copy[col], errors='coerce')
    return df_copy

def handle_missing_data(df: pd.DataFrame, numerical_cols: list, categorical_cols: list) -> pd.DataFrame:
    """
    Handles missing data by imputing numerical columns with their median
    and categorical columns with their mode.
    """
    df_processed = df.copy()

    for col in numerical_cols:
        if col in df_processed.columns:
            df_processed[col] = pd.to_numeric(df_processed[col], errors='coerce')
            if df_processed[col].isnull().any():
                median_value = df_processed[col].median()
                if pd.isna(median_value):
                    median_value = 0.0
                df_processed[col] = df_processed[col].fillna(median_value)
            
            numerical_cols_to_clip = [
                TOTAL_PREMIUM_COL, TOTAL_CLAIMS_COL, 'CustomValueEstimate', 'CapitalOutstanding',
                'SumInsured', 'CalculatedPremiumPerTerm', MARGIN_COL, 'Age', 'VehicleAge',
                'RegistrationYear', 'Cylinders', 'cubiccapacity', 'kilowatts',
                'NumberOfDoors', 'NumberOfVehiclesInFleet'
            ]
            if col in numerical_cols_to_clip:
                df_processed[col] = df_processed[col].clip(lower=0)
            
            if col in ['Age', 'VehicleAge', 'RegistrationYear', 'Cylinders', 'NumberOfDoors', 'NumberOfVehiclesInFleet']:
                if df_processed[col].isnull().sum() == 0 and (df_processed[col] == df_processed[col].astype(int)).all():
                     df_processed[col] = df_processed[col].astype(int)

    for col in categorical_cols:
        if col in df_processed.columns:
            df_processed[col] = df_processed[col].astype(str)
            df_processed[col] = df_processed[col].replace(r'^\s*$', np.nan, regex=True)
            df_processed[col] = df_processed[col].replace('nan', np.nan, regex=False)

            if df_processed[col].isnull().any():
                mode_value = df_processed[col].mode()
                if not mode_value.empty:
                    df_processed[col] = df_processed[col].fillna(mode_value[0])
                else:
                    df_processed[col] = df_processed[col].fillna('Unknown')
            
            df_processed[col] = df_processed[col].astype(str).str.strip()
    return df_processed

def handle_high_cardinality_categorical(df: pd.DataFrame, categorical_cols: list, max_categories: int = 50) -> pd.DataFrame:
    """
    Reduces cardinality of categorical features by grouping infrequent categories into 'Other'.
    
    Args:
        df (pd.DataFrame): The input DataFrame.
        categorical_cols (list): List of categorical columns to process.
        max_categories (int): Maximum number of unique categories allowed before grouping.
                              Any categories beyond this count will be grouped into 'Other'.

    Returns:
        pd.DataFrame: DataFrame with reduced cardinality categorical features.
    """
    df_reduced = df.copy()
    print(f"\n--- Handling High Cardinality Categorical Features (max_categories={max_categories}) ---")
    for col in categorical_cols:
        if col in df_reduced.columns:
            # Ensure column is string type for consistent operations
            df_reduced[col] = df_reduced[col].astype(str)
            
            unique_counts = df_reduced[col].value_counts()
            
            if unique_counts.nunique() > max_categories:
                print(f"  Column '{col}' has {unique_counts.nunique()} unique categories (reducing to {max_categories}).")
                # Identify categories to keep (top N most frequent)
                categories_to_keep = unique_counts.nlargest(max_categories - 1).index # -1 to leave room for 'Other'
                
                # Replace categories not in 'categories_to_keep' with 'Other'
                df_reduced[col] = np.where(
                    df_reduced[col].isin(categories_to_keep),
                    df_reduced[col],
                    'Other'
                )
                print(f"  Cardinality of '{col}' reduced to {df_reduced[col].nunique()} after grouping.")
            else:
                print(f"  Column '{col}' has {unique_counts.nunique()} unique categories (no reduction needed).")
        else:
            print(f"  Warning: Categorical column '{col}' not found for cardinality handling.")
    print("--------------------------------------------------------------------------")
    return df_reduced


def create_preprocessor_pipeline(numerical_features: list, categorical_features: list) -> ColumnTransformer:
    """
    Creates a ColumnTransformer for preprocessing numerical and categorical features.
    IMPORTANT: Imputation and cardinality reduction are handled BEFORE this stage.
    This only does scaling and one-hot encoding.
    """
    transformers_list = []

    if numerical_features:
        numerical_transformer = Pipeline(steps=[
            ('scaler', StandardScaler())
        ])
        transformers_list.append(('num', numerical_transformer, numerical_features))
    else:
        print("Info: No numerical features provided for preprocessor.")

    if categorical_features:
        categorical_transformer = Pipeline(steps=[
            # Removed 'verbose=True' as it's deprecated/removed in newer sklearn versions.
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ])
        transformers_list.append(('cat', categorical_transformer, categorical_features))
    else:
        print("Info: No categorical features provided for preprocessor.")

    if not transformers_list:
        raise ValueError("No transformers created. Check numerical and categorical feature lists passed to preprocessor.")

    preprocessor = ColumnTransformer(
        transformers=transformers_list,
        remainder='drop'
    )
    return preprocessor

def get_feature_names_after_preprocessing(preprocessor: ColumnTransformer) -> list:
    """
    Retrieves the feature names after preprocessing by a ColumnTransformer.
    """
    return list(preprocessor.get_feature_names_out())


def prepare_data_for_modeling(
    df: pd.DataFrame,
    numerical_cols_initial: list,
    categorical_cols_initial: list,
    date_cols_initial: list,
    cols_to_drop_initial: list,
    problem_type: str = 'classification',
    max_categories_for_ohe: int = 50 # New parameter for cardinality control
) -> tuple[pd.DataFrame, pd.Series, ColumnTransformer]:
    """
    Prepares data for machine learning modeling:
    1. Centralized comma-to-dot conversion and initial numeric coercion.
    2. Drops specified columns early.
    3. Engineers new features.
    4. Handles missing data (imputation and robust type conversion).
    5. Handles high cardinality categorical features.
    6. Splits into features (X) and target (y).
    7. Filters data for regression task (only rows with claims).
    8. Creates an UNFITTED ColumnTransformer for preprocessing (scaling, one-hot encoding only).

    Args:
        df (pd.DataFrame): The raw input DataFrame.
        numerical_cols_initial (list): List of initial numerical column names.
        categorical_cols_initial (list): List of initial categorical column names.
        date_cols_initial (list): List of initial date column names.
        cols_to_drop_initial (list): List of column names to drop.
        problem_type (str): 'classification' or 'regression'.
        max_categories_for_ohe (int): Max unique categories for OHE before grouping to 'Other'.

    Returns:
        tuple[pd.DataFrame, pd.Series, ColumnTransformer]: X (features), y (target), and the unfitted preprocessor.
    """
    df_processed = df.copy()

    # --- Step 0: Initial Column Cleaning and Type Conversion ---
    df_processed.columns = df_processed.columns.str.strip()

    all_potential_numeric_cols_for_conversion = list(set(
        numerical_cols_initial +
        [TOTAL_CLAIMS_COL, TOTAL_PREMIUM_COL, MARGIN_COL, 'CustomValueEstimate',
         'CapitalOutstanding', 'SumInsured', 'CalculatedPremiumPerTerm',
         'RegistrationYear', 'Cylinders', 'cubiccapacity', 'kilowatts',
         'NumberOfDoors', 'NumberOfVehiclesInFleet'
        ]
    ))
    all_potential_numeric_cols_for_conversion = [col for col in all_potential_numeric_cols_for_conversion if col in df_processed.columns]

    df_processed = _convert_comma_to_dot_and_numeric(df_processed, all_potential_numeric_cols_for_conversion)
    print("Initial numeric coercion (empty string/whitespace to NaN, then to_numeric) complete.")

    if TOTAL_CLAIMS_COL in df_processed.columns:
        df_processed[HAS_CLAIM_COL] = (pd.to_numeric(df_processed[TOTAL_CLAIMS_COL], errors='coerce') > 0).astype(int)
    else:
        df_processed[HAS_CLAIM_COL] = 0

    if TOTAL_PREMIUM_COL in df_processed.columns and TOTAL_CLAIMS_COL in df_processed.columns:
        df_processed[TOTAL_PREMIUM_COL] = pd.to_numeric(df_processed[TOTAL_PREMIUM_COL], errors='coerce')
        df_processed[TOTAL_CLAIMS_COL] = pd.to_numeric(df_processed[TOTAL_CLAIMS_COL], errors='coerce')
        df_processed[MARGIN_COL] = df_processed[TOTAL_PREMIUM_COL].fillna(0) - df_processed[TOTAL_CLAIMS_COL].fillna(0)
    else:
        df_processed[MARGIN_COL] = 0.0

    for col in all_potential_numeric_cols_for_conversion:
        if col in df_processed.columns and pd.api.types.is_numeric_dtype(df_processed[col]):
            df_processed[col] = df_processed[col].clip(lower=0)

    # --- Step 1: Drop specified columns early (e.g., IDs) ---
    final_cols_to_drop = [col for col in cols_to_drop_initial if col in df_processed.columns]
    df_processed = df_processed.drop(columns=final_cols_to_drop, errors='ignore')
    print(f"Dropped initial columns: {final_cols_to_drop}.")

    # --- Step 2: Feature Engineering ---
    df_features_engineered = engineer_features(df_processed, date_cols_initial)
    print("Feature engineering complete using src/feature_engineering.py.")

    # --- Step 3: Define Target and Features (X, y) ---
    y = None
    X = df_features_engineered.copy()

    if problem_type == 'classification':
        if HAS_CLAIM_COL not in X.columns:
             raise ValueError(f"'{HAS_CLAIM_COL}' not found for classification target. Check data and config.")
        y = X[HAS_CLAIM_COL]
        X = X.drop(columns=[HAS_CLAIM_COL, TOTAL_CLAIMS_COL, CLAIM_SEVERITY_TARGET], errors='ignore')
        print(f"Prepared data for Classification: Target '{CLAIM_PROBABILITY_TARGET}'.")

    elif problem_type == 'regression':
        if TOTAL_CLAIMS_COL not in X.columns:
            raise ValueError(f"'{TOTAL_CLAIMS_COL}' not found for regression target. Check data and config.")
        
        X[TOTAL_CLAIMS_COL] = pd.to_numeric(X[TOTAL_CLAIMS_COL], errors='coerce').fillna(0)
        
        X_reg_filtered = X[X[TOTAL_CLAIMS_COL] > 0].copy()
        
        if X_reg_filtered.empty:
            print("Warning: No claims found for regression task. Returning empty data.")
            return pd.DataFrame(), pd.Series(dtype=float), create_preprocessor_pipeline([],[])
            
        y = X_reg_filtered[TOTAL_CLAIMS_COL]
        X = X_reg_filtered.drop(columns=[TOTAL_CLAIMS_COL, HAS_CLAIM_COL, CLAIM_PROBABILITY_TARGET, CLAIM_SEVERITY_TARGET], errors='ignore')
        print(f"Prepared data for Regression: Target '{CLAIM_SEVERITY_TARGET}'. Filtered for claims > 0.")

    else:
        raise ValueError("problem_type must be 'classification' or 'regression'.")

    # --- Step 4: Refine Feature Lists for Preprocessor based on actual X columns ---
    final_numerical_features = [col for col in numerical_cols_initial if col in X.columns]
    final_categorical_features = [col for col in categorical_cols_initial if col in X.columns]

    engineered_num_features_to_check = ['VehicleAge_at_Transaction', 'PremiumPerSumInsured']
    for date_col in date_cols_initial:
        engineered_num_features_to_check.extend([f'{date_col}_Year', f'{date_col}_Month'])

    for eng_col in engineered_num_features_to_check:
        if eng_col in X.columns:
            if eng_col not in final_numerical_features:
                final_numerical_features.append(eng_col)

    for date_col in date_cols_initial:
        if date_col in X.columns:
            if (f'{date_col}_Year' in X.columns and f'{date_col}_Year' in final_numerical_features) or \
               (f'{date_col}_Month' in X.columns and f'{date_col}_Month' in final_numerical_features):
                X = X.drop(columns=[date_col], errors='ignore')

    final_numerical_features = sorted(list(set(final_numerical_features)))
    final_categorical_features = sorted(list(set(final_categorical_features)))

    all_final_features = final_numerical_features + final_categorical_features
    missing_in_X = [col for col in all_final_features if col not in X.columns]
    if missing_in_X:
        print(f"Warning: The following intended features are not found in X after initial processing: {missing_in_X}. They will be excluded from modeling.")
        final_numerical_features = [f for f in final_numerical_features if f not in missing_in_X]
        final_categorical_features = [f for f in final_categorical_features if f not in missing_in_X]
        all_final_features = final_numerical_features + final_categorical_features

    X = X[all_final_features].copy()
    print(f"X filtered to only include {len(all_final_features)} intended features.")

    # --- Step 5: Handle Missing Data (Imputation) ---
    X = handle_missing_data(X, final_numerical_features, final_categorical_features)
    print("Final missing data handling (imputation) complete.")

    # --- Step 6: Handle High Cardinality Categorical Features ---
    # Apply cardinality reduction *before* passing to ColumnTransformer for OHE
    X = handle_high_cardinality_categorical(X, final_categorical_features, max_categories=max_categories_for_ohe)
    print("High cardinality categorical feature handling complete.")

    # --- FINAL CRITICAL DTYPE CHECK AND ENFORCEMENT BEFORE PREPROCESSOR CREATION ---
    print("\n--- FINAL CRITICAL DTYPE CHECK AND ENFORCEMENT FOR NUMERICAL FEATURES ---")
    problem_cols_found_num_final = False
    for col in final_numerical_features:
        if col in X.columns:
            original_dtype = X[col].dtype
            X[col] = pd.to_numeric(X[col], errors='coerce')
            if not pd.api.types.is_numeric_dtype(X[col]):
                print(f"  ERROR: Column '{col}' is NOT NUMERIC ({X[col].dtype}) even after final force-to-numeric! This will likely cause a model error.")
                print(f"    Sample values: {X[col].head(10).tolist()}")
                problem_cols_found_num_final = True
            elif X[col].isnull().any():
                print(f"  WARNING: Column '{col}' still has NaNs ({X[col].isnull().sum()}) after final imputation/coercion!")
                median_val = X[col].median() if not pd.isna(X[col].median()) else 0.0
                X[col] = X[col].fillna(median_val)
                print(f"    Filled remaining NaNs with median ({median_val}).")
                if X[col].isnull().any():
                    print(f"    CRITICAL: Column '{col}' still has NaNs after all attempts.")
                    problem_cols_found_num_final = True
            if original_dtype != X[col].dtype and pd.api.types.is_numeric_dtype(X[col]):
                print(f"  Forced '{col}' from {original_dtype} to {X[col].dtype}.")
        else:
            print(f"  WARNING: Numerical feature '{col}' not found in X at final check.")
    if not problem_cols_found_num_final:
        print("  All numerical features are numeric and have no NaNs before preprocessor creation.")
    print("------------------------------------------------------------------\n")

    print("\n--- FINAL CRITICAL DTYPE CHECK AND ENFORCEMENT FOR CATEGORICAL FEATURES ---")
    problem_cat_cols_found_final = False
    for col in final_categorical_features:
        if col in X.columns:
            original_dtype = X[col].dtype
            X[col] = X[col].astype(str)
            X[col] = X[col].replace(r'^\s*$', np.nan, regex=True)
            X[col] = X[col].replace('nan', np.nan, regex=False)

            if X[col].isnull().any():
                print(f"  WARNING: Column '{col}' has NaNs ({X[col].isnull().sum()}) after final string conversion.")
                mode_val = X[col].mode()
                if not mode_val.empty:
                    X[col] = X[col].fillna(mode_val[0])
                    print(f"    Filled NaNs in '{col}' with mode: '{mode_val[0]}'")
                else:
                    X[col] = X[col].fillna('Unknown')
                    print(f"    Filled NaNs in '{col}' with 'Unknown' (no mode found).")
            
            if '' in X[col].unique():
                print(f"  ERROR: Column '{col}' still contains empty strings after all attempts! This might cause OHE issues.")
                problem_cat_cols_found_final = True
            
            if original_dtype != X[col].dtype and (pd.api.types.is_string_dtype(X[col]) or pd.api.types.is_object_dtype(X[col])):
                print(f"  Forced '{col}' from {original_dtype} to {X[col].dtype}.")

            if not (pd.api.types.is_string_dtype(X[col]) or pd.api.types.is_object_dtype(X[col])):
                 print(f"  ERROR: Column '{col}' is NOT OBJECT/STRING ({X[col].dtype}) even after final force-to-string!")
                 problem_cat_cols_found_final = True
        else:
            print(f"  WARNING: Categorical feature '{col}' not found in X at final check.")

    if not problem_cat_cols_found_final:
        print("  All categorical features are object/string and have no NaNs or empty strings before preprocessor creation.")
    print("------------------------------------------------------------------\n")

    # --- Step 7: Create Preprocessor Pipeline ---
    preprocessor = create_preprocessor_pipeline(final_numerical_features, final_categorical_features)
    print("Preprocessor pipeline (without imputer) created successfully.")

    print(f"Final X shape: {X.shape}, y shape: {y.shape}")
    return X, y, preprocessor


def split_data(X: pd.DataFrame, y: pd.Series, test_size: float = TEST_SIZE, random_state: int = RANDOM_STATE, stratify_y: bool = False) -> tuple:
    """
    Splits data into training and testing sets.
    """
    stratify_param = y if stratify_y and y.nunique() > 1 else None
    
    if stratify_y and y.nunique() <= 1:
        print(f"Warning: Stratification requested but target has only {y.nunique()} unique value(s). Skipping stratification.")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=stratify_param
    )
    print(f"Train/Test Split: X_train={X_train.shape}, X_test={X_test.shape}")
    return X_train, X_test, y_train, y_test

if __name__ == "__main__":
    print("--- Warning: You are running data_preparation.py directly ---")
    print("This file is a module within the 'src' package.")
    print("Direct execution like this can lead to 'ModuleNotFoundError'.")
    print("Please use 'scripts/training_models.py' from your project root for proper execution.")
    try:
        from src.config import RAW_DATA_PATH
        print(f"Test import of RAW_DATA_PATH: {RAW_DATA_PATH}")
    except ModuleNotFoundError as e:
        print(f"Test import failed: {e}. As expected if not run from project root.")
    except Exception as e:
        print(f"An unexpected error occurred during direct test: {e}")

