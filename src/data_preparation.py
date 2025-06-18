# src/data_preparation.py

import pandas as pd
import numpy as np # Make sure numpy is imported for np.nan if used
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
# --- IMPORTANT: This import is the one failing if run directly ---
from src.config import RANDOM_STATE, CLAIM_PROBABILITY_TARGET, CLAIM_SEVERITY_TARGET, TOTAL_CLAIMS_COL, RAW_DATA_PATH, POLICY_ID_COL, TRANSACTION_MONTH_COL, TOTAL_PREMIUM_COL, HAS_CLAIM_COL, MARGIN_COL # Added more config imports needed for context

# --- MISSING FUNCTION ADDED BACK IN ---
def load_raw_data(file_path: str = RAW_DATA_PATH) -> pd.DataFrame:
    """
    Loads raw data from a specified file path.

    Args:
        file_path (str): The path to the raw data file. Defaults to RAW_DATA_PATH from config.

    Returns:
        pd.DataFrame: The loaded pandas DataFrame. Returns an empty DataFrame on error.
    """
    try:
        df = pd.read_csv(file_path, sep='|')
        print(f"Data loaded successfully from {file_path}. Shape: {df.shape}")
        return df
    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found.")
        return pd.DataFrame()
    except Exception as e:
        print(f"An error occurred while loading the data: {e}")
        return pd.DataFrame()
# --- END OF MISSING FUNCTION ADDED BACK IN ---


def handle_missing_data(df):
    """
    Basic handling of missing values. Imputes numerical columns with mean,
    and categorical with mode.
    """
    for col in df.columns:
        if df[col].isnull().sum() > 0:
            if pd.api.types.is_numeric_dtype(df[col]):
                df[col] = df[col].fillna(df[col].mean())
            else:
                # For categorical, ensure it's treated as object/category before filling
                df[col] = df[col].astype('object').fillna(df[col].mode()[0])
    return df

def engineer_features(df, date_features):
    """
    Creates new features from existing ones.
    - Extracts year, month, day from date columns.
    - HasClaim: Binary indicator if a policy had any claim.
    """
    # Feature engineering from date columns
    for date_col in date_features:
        if date_col in df.columns:
            # First, try to convert to datetime if it's not already
            if not pd.api.types.is_datetime64_any_dtype(df[date_col]):
                df[date_col] = pd.to_datetime(df[date_col], errors='coerce')

            if pd.api.types.is_datetime64_any_dtype(df[date_col]): # Check again after coerce
                df[f'{date_col}_Year'] = df[date_col].dt.year
                df[f'{date_col}_Month'] = df[date_col].dt.month
                # df[f'{date_col}_Day'] = df[date_col].dt.day # Optional: if day-level granularity is useful
                df = df.drop(columns=[date_col]) # Drop original date column after extraction
            else:
                print(f"Warning: Column '{date_col}' could not be reliably converted to datetime. Skipping feature engineering for this column.")

    # Ensure 'TotalClaims' column exists before creating 'HasClaim'
    if TOTAL_CLAIMS_COL in df.columns:
        df[CLAIM_PROBABILITY_TARGET] = (df[TOTAL_CLAIMS_COL] > 0).astype(int)
    
    return df


def create_preprocessor_pipeline(numerical_features, categorical_features):
    """
    Creates a ColumnTransformer for preprocessing numerical and categorical features.
    """
    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features)
        ],
        remainder='passthrough' # Keep other columns (like IDs, targets after drop)
    )
    return preprocessor

# --- SIMPLIFIED FUNCTION ---
def get_feature_names_after_preprocessing(preprocessor):
    """
    Returns the list of feature names after preprocessing using the fitted ColumnTransformer's
    get_feature_names_out method.

    Args:
        preprocessor (ColumnTransformer): The fitted ColumnTransformer.

    Returns:
        list: List of feature names after preprocessing.
    """
    return list(preprocessor.get_feature_names_out())
# --- END SIMPLIFIED FUNCTION ---

# ... (prepare_data_for_modeling function - no change to its logic, but numerical_features and categorical_features
#      are now directly passed to create_preprocessor_pipeline, and the return of these lists from
#      prepare_data_for_modeling is no longer needed by get_feature_names_after_preprocessing) ...

def prepare_data_for_modeling(df, numerical_cols, categorical_cols, date_cols, cols_to_drop, problem_type='regression'):
    """
    Main function to preprocess data for modeling.
    Handles missing values, feature engineering, and returns X, y, and preprocessor.
    """
    df_processed = df.copy()

    # Drop specified columns early
    df_processed = df_processed.drop(columns=[col for col in cols_to_drop if col in df_processed.columns], errors='ignore')

    # Apply feature engineering
    df_processed = engineer_features(df_processed, date_cols)

    # Apply general missing value handling
    df_processed = handle_missing_data(df_processed)

    # Select features and target based on the problem type
    X = df_processed.copy()
    y = None

    if problem_type == 'regression': # Claim Severity
        # Only consider policies that had claims for severity prediction
        X = X[X[TOTAL_CLAIMS_COL] > 0].copy() # Filter for policies with claims
        y = X[TOTAL_CLAIMS_COL]
        X = X.drop(columns=[TOTAL_CLAIMS_COL, CLAIM_PROBABILITY_TARGET], errors='ignore') # Drop both targets
    elif problem_type == 'classification': # Claim Probability
        y = X[CLAIM_PROBABILITY_TARGET]
        X = X.drop(columns=[TOTAL_CLAIMS_COL, CLAIM_PROBABILITY_TARGET], errors='ignore') # Drop original claim and binary target for X
    else:
        raise ValueError("problem_type must be 'regression' or 'classification'")
    
    # Update feature lists based on what's actually in X after drops/filters/engineering
    # Ensure numerical_cols and categorical_cols contain only columns present in X
    final_numerical_features = [col for col in numerical_cols if col in X.columns]
    final_categorical_features = [col for col in categorical_cols if col in X.columns]

    # Add engineered date features to numerical features if they exist and are numeric
    for date_col in date_cols:
        for suffix in ['_Year', '_Month']: # '_Day' if you uncommented it
            eng_col = f'{date_col}{suffix}'
            if eng_col in X.columns:
                if pd.api.types.is_numeric_dtype(X[eng_col]) and eng_col not in final_numerical_features:
                    final_numerical_features.append(eng_col)
                elif not pd.api.types.is_numeric_dtype(X[eng_col]) and eng_col not in final_categorical_features:
                    # If somehow non-numeric but not intended categorical
                    # For this case, it might indicate an issue with date engineering
                    print(f"Warning: Engineered date feature '{eng_col}' is not numeric and not added to categorical features.")

    # Remove duplicates from feature lists (can happen if a column is in both original and engineered, or listed twice)
    final_numerical_features = list(set(final_numerical_features))
    final_categorical_features = list(set(final_categorical_features))


    preprocessor = create_preprocessor_pipeline(final_numerical_features, final_categorical_features)
    
    # The preprocessor is not fitted yet at this point, it will be fitted in run_full_pipeline.py
    # We no longer need to return num_features_clf, cat_features_clf as get_feature_names_after_preprocessing
    # now uses the preprocessor directly.
    return X, y, preprocessor # Simplified return


def split_data(X, y, test_size, random_state, stratify=None):
    """Splits data into training and testing sets."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=stratify
    )
    return X_train, X_test, y_train, y_test

# --- IMPORTANT: This block should NOT be directly run in a modular project ---
# --- IMPORTANT: This block should NOT be directly run in a modular project ---
if __name__ == "__main__":
    print("--- Warning: You are running data_preparation.py directly ---")
    print("This file is a module within the 'src' package.")
    print("Direct execution like this can lead to 'ModuleNotFoundError'.")
    # This is the line with the error, it's missing the closing quote
    print("Please use 'scripts/run_full_pipeline.py' from your project root for proper execution.")
    print("\nAttempting a minimal test (may still fail if not properly set up):")
    try:
        from src.config import RAW_DATA_PATH # Fixed: Changed from 'config' to 'src.config' for robust direct run
        print(f"Test import of RAW_DATA_PATH: {RAW_DATA_PATH}")
    except ModuleNotFoundError as e: # Catch the specific error for better message
        print(f"Test import failed: {e}. As expected if not run from project root.")
    # Example usage would go here, but it's better demonstrated in the main pipeline script.
