# src/data_preprocessing.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from src.config import RANDOM_STATE, CLAIM_PROBABILITY_TARGET, CLAIM_SEVERITY_TARGET, TOTAL_CLAIMS_COL

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
        if date_col in df.columns and pd.api.types.is_datetime64_any_dtype(df[date_col]):
            df[f'{date_col}_Year'] = df[date_col].dt.year
            df[f'{date_col}_Month'] = df[date_col].dt.month
            # df[f'{date_col}_Day'] = df[date_col].dt.day # Optional: if day-level granularity is useful
            df = df.drop(columns=[date_col]) # Drop original date column after extraction
        elif date_col in df.columns: # Attempt to convert if not already datetime
            try:
                df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
                # Recurse or re-apply after conversion
                if df[date_col].isnull().sum() < len(df) / 2: # Only if conversion was largely successful
                    df[f'{date_col}_Year'] = df[date_col].dt.year
                    df[f'{date_col}_Month'] = df[date_col].dt.month
                    df = df.drop(columns=[date_col])
                else:
                    print(f"Warning: Column '{date_col}' could not be reliably converted to datetime. Skipping feature engineering.")
            except:
                print(f"Warning: Column '{date_col}' could not be converted to datetime. Skipping feature engineering.")

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

def get_feature_names_after_preprocessing(preprocessor, numerical_features, categorical_features):
    """
    Returns the list of feature names after preprocessing, including one-hot encoded ones.
    """
    processed_feature_names = []

    # Add numerical features (original and engineered)
    processed_feature_names.extend(numerical_features)
    
    # Get names of one-hot encoded categorical features
    # Check if 'cat' transformer exists and has 'onehot' step
    if 'cat' in [t[0] for t in preprocessor.transformers] and hasattr(preprocessor.named_transformers_['cat'], 'named_steps'):
        ohe_transformer = preprocessor.named_transformers_['cat'].named_steps.get('onehot')
        if ohe_transformer:
            ohe_feature_names = ohe_transformer.get_feature_names_out(categorical_features)
            processed_feature_names.extend(list(ohe_feature_names))
    
    # Add any columns passed through by 'remainder' (if relevant and not already in numerical_features)
    # This requires more complex logic if 'remainder' is not 'drop' or if there are other transformers.
    # For now, assume numerical and categorical cover most cases.
    
    return processed_feature_names

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
        X = X[X[TOTAL_CLAIMS_COL] > 0].copy()
        y = X[TOTAL_CLAIMS_COL]
        X = X.drop(columns=[TOTAL_CLAIMS_COL, CLAIM_PROBABILITY_TARGET], errors='ignore') # Drop both targets
    elif problem_type == 'classification': # Claim Probability
        y = X[CLAIM_PROBABILITY_TARGET]
        X = X.drop(columns=[TOTAL_CLAIMS_COL, CLAIM_PROBABILITY_TARGET], errors='ignore') # Drop original claim and binary target for X
    else:
        raise ValueError("problem_type must be 'regression' or 'classification'")
    
    # Update feature lists based on what's actually in X after drops/filters/engineering
    # Dynamically build feature lists for the preprocessor based on the actual columns in X
    final_numerical_features = [col for col in numerical_cols if col in X.columns]
    final_categorical_features = [col for col in categorical_cols if col in X.columns]

    # Add engineered date features to numerical features if they exist and are numeric
    for date_col in date_cols:
        for suffix in ['_Year', '_Month']: # '_Day' if you uncommented it
            eng_col = f'{date_col}{suffix}'
            if eng_col in X.columns and pd.api.types.is_numeric_dtype(X[eng_col]) and eng_col not in final_numerical_features:
                final_numerical_features.append(eng_col)
            # If engineered features are categorical (e.g., month as category), add to categorical
            elif eng_col in X.columns and not pd.api.types.is_numeric_dtype(X[eng_col]) and eng_col not in final_categorical_features:
                 final_categorical_features.append(eng_col)


    preprocessor = create_preprocessor_pipeline(final_numerical_features, final_categorical_features)
    
    return X, y, preprocessor, final_numerical_features, final_categorical_features # Return updated lists for feature name extraction


def split_data(X, y, test_size, random_state, stratify=None):
    """Splits data into training and testing sets."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=stratify
    )
    return X_train, X_test, y_train, y_test