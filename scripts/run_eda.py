# scripts/run_eda.py

import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# --- Add the project root to the Python path ---
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, '..'))

if project_root not in sys.path:
    sys.path.insert(0, project_root)
    print(f"Added project root '{project_root}' to sys.path.")
else:
    print(f"Project root '{project_root}' was already in sys.path.")

# Import modules from src
from src.data_preparation import load_raw_data, _convert_comma_to_dot_and_numeric, engineer_features, handle_missing_data
from src.config import (
    RAW_DATA_PATH, TOTAL_CLAIMS_COL, TOTAL_PREMIUM_COL, HAS_CLAIM_COL, MARGIN_COL,
    NUMERICAL_FEATURES_CLASSIFICATION, CATEGORICAL_FEATURES_CLASSIFICATION, DATE_FEATURES_CLASSIFICATION,
    POLICY_ID_COL, DRIVER_GENDER_COL, PROVINCE_COL, POSTAL_CODE_COL
)

def perform_eda(df: pd.DataFrame, title: str = "Exploratory Data Analysis"):
    """
    Performs comprehensive Exploratory Data Analysis (EDA) on the given DataFrame.
    Includes data overview, descriptive statistics, missing values, and visualizations.

    Args:
        df (pd.DataFrame): The DataFrame to analyze.
        title (str): Title for the EDA report.
    """
    print(f"\n--- {title} ---")
    
    print("\n1. Data Overview:")
    print(f"Shape: {df.shape} (Rows: {df.shape[0]}, Columns: {df.shape[1]})")
    print("\nFirst 5 Rows:")
    print(df.head())
    print("\nLast 5 Rows:")
    print(df.tail())
    print("\nColumn Information (dtypes and non-null counts):")
    df.info()

    print("\n2. Descriptive Statistics for Numerical Columns:")
    print(df.describe().T)

    print("\n3. Missing Values Analysis:")
    missing_data = df.isnull().sum()
    missing_percent = (df.isnull().sum() / len(df)) * 100
    missing_df = pd.DataFrame({'Total Missing': missing_data, 'Percentage (%)': missing_percent})
    missing_df = missing_df[missing_df['Total Missing'] > 0].sort_values(by='Total Missing', ascending=False)
    print(missing_df)
    if missing_df.empty:
        print("No missing values found in the DataFrame.")

    print("\n4. Unique Values for Categorical Columns (Top 10 categories):")
    for col in df.select_dtypes(include='object').columns:
        print(f"\n--- Column: {col} ---")
        print(df[col].value_counts(dropna=False).head(10))

    print("\n5. Target Variable Analysis:")
    if HAS_CLAIM_COL in df.columns:
        print(f"\n{HAS_CLAIM_COL} (Claim Frequency) Distribution:")
        print(df[HAS_CLAIM_COL].value_counts(normalize=True))
        sns.countplot(x=HAS_CLAIM_COL, data=df)
        plt.title(f'Distribution of {HAS_CLAIM_COL}')
        plt.show()
    else:
        print(f"'{HAS_CLAIM_COL}' column not found for target analysis.")

    if TOTAL_CLAIMS_COL in df.columns and HAS_CLAIM_COL in df.columns:
        print(f"\n{TOTAL_CLAIMS_COL} (Claim Severity) Distribution (for policies with claims):")
        claims_only_df = df[df[HAS_CLAIM_COL] == 1].copy()
        if not claims_only_df.empty:
            print(claims_only_df[TOTAL_CLAIMS_COL].describe())
            plt.figure(figsize=(10, 6))
            sns.histplot(claims_only_df[TOTAL_CLAIMS_COL], bins=50, kde=True)
            plt.title(f'Distribution of {TOTAL_CLAIMS_COL} (Conditional on Claim)')
            plt.xlabel('Claim Amount')
            plt.ylabel('Frequency')
            plt.yscale('log') # Log scale often useful for skewed financial data
            plt.show()
        else:
            print("No claims found for Claim Severity analysis.")

    if MARGIN_COL in df.columns:
        print(f"\n{MARGIN_COL} Distribution:")
        print(df[MARGIN_COL].describe())
        plt.figure(figsize=(10, 6))
        sns.histplot(df[MARGIN_COL], bins=50, kde=True)
        plt.title(f'Distribution of {MARGIN_COL}')
        plt.xlabel('Margin')
        plt.ylabel('Frequency')
        plt.show()


    print("\n6. Feature Distributions (Sample for key numerical and categorical):")
    # Plot distributions for a few key numerical features
    num_cols_to_plot = [col for col in ['TotalPremium', 'SumInsured', 'Age', 'VehicleAge'] if col in df.columns]
    for col in num_cols_to_plot:
        plt.figure(figsize=(8, 5))
        sns.histplot(df[col].dropna(), kde=True)
        plt.title(f'Distribution of {col}')
        plt.show()

    # Plot distributions for a few key categorical features
    cat_cols_to_plot = [col for col in ['PolicyType', DRIVER_GENDER_COL, 'VehicleType', PROVINCE_COL] if col in df.columns]
    for col in cat_cols_to_plot:
        plt.figure(figsize=(8, 5))
        sns.countplot(y=col, data=df, order=df[col].value_counts().index)
        plt.title(f'Distribution of {col}')
        plt.show()

    print("\n7. Correlation Matrix (Numerical Features):")
    # Select only potentially numerical columns before correlation to avoid errors
    numeric_df = df.select_dtypes(include=np.number)
    if not numeric_df.empty:
        plt.figure(figsize=(12, 10))
        sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
        plt.title('Correlation Matrix of Numerical Features')
        plt.show()
    else:
        print("No numerical columns found for correlation matrix.")


def main():
    """
    Main function to run the EDA pipeline.
    """
    print("--- Starting Exploratory Data Analysis Pipeline ---")

    # Load raw data
    df_raw = load_raw_data(RAW_DATA_PATH)
    if df_raw.empty:
        print("Could not load raw data. Exiting EDA pipeline.")
        return

    # --- Initial Data Cleaning and Feature Derivation for EDA ---
    # Perform basic type conversions and calculate derived features needed for EDA
    # These are simplified versions compared to prepare_data_for_modeling,
    # just enough to get meaningful EDA.
    df_eda = df_raw.copy()

    # Define all columns that are or should be numeric (including those derived/engineered later)
    # This list should be comprehensive based on your config's numerical features,
    # plus any raw columns that become targets or intermediate calcs (like TotalClaims)
    potential_numeric_cols = list(set([
        TOTAL_PREMIUM_COL, TOTAL_CLAIMS_COL, 'CustomValueEstimate', 'CapitalOutstanding',
        'SumInsured', 'CalculatedPremiumPerTerm', 'Age', 'VehicleAge',
        'RegistrationYear', 'Cylinders', 'cubiccapacity', 'kilowatts',
        'NumberOfDoors', 'NumberOfVehiclesInFleet'
    ]))
    # Filter to only include columns actually present in the DataFrame
    potential_numeric_cols_in_df = [col for col in potential_numeric_cols if col in df_eda.columns]

    # Perform comma-to-dot and numeric conversion
    df_eda = _convert_comma_to_dot_and_numeric(df_eda, potential_numeric_cols_in_df)

    # Calculate HasClaim and Margin (needed for EDA on targets)
    if TOTAL_CLAIMS_COL in df_eda.columns:
        df_eda[HAS_CLAIM_COL] = (df_eda[TOTAL_CLAIMS_COL] > 0).astype(int)
    else:
        df_eda[HAS_CLAIM_COL] = 0

    if TOTAL_PREMIUM_COL in df_eda.columns and TOTAL_CLAIMS_COL in df_eda.columns:
        df_eda[MARGIN_COL] = df_eda[TOTAL_PREMIUM_COL] - df_eda[TOTAL_CLAIMS_COL]
    else:
        df_eda[MARGIN_COL] = 0.0

    # Ensure non-negativity for relevant financial/count columns after derivation
    for col in potential_numeric_cols_in_df + [MARGIN_COL]:
        if col in df_eda.columns and pd.api.types.is_numeric_dtype(df_eda[col]):
            df_eda[col] = df_eda[col].clip(lower=0)


    # Apply feature engineering for derived date features and others
    # Use the same date_cols as defined in config for consistency
    df_eda_engineered = engineer_features(df_eda, DATE_FEATURES_CLASSIFICATION)

    # Finally, handle missing data using the robust function from data_preparation
    # We pass all features here, as we want to analyze them, not just modeling features
    all_numeric_features_for_eda_imputation = [col for col in df_eda_engineered.select_dtypes(include=np.number).columns if col not in [HAS_CLAIM_COL]]
    all_categorical_features_for_eda_imputation = [col for col in df_eda_engineered.select_dtypes(include='object').columns]
    
    df_final_eda = handle_missing_data(df_eda_engineered, 
                                       all_numeric_features_for_eda_imputation, 
                                       all_categorical_features_for_eda_imputation)


    perform_eda(df_final_eda, "Comprehensive EDA Report")

    print("\n--- EDA Pipeline Completed ---")

if __name__ == "__main__":
    main()

