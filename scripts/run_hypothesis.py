# scripts/run_hypothesis.py

import os
import sys
import pandas as pd
import numpy as np

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
from src.statistical_testing import calculate_risk_metrics, perform_chi_squared_test, perform_t_test, perform_z_test_proportions
from src.config import (
    RAW_DATA_PATH, ALPHA, TOTAL_CLAIMS_COL, TOTAL_PREMIUM_COL, HAS_CLAIM_COL, MARGIN_COL,
    DRIVER_GENDER_COL, PROVINCE_COL, POSTAL_CODE_COL, DATE_FEATURES_CLASSIFICATION,
    NUMERICAL_FEATURES_CLASSIFICATION, CATEGORICAL_FEATURES_CLASSIFICATION
)

def main():
    """
    Main function to orchestrate the hypothesis testing pipeline.
    """
    print("--- Starting Hypothesis Testing Pipeline ---")

    # 1. Load Raw Data
    df_raw = load_raw_data(RAW_DATA_PATH)
    if df_raw.empty:
        print("Could not load raw data. Exiting hypothesis testing pipeline.")
        return

    # --- Prepare Data for Hypothesis Testing ---
    # Apply initial cleaning and feature derivation to get data ready for metrics calculation.
    df_ht = df_raw.copy()

    # Define all columns that are or should be numeric for initial conversion
    potential_numeric_cols_ht = list(set([
        TOTAL_PREMIUM_COL, TOTAL_CLAIMS_COL, 'CustomValueEstimate', 'CapitalOutstanding',
        'SumInsured', 'CalculatedPremiumPerTerm', 'Age', 'VehicleAge',
        'RegistrationYear', 'Cylinders', 'cubiccapacity', 'kilowatts',
        'NumberOfDoors', 'NumberOfVehiclesInFleet'
    ]))
    potential_numeric_cols_ht_in_df = [col for col in potential_numeric_cols_ht if col in df_ht.columns]

    df_ht = _convert_comma_to_dot_and_numeric(df_ht, potential_numeric_cols_ht_in_df)

    # --- CRITICAL FIX: Explicitly ensure key categorical columns remain string/object ---
    # This prevents them from being misinterpreted or dropped by intermediate steps
    # before handle_missing_data can process them as categorical.
    for col in [DRIVER_GENDER_COL, PROVINCE_COL, POSTAL_CODE_COL]:
        if col in df_ht.columns:
            df_ht[col] = df_ht[col].astype(str)
            # Also clean whitespace and replace 'nan' string values
            df_ht[col] = df_ht[col].str.strip().replace('nan', np.nan, regex=False)
        else:
            print(f"Warning: Critical categorical column '{col}' not found in raw data. May affect hypothesis tests.")
            
    # Engineer features (like date components and vehicle age at transaction)
    df_ht = engineer_features(df_ht, DATE_FEATURES_CLASSIFICATION)

    # Calculate core risk metrics including HasClaim and Margin.
    df_metrics = calculate_risk_metrics(df_ht)

    # Define the full set of numerical and categorical features expected after initial processing
    # These are used for handling missing data across the entire dataset for consistency
    all_numeric_cols = [col for col in df_metrics.select_dtypes(include=np.number).columns if col not in [HAS_CLAIM_COL]] # Exclude HAS_CLAIM_COL as it's a binary target
    all_categorical_cols = [col for col in df_metrics.select_dtypes(include='object').columns]

    # Handle missing data comprehensively across the prepared DataFrame
    df_prepared_ht = handle_missing_data(df_metrics, all_numeric_cols, all_categorical_cols)

    print("\n--- Data Prepared for Hypothesis Testing ---")
    print(f"Shape of prepared data: {df_prepared_ht.shape}")
    print("Sample data with metrics:")
    # Ensure all columns in the .head() print statement are present
    display_cols = [HAS_CLAIM_COL, 'ClaimAmountWhenClaimed', MARGIN_COL, PROVINCE_COL, POSTAL_CODE_COL, DRIVER_GENDER_COL]
    # Filter to only show columns that actually exist in the DataFrame
    existing_display_cols = [col for col in display_cols if col in df_prepared_ht.columns]
    print(df_prepared_ht[existing_display_cols].head())


    # --- 2. Perform Hypothesis Tests ---

    # H₀: There are no risk differences across provinces
    print("\n--- Testing H₀: No risk differences across provinces ---")
    if PROVINCE_COL in df_prepared_ht.columns:
        # For Claim Frequency (Categorical outcome: HasClaim)
        # Select two provinces for comparison, e.g., 'Gauteng' and 'Western Cape'
        # You may need to adjust these based on the actual unique values in your data
        province1 = 'Gauteng'
        province2 = 'Western Cape'

        if province1 in df_prepared_ht[PROVINCE_COL].unique() and \
           province2 in df_prepared_ht[PROVINCE_COL].unique():
            
            df_prov1 = df_prepared_ht[df_prepared_ht[PROVINCE_COL] == province1]
            df_prov2 = df_prepared_ht[df_prepared_ht[PROVINCE_COL] == province2]

            # Test 1a: Claim Frequency (Proportions)
            claims_prov1 = df_prov1[HAS_CLAIM_COL].sum()
            total_prov1 = len(df_prov1)
            claims_prov2 = df_prov2[HAS_CLAIM_COL].sum()
            total_prov2 = len(df_prov2)
            
            perform_z_test_proportions(
                claims_prov1, total_prov1, claims_prov2, total_prov2,
                alpha=ALPHA,
                hypothesis_text=f"Claim Frequency difference between {province1} and {province2}"
            )

            # Test 1b: Claim Severity (Mean of ClaimAmountWhenClaimed)
            perform_t_test(
                df_prov1['ClaimAmountWhenClaimed'].dropna(),
                df_prov2['ClaimAmountWhenClaimed'].dropna(),
                alpha=ALPHA,
                hypothesis_text=f"Claim Severity difference between {province1} and {province2}"
            )
        else:
            print(f"Warning: Provinces '{province1}' or '{province2}' not found in data for comparison.")
    else:
        print(f"Warning: Column '{PROVINCE_COL}' not found for province-based hypothesis testing.")


    # H₀: There are no risk differences between zip codes
    print("\n--- Testing H₀: No risk differences between zip codes ---")
    if POSTAL_CODE_COL in df_prepared_ht.columns:
        # Due to high cardinality of zip codes, select top 2 most frequent for a meaningful comparison.
        top_zip_codes = df_prepared_ht[POSTAL_CODE_COL].value_counts().nlargest(2).index.tolist()

        if len(top_zip_codes) >= 2:
            zip_code1 = top_zip_codes[0]
            zip_code2 = top_zip_codes[1]

            df_zip1 = df_prepared_ht[df_prepared_ht[POSTAL_CODE_COL] == zip_code1]
            df_zip2 = df_prepared_ht[df_prepared_ht[POSTAL_CODE_COL] == zip_code2]

            # Test 2a: Claim Frequency (Proportions)
            claims_zip1 = df_zip1[HAS_CLAIM_COL].sum()
            total_zip1 = len(df_zip1)
            claims_zip2 = df_zip2[HAS_CLAIM_COL].sum()
            total_zip2 = len(df_zip2)

            perform_z_test_proportions(
                claims_zip1, total_zip1, claims_zip2, total_zip2,
                alpha=ALPHA,
                hypothesis_text=f"Claim Frequency difference between Zip Code {zip_code1} and {zip_code2}"
            )

            # Test 2b: Claim Severity (Mean of ClaimAmountWhenClaimed)
            perform_t_test(
                df_zip1['ClaimAmountWhenClaimed'].dropna(),
                df_zip2['ClaimAmountWhenClaimed'].dropna(),
                alpha=ALPHA,
                hypothesis_text=f"Claim Severity difference between Zip Code {zip_code1} and {zip_code2}"
            )
        else:
            print(f"Warning: Not enough unique {POSTAL_CODE_COL} to compare (found {len(top_zip_codes)}).")
    else:
        print(f"Warning: Column '{POSTAL_CODE_COL}' not found for zip code-based hypothesis testing.")


    # H₀: There are no significant margin (profit) difference between zip codes
    print("\n--- Testing H₀: No significant margin (profit) difference between zip codes ---")
    if POSTAL_CODE_COL in df_prepared_ht.columns and MARGIN_COL in df_prepared_ht.columns:
        top_zip_codes = df_prepared_ht[POSTAL_CODE_COL].value_counts().nlargest(2).index.tolist()

        if len(top_zip_codes) >= 2:
            zip_code1 = top_zip_codes[0]
            zip_code2 = top_zip_codes[1]

            df_zip1 = df_prepared_ht[df_prepared_ht[POSTAL_CODE_COL] == zip_code1]
            df_zip2 = df_prepared_ht[df_prepared_ht[POSTAL_CODE_COL] == zip_code2]
            
            # Test 3: Margin difference (Mean of Margin)
            perform_t_test(
                df_zip1[MARGIN_COL].dropna(),
                df_zip2[MARGIN_COL].dropna(),
                alpha=ALPHA,
                hypothesis_text=f"Margin difference between Zip Code {zip_code1} and {zip_code2}"
            )
        else:
            print(f"Warning: Not enough unique {POSTAL_CODE_COL} to compare (found {len(top_zip_codes)}).")
    else:
        print(f"Warning: Columns '{POSTAL_CODE_COL}' or '{MARGIN_COL}' not found for margin hypothesis testing.")


    # H₀: There are not significant risk difference between Women and Men (DriverGender)
    print("\n--- Testing H₀: No significant risk difference between Women and Men ---")
    if DRIVER_GENDER_COL in df_prepared_ht.columns:
        # Assuming 'Male' and 'Female' are the primary categories for DriverGender
        gender1 = 'Male'
        gender2 = 'Female'
        
        # Filter for policies where DriverGender is explicitly Male or Female
        df_gender_filtered = df_prepared_ht[df_prepared_ht[DRIVER_GENDER_COL].isin([gender1, gender2])]

        if gender1 in df_gender_filtered[DRIVER_GENDER_COL].unique() and \
           gender2 in df_gender_filtered[DRIVER_GENDER_COL].unique():
            
            df_male = df_gender_filtered[df_gender_filtered[DRIVER_GENDER_COL] == gender1]
            df_female = df_gender_filtered[df_gender_filtered[DRIVER_GENDER_COL] == gender2]

            # Test 4a: Claim Frequency (Proportions)
            claims_male = df_male[HAS_CLAIM_COL].sum()
            total_male = len(df_male)
            claims_female = df_female[HAS_CLAIM_COL].sum()
            total_female = len(df_female)

            perform_z_test_proportions(
                claims_male, total_male, claims_female, total_female,
                alpha=ALPHA,
                hypothesis_text=f"Claim Frequency difference between {gender1} and {gender2} Drivers"
            )

            # Test 4b: Claim Severity (Mean of ClaimAmountWhenClaimed)
            perform_t_test(
                df_male['ClaimAmountWhenClaimed'].dropna(),
                df_female['ClaimAmountWhenClaimed'].dropna(),
                alpha=ALPHA,
                hypothesis_text=f"Claim Severity difference between {gender1} and {gender2} Drivers"
            )
        else:
            print(f"Warning: Genders '{gender1}' or '{gender2}' not found or insufficient data for comparison. Found: {df_prepared_ht[DRIVER_GENDER_COL].unique()}")
    else:
        print(f"Warning: Column '{DRIVER_GENDER_COL}' not found for gender-based hypothesis testing. Please check config.py and data.")


    print("\n--- Hypothesis Testing Pipeline Completed ---")

if __name__ == "__main__":
    main()

