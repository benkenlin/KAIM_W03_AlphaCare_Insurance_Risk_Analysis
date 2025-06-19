# src/feature_engineering.py

import pandas as pd
import numpy as np

# Import necessary constants from config
from src.config import TOTAL_PREMIUM_COL, TOTAL_CLAIMS_COL

def engineer_features(df: pd.DataFrame, date_cols: list) -> pd.DataFrame:
    """
    Engineers new features from existing columns.
    This function includes logic for creating:
    - PremiumPerSumInsured
    - Date-related features (Year, Month) from date columns
    - VehicleAge_at_Transaction
    
    Assumes numerical conversions for relevant columns have already been done prior to this.
    
    Args:
        df (pd.DataFrame): The DataFrame with raw or partially processed features.
        date_cols (list): A list of column names that are date strings and should be parsed.

    Returns:
        pd.DataFrame: The DataFrame with engineered features.
    """
    df_engineered = df.copy()

    # Premium per Sum Insured
    if TOTAL_PREMIUM_COL in df_engineered.columns and 'SumInsured' in df_engineered.columns:
        # Ensure SumInsured is not zero to avoid division by zero
        df_engineered['SumInsured_for_calc'] = df_engineered['SumInsured'].replace(0, np.nan)
        # Fill NaN values in 'SumInsured_for_calc' with the median of 'SumInsured'
        # This handles cases where 'SumInsured' was 0 or NaN initially
        median_sum_insured = df_engineered['SumInsured'].median() if not df_engineered['SumInsured'].isnull().all() else 1.0 # Default to 1.0 if all NaN
        df_engineered['SumInsured_for_calc'] = df_engineered['SumInsured_for_calc'].fillna(median_sum_insured)
        
        # Calculate PremiumPerSumInsured
        df_engineered['PremiumPerSumInsured'] = df_engineered[TOTAL_PREMIUM_COL] / df_engineered['SumInsured_for_calc']
        
        # Handle inf/-inf results from division (e.g., if TotalPremium was non-zero but SumInsured was too small after cleaning)
        # Fill resulting NaNs with 0, and ensure non-negative
        df_engineered['PremiumPerSumInsured'] = df_engineered['PremiumPerSumInsured'].replace([np.inf, -np.inf], np.nan).fillna(0)
        df_engineered['PremiumPerSumInsured'] = df_engineered['PremiumPerSumInsured'].clip(lower=0)
        
        df_engineered = df_engineered.drop(columns=['SumInsured_for_calc']) # Drop the helper column
    else:
        # Create a placeholder column if inputs are missing to avoid key errors later
        df_engineered['PremiumPerSumInsured'] = 0.0 # Default value if components are missing


    # Date feature engineering
    for col in date_cols:
        if col in df_engineered.columns:
            # Convert to datetime, coercing errors
            df_engineered[col] = pd.to_datetime(df_engineered[col], errors='coerce')
            
            # Extract Year and Month, fill NaT from coercion, convert to int
            # Fill NaN years/months with 0, or a more appropriate default like the mode/median year/month if data exists
            df_engineered[f'{col}_Year'] = df_engineered[col].dt.year.fillna(0).astype(int)
            df_engineered[f'{col}_Month'] = df_engineered[col].dt.month.fillna(0).astype(int)
        else:
            print(f"Info: Date column '{col}' not found for feature engineering. Skipping its date part extraction.")
            # Ensure engineered columns exist, even if with default values
            df_engineered[f'{col}_Year'] = 0
            df_engineered[f'{col}_Month'] = 0


    # Additional feature: VehicleAge_at_Transaction
    if 'VehicleIntroDate_Year' in df_engineered.columns and 'TransactionMonth_Year' in df_engineered.columns:
        # Ensure these are numeric and handle any potential NaNs from date engineering
        vehicle_intro_year = pd.to_numeric(df_engineered['VehicleIntroDate_Year'], errors='coerce').fillna(0)
        transaction_month_year = pd.to_numeric(df_engineered['TransactionMonth_Year'], errors='coerce').fillna(0)
        
        # Calculate age, ensuring it's non-negative and integer
        df_engineered['VehicleAge_at_Transaction'] = (transaction_month_year - vehicle_intro_year)
        df_engineered['VehicleAge_at_Transaction'] = df_engineered['VehicleAge_at_Transaction'].clip(lower=0).astype(int)
    else:
        df_engineered['VehicleAge_at_Transaction'] = 0 # Default value if components are missing

    return df_engineered

if __name__ == "__main__":
    print("This is the feature_engineering.py module. It defines functions for creating new features.")
    print("It should not be run directly. It's imported by data_preparation.py.")

