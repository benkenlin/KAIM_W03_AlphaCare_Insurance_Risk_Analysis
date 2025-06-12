# scripts/run_eda_pipeline.py
import pandas as pd
import os
import sys

# Add the parent directory (project root) to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Now your imports will work:
from src.utils import load_data
from src.eda import generate_eda_report
from src.utils import load_data
from src.eda import generate_eda_report
from src.config import (
    RAW_DATA_PATH, NUMERICAL_FEATURES, CATEGORICAL_FEATURES, DATE_FEATURES,
    EDA_REPORTS_DIR, TOTAL_PREMIUM_COL, TOTAL_CLAIMS_COL
)

def main():
    print("--- Starting EDA Pipeline ---")
    
    # Load raw data
    df = load_data(RAW_DATA_PATH, sep='|') 
    
    # Ensure TOTAL_PREMIUM_COL and TOTAL_CLAIMS_COL are present and numeric
    if TOTAL_PREMIUM_COL not in df.columns:
        print(f"Warning: '{TOTAL_PREMIUM_COL}' not found. Adding dummy column for EDA testing.")
        df[TOTAL_PREMIUM_COL] = 1000.0 # Dummy value
    else:
        df[TOTAL_PREMIUM_COL] = pd.to_numeric(df[TOTAL_PREMIUM_COL], errors='coerce').fillna(0)
        
    if TOTAL_CLAIMS_COL not in df.columns:
        print(f"Warning: '{TOTAL_CLAIMS_COL}' not found. Adding dummy column for EDA testing.")
        df[TOTAL_CLAIMS_COL] = 0.0 # Dummy value
    else:
        df[TOTAL_CLAIMS_COL] = pd.to_numeric(df[TOTAL_CLAIMS_COL], errors='coerce').fillna(0)


    # Now call the main EDA report generation function
    generate_eda_report(
        df=df,
        numerical_cols=NUMERICAL_FEATURES,
        categorical_cols=CATEGORICAL_FEATURES,
        date_cols=DATE_FEATURES,
        output_dir=EDA_REPORTS_DIR
    )
    
    print("--- EDA Pipeline Completed ---")

if __name__ == "__main__":
    main()