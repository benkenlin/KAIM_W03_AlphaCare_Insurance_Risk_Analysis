# src/config.py

import os

# --- General Project Settings ---
RANDOM_STATE = 42 # For reproducibility in random operations

# --- Column Names from Raw Data ---
TOTAL_CLAIMS_COL = 'TotalClaims'
TOTAL_PREMIUM_COL = 'TotalPremium'
POLICY_ID_COL = 'PolicyID'
TRANSACTION_MONTH_COL = 'TransactionMonth' # Expected format: YYYYMM

# --- Derived Metric Names (for clarity and consistency) ---
CLAIM_PROBABILITY_TARGET = 'HasClaim' # For classification target
CLAIM_SEVERITY_TARGET = 'TotalClaims' # For regression target (when a claim occurs)
HAS_CLAIM_COL = 'HasClaim' # Same as CLAIM_PROBABILITY_TARGET, for internal use
MARGIN_COL = 'Margin'

# --- File Paths (relative to the project root) ---

# Get the directory of the current script (config.py)
_current_dir = os.path.dirname(os.path.abspath(__file__))

# Construct the path to the project root (one level up from src)
# This assumes src/config.py is directly inside the 'src' folder which is in the project root.
_project_root = os.path.abspath(os.path.join(_current_dir, '..'))

# Define paths relative to the project root
RAW_DATA_PATH = os.path.join(_project_root, 'data', 'raw', 'MachineLearningRating_v3.txt')
PROCESSED_DATA_PATH = os.path.join(_project_root, 'data', 'processed', 'processed_insurance_data.parquet')