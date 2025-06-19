# src/config.py

import os

# --- General Project Settings ---
RANDOM_STATE = 42 # For reproducibility in random operations
TEST_SIZE = 0.2   # Standard test set size for train_test_split (e.g., 20% for testing)
ALPHA = 0.05      # Significance level for hypothesis testing

# --- Column Names from Raw Data ---
# These should match the column names in your 'MachineLearningRating_v3.txt'
TOTAL_CLAIMS_COL = 'TotalClaims'
TOTAL_PREMIUM_COL = 'TotalPremium'
POLICY_ID_COL = 'PolicyID'
TRANSACTION_MONTH_COL = 'TransactionMonth' # Expected format: YYYY-MM-DD HH:MM:SS or similar
DRIVER_GENDER_COL = 'DriverGender'
PROVINCE_COL = 'Province'
POSTAL_CODE_COL = 'PostalCode' # For zip code analysis

# --- Derived Metric Names (for clarity and consistency) ---
CLAIM_PROBABILITY_TARGET = 'HasClaim' # For classification target
CLAIM_SEVERITY_TARGET = 'TotalClaims' # For regression target (when a claim occurs)
HAS_CLAIM_COL = 'HasClaim' # Same as CLAIM_PROBABILITY_TARGET, for internal use
MARGIN_COL = 'Margin'      # TotalPremium - TotalClaims

# --- File Paths (relative to the the project root) ---

# Get the directory of the current script (config.py)
_current_dir = os.path.dirname(os.path.abspath(__file__))

# Construct the path to the project root (one level up from src)
# This assumes src/config.py is directly inside the 'src' folder which is in the project root.
PROJECT_ROOT = os.path.abspath(os.path.join(_current_dir, '..'))

# Define a data directory within the project root
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')

# Paths for raw and processed data files
RAW_DATA_PATH = os.path.join(DATA_DIR, 'raw', 'MachineLearningRating_v3.txt')
PROCESSED_DATA_PATH = os.path.join(DATA_DIR, 'processed', 'processed_insurance_data.parquet')

# Path for saving trained models and preprocessors
MODEL_DIR = os.path.join(PROJECT_ROOT, 'models')


# --- Feature Lists for Modeling ---
# These lists define which columns are used as numerical, categorical, and date features.
# Adjust these based on your actual dataset columns and your feature engineering strategy.

# Features for Claim Probability (Classification) Model
NUMERICAL_FEATURES_CLASSIFICATION = [
    'TotalPremium',
    'SumInsured',
    'CustomValueEstimate',
    'CapitalOutstanding',
    'CalculatedPremiumPerTerm',
    'Age',
    'VehicleAge',
    MARGIN_COL, # Engineered feature
    'RegistrationYear', # Treat as numerical, will be coerced
    'Cylinders',        # Treat as numerical, will be coerced
    'cubiccapacity',    # Treat as numerical, will be coerced
    'kilowatts',        # Treat as numerical, will be coerced
    'NumberOfDoors',    # Treat as numerical, will be coerced
    'NumberOfVehiclesInFleet', # Treat as numerical, will be coerced
    # Engineered date features like 'TransactionMonth_Year', 'TransactionMonth_Month',
    # 'VehicleIntroDate_Year', 'VehicleIntroDate_Month' will be added dynamically.
    # 'VehicleAge_at_Transaction', # Engineered feature
    # 'PremiumPerSumInsured' # Engineered feature
]

CATEGORICAL_FEATURES_CLASSIFICATION = [
    'PolicyType',
    DRIVER_GENDER_COL, # 'DriverGender'
    'VehicleType',
    'Geolocation',
    'FuelType',
    'VehicleSegment',
    PROVINCE_COL, # 'Province'
    POSTAL_CODE_COL, # 'PostalCode'
    # Additional categorical features identified from raw data
    'IsVATRegistered',
    'Citizenship',
    'LegalType',
    'Title',
    'Language',
    'Bank',
    'AccountType',
    'MaritalStatus',
    'Gender',
    'Country',
    'MainCrestaZone',
    'SubCrestaZone',
    'ItemType',
    'mmcode', # Treat as categorical (often codes)
    'make',
    'Model',
    'bodytype',
    'AlarmImmobiliser',
    'TrackingDevice',
    'NewVehicle',
    'WrittenOff',
    'Rebuilt',
    'Converted',
    'CrossBorder',
    'TermFrequency',
    'ExcessSelected',
    'CoverCategory',
    'CoverType',
    'CoverGroup',
    'Section',
    'Product',
    'StatutoryClass',
    'StatutoryRiskType'
]

DATE_FEATURES_CLASSIFICATION = [
    TRANSACTION_MONTH_COL, # 'TransactionMonth'
    'VehicleIntroDate'
]

COLS_TO_DROP_CLASSIFICATION = [
    POLICY_ID_COL # Drop policy ID as it's an identifier and not a feature
    # 'UnderwrittenCoverID' # Also an ID, can be dropped if not needed for specific lookup
    # Add other columns here that are not features or targets (e.g., specific metadata)
]

# Features for Claim Severity (Regression) Model - often the same as classification features
NUMERICAL_FEATURES_REGRESSION = NUMERICAL_FEATURES_CLASSIFICATION[:] # Copy list
CATEGORICAL_FEATURES_REGRESSION = CATEGORICAL_FEATURES_CLASSIFICATION[:] # Copy list
DATE_FEATURES_REGRESSION = DATE_FEATURES_CLASSIFICATION[:] # Copy list
COLS_TO_DROP_REGRESSION = COLS_TO_DROP_CLASSIFICATION[:] # Copy list

