# src/config.py
import os

# Define the base directory of the project
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# --- Data Paths ---
DATA_DIR = os.path.join(BASE_DIR, 'data')
RAW_DATA_DIR = os.path.join(DATA_DIR, 'raw')
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, 'processed')
INTERIM_DATA_DIR = os.path.join(DATA_DIR, 'interim')

# Raw data file
RAW_DATA_PATH = os.path.join(RAW_DATA_DIR, 'MachineLearning_v3.txt')

# --- Output Paths ---
OUTPUT_DIR = os.path.join(BASE_DIR, 'output')
EDA_REPORTS_DIR = os.path.join(OUTPUT_DIR, 'eda_reports')
HYPOTHESIS_REPORTS_DIR = os.path.join(OUTPUT_DIR, 'hypothesis_reports')
MODELS_DIR = os.path.join(OUTPUT_DIR, 'models')
INTERPRETATIONS_DIR = os.path.join(OUTPUT_DIR, 'interpretations')

# Create directories if they don't exist
os.makedirs(RAW_DATA_DIR, exist_ok=True)
os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
os.makedirs(INTERIM_DATA_DIR, exist_ok=True)
os.makedirs(EDA_REPORTS_DIR, exist_ok=True)
os.makedirs(HYPOTHESIS_REPORTS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(INTERPRETATIONS_DIR, exist_ok=True)


# --- Column Names (Ensure these exactly match your data's column names) ---
# Main target and key financial columns
TOTAL_PREMIUM_COL = 'TotalPremium'
TOTAL_CLAIMS_COL = 'TotalClaims'
TRANSACTION_MONTH_COL = 'TransactionMonth'

# Numerical Features
NUMERICAL_FEATURES = [
    'CalculatedPremiumPerTerm',
    'RegistrationYear',
    'Cylinders',
    'cubiccapacity',
    'kilowatts',
    'NumberOfDoors',
    'CustomValueEstimate',
    'CapitalOutstanding',
    'NumberOfVehiclesInFleet',
    'SumInsured',
    'TermFrequency'
]

# Categorical Features
CATEGORICAL_FEATURES = [
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
    'Province',
    'PostalCode', # Could be numerical or categorical, depends on use
    'MainCrestaZone',
    'SubCrestaZone',
    'ItemType',
    'mmcode', # Often treated as categorical
    'VehicleType',
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
    'ExcessSelected',
    'CoverCategory',
    'CoverType',
    'CoverGroup',
    'Section',
    'Product',
    'StatutoryClass',
    'StatutoryRiskType'
]

# Date Features (columns that need to be parsed as datetime objects)
DATE_FEATURES = [
    'TransactionMonth',
    'VehicleIntroDate'
]

# --- Specific columns for certain plots/analyses ---
PROVINCE_COL = 'Province'
GENDER_COL = 'Gender'
VEHICLE_TYPE_COL = 'VehicleType'
CUSTOM_VALUE_ESTIMATE_COL = 'CustomValueEstimate'
REGISTRATION_YEAR_COL = 'RegistrationYear' # Already in numerical, but good to have explicit name
SUM_INSURED_COL = 'SumInsured'
CALCULATED_PREMIUM_PER_TERM_COL = 'CalculatedPremiumPerTerm'
POSTAL_CODE_COL = 'PostalCode'