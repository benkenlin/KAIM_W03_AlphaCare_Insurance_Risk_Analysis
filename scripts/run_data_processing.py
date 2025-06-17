# scripts/run_data_processing.py

import os
import sys
import pandas as pd

# --- HYPER DEBUGGING PRINTS START ---
print("\n--- HYPER DEBUGGING PATH INFORMATION ---")
print(f"Current working directory (os.getcwd()): {os.getcwd()}")

# Calculate project_root
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, '..'))
print(f"Calculated script_dir: {script_dir}")
print(f"Calculated project_root: {project_root}")

# Add project_root to sys.path if not already there
if project_root not in sys.path:
    sys.path.insert(0, project_root)
    print(f"ACTION: Added project root '{project_root}' to sys.path.")
else:
    print(f"Project root '{project_root}' was already in sys.path.")


print("\n--- Current sys.path contents (first 5 entries) ---")
for i, p in enumerate(sys.path[:5]): # Print top 5 paths
    print(f"  sys.path[{i}]: {p}")
print("------------------------------------------\n")

# Check if 'src' directory exists at the project_root
src_path = os.path.join(project_root, 'src')
print(f"Checking for 'src' directory at: {src_path}")
if os.path.isdir(src_path):
    print(f"SUCCESS: 'src' directory FOUND at {src_path}")
    print(f"Contents of '{src_path}':")
    try:
        for item in os.listdir(src_path):
            print(f"  - {item}")
    except Exception as e:
        print(f"  Error listing contents of src: {e}")
else:
    print(f"ERROR: 'src' directory NOT FOUND at {src_path}")
    print("This is likely the problem. Ensure 'src' folder exists directly in your project root.")
    # Exit here if src is not found, no point in continuing
    sys.exit(1) # Exit with an error code

print("\n--- Attempting Imports Now ---")
# --- HYPER DEBUGGING PRINTS END ---

# Now, attempt the imports
# IMPORTANT: These imports MUST come *after* the sys.path modification and checks
try:
    from src.data_preparation import load_raw_data, clean_and_prepare_data
    from src.config import RAW_DATA_PATH, PROCESSED_DATA_PATH
    print("SUCCESS: 'src.data_preparation' and 'src.config' imported successfully!")
except ModuleNotFoundError as e:
    print(f"CRITICAL ERROR: ModuleNotFoundError caught: {e}")
    print("This means Python still can't find the module despite path adjustments.")
    print("Possible causes:")
    print("1. 'data_preparation.py' or '__init__.py' is missing or misspelled in 'src/'.")
    print("2. The calculated 'project_root' is incorrect, or 'src' is not a direct child.")
    print("3. Python interpreter caching (less likely but possible - restart terminal/IDE).")
    print("4. Virtual environment not active or corrupted.")
    sys.exit(1) # Exit with an error code
except Exception as e:
    print(f"An unexpected error occurred during import: {e}")
    sys.exit(1)


def main():
    """
    Main function to orchestrate the data loading, cleaning, and saving pipeline.
    """
    print("\nStarting data processing pipeline...")

    # 1. Load Raw Data
    df_raw = load_raw_data(RAW_DATA_PATH)

    if df_raw.empty:
        print("Raw data could not be loaded. Exiting data processing.")
        return

    # 2. Clean and Prepare Data
    df_processed = clean_and_prepare_data(df_raw)

    if df_processed.empty:
        print("Data processing resulted in an empty DataFrame. Exiting.")
        return

    # 3. Save Processed Data
    os.makedirs(os.path.dirname(PROCESSED_DATA_PATH), exist_ok=True)
    df_processed.to_parquet(PROCESSED_DATA_PATH, index=False)
    print(f"Processed data saved to: {PROCESSED_DATA_PATH}")

    print("Data processing pipeline completed successfully.")

if __name__ == "__main__":
    main()