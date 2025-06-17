# src/utils.py
import pandas as pd
import os
import matplotlib.pyplot as plt # Needed for save_plot
from io import StringIO
import csv # Needed for robust data loading

def load_data(file_path, sep='|'):
    """
    Loads data from a specified file path using csv module for robust parsing.
    Args:
        file_path (str): The path to the data file.
        sep (str): The delimiter used in the file. Defaults to '|' for pipe-separated.
    Returns:
        pd.DataFrame: The loaded DataFrame.
    Raises:
        FileNotFoundError: If the file_path does not exist.
        Exception: For other errors during file loading.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Data file not found at: {file_path}")
    try:
        data_rows = []
        columns = []

        # Open with newline='' to prevent csv.reader from mangling newlines
        with open(file_path, 'r', encoding='utf-8', newline='') as f:
            reader = csv.reader(f, delimiter=sep)

            # Read header
            try:
                header_line = next(reader)
                # Strip leading/trailing whitespace and BOM from each column name
                columns = [col.strip().lstrip('\ufeff') for col in header_line]
            except StopIteration:
                raise ValueError("File is empty or has no header.")

            # Read data rows
            for row in reader:
                # Ensure the row has the same number of columns as the header
                if len(row) == len(columns):
                    data_rows.append(row)
                else:
                    # Optional: Log rows that don't match column count
                    # print(f"Skipping malformed row: {row}")
                    pass
        
        # Create DataFrame from data_rows and columns
        df = pd.DataFrame(data_rows, columns=columns)

        print(f"Successfully loaded data from {file_path} with {len(df)} rows and {len(df.columns)} columns.")
        print("\n--- DEBUG: DataFrame Info (from utils.py) ---")
        df.info()
        print("\n--- DEBUG: First 5 rows of DataFrame (from utils.py) ---")
        print(df.head())
        print("\n--- DEBUG: DataFrame Columns (from utils.py) ---")
        print(df.columns.tolist())
        print("-----------------------------\n")

        return df
    except Exception as e:
        print(f"Error loading data from {file_path}: {e}")
        raise

def save_dataframe(df, file_path):
    """
    Saves a DataFrame to a specified file path.
    Args:
        df (pd.DataFrame): The DataFrame to save.
        file_path (str): The path to save the DataFrame (e.g., 'output.csv').
    """
    try:
        df.to_csv(file_path, index=False)
        print(f"DataFrame successfully saved to {file_path}")
    except Exception as e:
        print(f"Error saving DataFrame to {file_path}: {e}")

def save_plot(fig, file_name, output_dir):
    """
    Saves a matplotlib figure to a specified directory.
    Args:
        fig (matplotlib.figure.Figure): The figure to save.
        file_name (str): The name of the file to save (e.g., 'my_plot.png').
        output_dir (str): The directory where the plot will be saved.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    file_path = os.path.join(output_dir, file_name)
    fig.savefig(file_path, bbox_inches='tight')
    plt.close(fig) # Close the figure to free up memory
    print(f"Plot saved to {file_path}")

def calculate_loss_ratio(total_claims, total_premium):
    """
    Calculates the loss ratio.
    Args:
        total_claims (float): The total claims amount.
        total_premium (float): The total premium amount.
    Returns:
        float: The calculated loss ratio. Returns 0 if total_premium is 0 or NaN.
    """
    # Ensure total_premium is a numeric type that can be checked for NaN and compared to 0
    if pd.isna(total_premium) or total_premium == 0:
        return 0.0
    return total_claims / total_premium