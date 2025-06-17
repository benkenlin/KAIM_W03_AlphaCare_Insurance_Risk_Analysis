# src/eda.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from io import StringIO
from src.utils import calculate_loss_ratio, save_plot
from src.config import (
    EDA_REPORTS_DIR, TOTAL_PREMIUM_COL, TOTAL_CLAIMS_COL, TRANSACTION_MONTH_COL,
    PROVINCE_COL, POSTAL_CODE_COL, GENDER_COL, VEHICLE_TYPE_COL, CUSTOM_VALUE_ESTIMATE_COL,
    REGISTRATION_YEAR_COL, SUM_INSURED_COL, CALCULATED_PREMIUM_PER_TERM_COL
)

def get_descriptive_stats(df, numerical_cols, output_dir=EDA_REPORTS_DIR):
    """Calculates and prints/saves descriptive statistics for numerical columns."""
    numerical_cols_present = [col for col in numerical_cols if col in df.columns and pd.api.types.is_numeric_dtype(df[col])]
    if not numerical_cols_present:
        print("No valid numerical columns found for descriptive statistics.")
        return

    stats_df = df[numerical_cols_present].describe()
    variability_df = df[numerical_cols_present].agg(['std', 'var'])
    
    print("Descriptive Statistics for Numerical Features:")
    print(stats_df)
    print("\nVariability (Std Dev, Variance):")
    print(variability_df)

    output_string = StringIO()
    output_string.write("Descriptive Statistics for Numerical Features:\n")
    stats_df.to_string(buf=output_string)
    output_string.write("\n\nVariability (Std Dev, Variance):\n")
    variability_df.to_string(buf=output_string)
    
    with open(os.path.join(output_dir, 'descriptive_stats.txt'), 'w') as f:
        f.write(output_string.getvalue())

def check_data_types(df, output_dir=EDA_REPORTS_DIR):
    """Reviews and prints/saves data types of all columns."""
    buffer = StringIO()
    df.info(buf=buffer)
    info_str = buffer.getvalue()
    
    print("\nColumn Data Types:")
    print(info_str)

    with open(os.path.join(output_dir, 'data_types.txt'), 'w') as f:
        f.write(info_str)

def check_missing_values(df, output_dir=EDA_REPORTS_DIR):
    """Identifies and quantifies missing values."""
    missing_data = df.isnull().sum()
    missing_percentage = (df.isnull().sum() / len(df)) * 100
    missing_table = pd.DataFrame({
        'Missing Count': missing_data,
        'Missing Percentage (%)': missing_percentage
    })
    missing_table = missing_table[missing_table['Missing Count'] > 0].sort_values(by='Missing Count', ascending=False)
    
    print("\nMissing Values:")
    print(missing_table)

    if not missing_table.empty:
        with open(os.path.join(output_dir, 'missing_values.txt'), 'w') as f:
            f.write("Missing Values:\n")
            f.write(missing_table.to_string())

def plot_univariate_distribution(df, column, is_categorical=False, bins=30, output_dir=EDA_REPORTS_DIR):
    """Plots and saves univariate distribution for a given column."""
    if column not in df.columns:
        print(f"Column '{column}' not found for univariate distribution plot.")
        return
    
    if not is_categorical and not pd.api.types.is_numeric_dtype(df[column]):
        print(f"Skipping univariate plot for '{column}': Not a numerical column.")
        return

    fig, ax = plt.subplots(figsize=(10, 6))
    if is_categorical:
        if df[column].nunique() > 50:
            print(f"Skipping countplot for '{column}' due to too many unique categories ({df[column].nunique()}).")
            plt.close(fig)
            return
        sns.countplot(data=df, x=column, order=df[column].value_counts().index, ax=ax)
        ax.set_title(f'Distribution of {column}')
        ax.tick_params(axis='x', labelrotation=45)

    else:
        sns.histplot(df[column], kde=True, bins=bins, ax=ax)
        ax.set_title(f'Distribution of {column}')
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    save_plot(fig, f'{column}_distribution.png', output_dir)

def plot_bivariate_relationships(df, x_col, y_col, hue_col=None, plot_type='scatter', output_dir=EDA_REPORTS_DIR):
    """Plots and saves bivariate relationships between columns."""
    if x_col not in df.columns or y_col not in df.columns:
        print(f"Skipping bivariate plot: '{x_col}' or '{y_col}' not found.")
        return
    if hue_col and hue_col not in df.columns:
        print(f"Skipping bivariate plot: hue_col '{hue_col}' not found.")
        hue_col = None

    if plot_type == 'scatter' and (not pd.api.types.is_numeric_dtype(df[x_col]) or not pd.api.types.is_numeric_dtype(df[y_col])):
        print(f"Skipping scatter plot for non-numerical columns: {x_col}, {y_col}")
        return
    
    fig, ax = plt.subplots(figsize=(12, 7))
    if plot_type == 'scatter':
        sns.scatterplot(data=df, x=x_col, y=y_col, hue=hue_col, alpha=0.6, ax=ax)
        ax.set_title(f'Relationship between {x_col} and {y_col}')
    elif plot_type == 'boxplot':
        sns.boxplot(data=df, x=x_col, y=y_col, ax=ax)
        ax.set_title(f'{y_col} Distribution by {x_col}')
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    save_plot(fig, f'{x_col}_vs_{y_col}_bivariate.png', output_dir)

def plot_loss_ratio_by_segment(df, segment_col, output_dir=EDA_REPORTS_DIR):
    """Calculates and plots loss ratio by a given segment."""
    if segment_col not in df.columns:
        print(f"Column '{segment_col}' not found for loss ratio plot.")
        return
    if df[segment_col].nunique() > 50:
        print(f"Skipping loss ratio plot for '{segment_col}' due to too many unique categories ({df[segment_col].nunique()}).")
        return

    if TOTAL_CLAIMS_COL not in df.columns or TOTAL_PREMIUM_COL not in df.columns:
        print(f"Skipping loss ratio plot: '{TOTAL_CLAIMS_COL}' or '{TOTAL_PREMIUM_COL}' not found.")
        return

    df_temp = df.copy()
    df_temp[TOTAL_CLAIMS_COL] = pd.to_numeric(df_temp[TOTAL_CLAIMS_COL], errors='coerce').fillna(0)
    df_temp[TOTAL_PREMIUM_COL] = pd.to_numeric(df_temp[TOTAL_PREMIUM_COL], errors='coerce').fillna(0)

    segment_data = df_temp.groupby(segment_col).agg(
        TotalClaims=(TOTAL_CLAIMS_COL, 'sum'),
        TotalPremium=(TOTAL_PREMIUM_COL, 'sum')
    ).reset_index()
    segment_data['LossRatio'] = segment_data.apply(lambda row: calculate_loss_ratio(row['TotalClaims'], row['TotalPremium']), axis=1)

    fig, ax = plt.subplots(figsize=(12, 7))
    # Corrected for FutureWarning: Assign the x variable to hue and set legend=False
    sns.barplot(data=segment_data, x=segment_col, y='LossRatio', hue=segment_col, palette='viridis', legend=False, ax=ax)
    ax.set_title(f'Loss Ratio by {segment_col}')
    ax.set_ylabel(f'Loss Ratio ({TOTAL_CLAIMS_COL} / {TOTAL_PREMIUM_COL})')
    ax.tick_params(axis='x', labelrotation=45)
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    save_plot(fig, f'loss_ratio_by_{segment_col}.png', output_dir)
    return segment_data[['LossRatio', segment_col]]

def plot_temporal_trends(df, date_column, metric_column, output_dir=EDA_REPORTS_DIR):
    """Plots and saves temporal trends for a given metric."""
    if date_column not in df.columns or metric_column not in df.columns:
        print(f"Skipping temporal trend plot: '{date_column}' or '{metric_column}' not found.")
        return

    df_temp = df.copy()
    df_temp[date_column] = pd.to_datetime(df_temp[date_column], errors='coerce')
    df_temp.dropna(subset=[date_column], inplace=True)
    df_temp[metric_column] = pd.to_numeric(df_temp[metric_column], errors='coerce').fillna(0)

    if df_temp.empty:
        print(f"No valid date data in '{date_column}' for temporal trend plot.")
        return

    df_monthly = df_temp.set_index(date_column).resample('M')[metric_column].sum().reset_index()
    
    fig, ax = plt.subplots(figsize=(14, 7))
    sns.lineplot(data=df_monthly, x=date_column, y=metric_column, marker='o', ax=ax)
    ax.set_title(f'Monthly Trend of {metric_column}')
    ax.set_xlabel('Date')
    ax.set_ylabel(metric_column)
    ax.grid(True, linestyle='--', alpha=0.7)
    save_plot(fig, f'monthly_trend_{metric_column}.png', output_dir)

def plot_monthly_trends_by_postal_code(df, output_dir=EDA_REPORTS_DIR, top_n_postal_codes=5):
    """
    Aggregates TotalPremium and TotalClaims by TransactionMonth and PostalCode
    and plots their monthly trends for top N postal codes.
    """
    required_cols = [TRANSACTION_MONTH_COL, TOTAL_PREMIUM_COL, TOTAL_CLAIMS_COL, POSTAL_CODE_COL]
    if not all(col in df.columns for col in required_cols):
        missing = [col for col in required_cols if col not in df.columns]
        print(f"Skipping monthly trends by postal code: Missing required columns: {missing}")
        return

    df_temp = df.copy()
    df_temp[TRANSACTION_MONTH_COL] = pd.to_datetime(df_temp[TRANSACTION_MONTH_COL], errors='coerce')
    df_temp.dropna(subset=[TRANSACTION_MONTH_COL], inplace=True)
    df_temp[TOTAL_PREMIUM_COL] = pd.to_numeric(df_temp[TOTAL_PREMIUM_COL], errors='coerce').fillna(0)
    df_temp[TOTAL_CLAIMS_COL] = pd.to_numeric(df_temp[TOTAL_CLAIMS_COL], errors='coerce').fillna(0)

    if df_temp.empty:
        print("No valid data for monthly trends by postal code after cleaning.")
        return

    # Aggregate by month and postal code
    monthly_data = df_temp.groupby([
        df_temp[TRANSACTION_MONTH_COL].dt.to_period('M'), # Aggregate to YYYY-MM period
        POSTAL_CODE_COL
    ]).agg(
        MonthlyPremium=(TOTAL_PREMIUM_COL, 'sum'),
        MonthlyClaims=(TOTAL_CLAIMS_COL, 'sum')
    ).reset_index()
    
    monthly_data['LossRatio'] = monthly_data.apply(
        lambda row: calculate_loss_ratio(row['MonthlyClaims'], row['MonthlyPremium']), axis=1
    )
    monthly_data[TRANSACTION_MONTH_COL] = monthly_data[TRANSACTION_MONTH_COL].dt.to_timestamp() # Convert period to timestamp for plotting

    # Identify top N postal codes by total premium for better visualization
    top_postal_codes = df_temp.groupby(POSTAL_CODE_COL)[TOTAL_PREMIUM_COL].sum().nlargest(top_n_postal_codes).index.tolist()
    
    # Filter data for top postal codes
    filtered_monthly_data = monthly_data[monthly_data[POSTAL_CODE_COL].isin(top_postal_codes)].copy()

    if filtered_monthly_data.empty:
        print(f"No data for top {top_n_postal_codes} postal codes after aggregation.")
        return

    # Plotting Total Premium by Month for Top Postal Codes
    fig1, ax1 = plt.subplots(figsize=(16, 8))
    sns.lineplot(
        data=filtered_monthly_data,
        x=TRANSACTION_MONTH_COL,
        y='MonthlyPremium',
        hue=POSTAL_CODE_COL,
        marker='o',
        ax=ax1
    )
    ax1.set_title(f'Monthly Total Premium by Postal Code (Top {top_n_postal_codes})')
    ax1.set_xlabel('Month')
    ax1.set_ylabel('Total Premium')
    ax1.grid(True, linestyle='--', alpha=0.7)
    ax1.tick_params(axis='x', labelrotation=45)
    save_plot(fig1, f'monthly_premium_by_{POSTAL_CODE_COL}.png', output_dir)

    # Plotting Total Claims by Month for Top Postal Codes
    fig2, ax2 = plt.subplots(figsize=(16, 8))
    sns.lineplot(
        data=filtered_monthly_data,
        x=TRANSACTION_MONTH_COL,
        y='MonthlyClaims',
        hue=POSTAL_CODE_COL,
        marker='o',
        ax=ax2
    )
    ax2.set_title(f'Monthly Total Claims by Postal Code (Top {top_n_postal_codes})')
    ax2.set_xlabel('Month')
    ax2.set_ylabel('Total Claims')
    ax2.grid(True, linestyle='--', alpha=0.7)
    ax2.tick_params(axis='x', labelrotation=45)
    save_plot(fig2, f'monthly_claims_by_{POSTAL_CODE_COL}.png', output_dir)

    # Plotting Loss Ratio by Month for Top Postal Codes
    fig3, ax3 = plt.subplots(figsize=(16, 8))
    sns.lineplot(
        data=filtered_monthly_data,
        x=TRANSACTION_MONTH_COL,
        y='LossRatio',
        hue=POSTAL_CODE_COL,
        marker='o',
        ax=ax3
    )
    ax3.set_title(f'Monthly Loss Ratio by Postal Code (Top {top_n_postal_codes})')
    ax3.set_xlabel('Month')
    ax3.set_ylabel('Loss Ratio')
    ax3.grid(True, linestyle='--', alpha=0.7)
    ax3.tick_params(axis='x', labelrotation=45)
    save_plot(fig3, f'monthly_loss_ratio_by_{POSTAL_CODE_COL}.png', output_dir)


def detect_outliers_boxplot(df, column, output_dir=EDA_REPORTS_DIR):
    """Uses box plot to detect outliers in a numerical column and saves the plot."""
    if column not in df.columns or not pd.api.types.is_numeric_dtype(df[column]):
        print(f"Skipping outlier detection: '{column}' not found or not numerical.")
        return

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.boxplot(y=df[column], ax=ax)
    ax.set_title(f'Box Plot of {column} to Detect Outliers')
    ax.set_ylabel(column)
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    save_plot(fig, f'outliers_{column}_boxplot.png', output_dir)

def generate_eda_report(df, numerical_cols, categorical_cols, date_cols, output_dir=EDA_REPORTS_DIR):
    """Generates a comprehensive EDA report with summary statistics and plots."""
    print("Starting EDA Report Generation...")
    
    # 1. Convert columns to appropriate types before EDA
    for col in numerical_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    for col in date_cols:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors='coerce')

    # 2. Run core EDA functions
    get_descriptive_stats(df, numerical_cols, output_dir)
    check_data_types(df, output_dir)
    check_missing_values(df, output_dir)

    # 3. Plotting univariate distributions
    for col in numerical_cols:
        plot_univariate_distribution(df, col, is_categorical=False, output_dir=output_dir)
    for col in categorical_cols:
        plot_univariate_distribution(df, col, is_categorical=True, output_dir=output_dir)

    # 4. Plotting loss ratio by segments
    relevant_segments = [PROVINCE_COL, GENDER_COL, VEHICLE_TYPE_COL]
    for col in relevant_segments:
        plot_loss_ratio_by_segment(df, col, output_dir=output_dir)
    
    # 5. Temporal trends (overall)
    if TRANSACTION_MONTH_COL in df.columns:
        plot_temporal_trends(df, TRANSACTION_MONTH_COL, TOTAL_CLAIMS_COL, output_dir=output_dir)
        df_temp = df.copy()
        df_temp['HasClaim'] = (df_temp[TOTAL_CLAIMS_COL] > 0).astype(int)
        plot_temporal_trends(df_temp, TRANSACTION_MONTH_COL, 'HasClaim', output_dir=output_dir)
        del df_temp

    # 6. Specific Bivariate/Multivariate Analysis for PostalCode trends
    plot_monthly_trends_by_postal_code(df, output_dir=output_dir) # <--- NEW CALL

    # 7. Outlier detection
    key_numerical_cols = [TOTAL_PREMIUM_COL, TOTAL_CLAIMS_COL, CUSTOM_VALUE_ESTIMATE_COL, SUM_INSURED_COL, CALCULATED_PREMIUM_PER_TERM_COL]
    for col in key_numerical_cols:
        if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
            detect_outliers_boxplot(df, col, output_dir=output_dir)

    # 8. Correlation Matrix
    numerical_cols_for_corr = [col for col in numerical_cols if col in df.columns and pd.api.types.is_numeric_dtype(df[col])]
    if len(numerical_cols_for_corr) > 1:
        correlation_matrix = df[numerical_cols_for_corr].corr()
        fig, ax = plt.subplots(figsize=(12, 10))
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", ax=ax, linewidths=.5)
        ax.set_title('Correlation Matrix of Key Numerical Features')
        save_plot(fig, 'correlation_matrix.png', output_dir)
    else:
        print("Not enough numerical columns to generate a correlation matrix.")

    print(f"EDA Report generated and saved to {output_dir}")