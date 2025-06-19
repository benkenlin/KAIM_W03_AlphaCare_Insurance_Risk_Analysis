# src/statistical_testing.py

import pandas as pd
import numpy as np
from scipy.stats import ttest_ind, chi2_contingency
import statsmodels.api as sm # For Z-test if needed, or proportions test

# Import constants from config
from src.config import ALPHA, TOTAL_CLAIMS_COL, TOTAL_PREMIUM_COL, HAS_CLAIM_COL, MARGIN_COL, DRIVER_GENDER_COL, PROVINCE_COL, POSTAL_CODE_COL

def calculate_risk_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates Claim Frequency, Claim Severity, and Margin per policy.
    
    Args:
        df (pd.DataFrame): DataFrame containing policy data with 'TotalClaims',
                           'TotalPremium', and ideally 'PolicyID'.

    Returns:
        pd.DataFrame: DataFrame with calculated 'ClaimFrequency', 'ClaimSeverity',
                      and 'Margin' metrics.
    """
    df_metrics = df.copy()

    # Ensure necessary columns are numeric
    for col in [TOTAL_CLAIMS_COL, TOTAL_PREMIUM_COL]:
        if col in df_metrics.columns:
            df_metrics[col] = pd.to_numeric(df_metrics[col], errors='coerce').fillna(0)
        else:
            print(f"Warning: Column '{col}' not found for risk metric calculation. Assuming 0.")
            df_metrics[col] = 0

    # Claim Frequency: proportion of policies with at least one claim
    df_metrics[HAS_CLAIM_COL] = (df_metrics[TOTAL_CLAIMS_COL] > 0).astype(int)
    # Claim Frequency can be aggregated as mean of HasClaim

    # Claim Severity: average amount of a claim, given a claim occurred
    # Create a column for claim amount only when a claim exists
    df_metrics['ClaimAmountWhenClaimed'] = df_metrics[TOTAL_CLAIMS_COL].copy()
    df_metrics.loc[df_metrics[HAS_CLAIM_COL] == 0, 'ClaimAmountWhenClaimed'] = np.nan
    # Claim Severity can be aggregated as mean of ClaimAmountWhenClaimed

    # Margin: (TotalPremium - TotalClaims)
    df_metrics[MARGIN_COL] = df_metrics[TOTAL_PREMIUM_COL] - df_metrics[TOTAL_CLAIMS_COL]

    return df_metrics

def perform_chi_squared_test(group1_counts: pd.Series, group2_counts: pd.Series, alpha: float = ALPHA, hypothesis_text: str = ""):
    """
    Performs a Chi-squared test for independence between two categorical variables,
    typically used for comparing proportions/frequencies.

    Args:
        group1_counts (pd.Series): Counts for categories in Group 1 (e.g., [claims_A, no_claims_A])
        group2_counts (pd.Series): Counts for categories in Group 2 (e.g., [claims_B, no_claims_B])
        alpha (float): Significance level.
        hypothesis_text (str): Descriptive text for the hypothesis being tested.
    """
    contingency_table = pd.DataFrame({
        'Group1': group1_counts,
        'Group2': group2_counts
    }, index=['Claimed', 'No Claimed']) # Assuming the input series are in this order

    print(f"\n--- Hypothesis Test: {hypothesis_text} (Chi-squared Test) ---")
    print("Contingency Table:")
    print(contingency_table)

    chi2, p_value, dof, expected = chi2_contingency(contingency_table)

    print(f"Chi-squared Statistic: {chi2:.4f}")
    print(f"P-value: {p_value:.4f}")
    print(f"Degrees of Freedom: {dof}")

    if p_value < alpha:
        print(f"Result: Reject the Null Hypothesis (p < {alpha}). There IS a statistically significant difference.")
    else:
        print(f"Result: Fail to reject the Null Hypothesis (p >= {alpha}). There is NO statistically significant difference.")
    print("------------------------------------------------------------------")
    return p_value

def perform_t_test(group1_data: pd.Series, group2_data: pd.Series, alpha: float = ALPHA, hypothesis_text: str = ""):
    """
    Performs an independent samples t-test. Used for comparing means of two groups
    on a numerical variable (e.g., Claim Severity, Margin).

    Args:
        group1_data (pd.Series): Numerical data for Group 1.
        group2_data (pd.Series): Numerical data for Group 2.
        alpha (float): Significance level.
        hypothesis_text (str): Descriptive text for the hypothesis being tested.
    """
    # Remove NaNs before t-test
    group1_data = group1_data.dropna()
    group2_data = group2_data.dropna()

    if group1_data.empty or group2_data.empty:
        print(f"\n--- Hypothesis Test: {hypothesis_text} (T-Test) ---")
        print("Warning: One or both groups are empty after dropping NaNs. Cannot perform t-test.")
        print("------------------------------------------------------------------")
        return np.nan

    print(f"\n--- Hypothesis Test: {hypothesis_text} (Independent Samples T-Test) ---")
    print(f"Group 1 (n={len(group1_data)}): Mean = {group1_data.mean():.4f}, Std = {group1_data.std():.4f}")
    print(f"Group 2 (n={len(group2_data)}): Mean = {group2_data.mean():.4f}, Std = {group2_data.std():.4f}")

    # equal_var=False for Welch's t-test, which does not assume equal population variance
    t_stat, p_value = ttest_ind(group1_data, group2_data, equal_var=False)

    print(f"T-statistic: {t_stat:.4f}")
    print(f"P-value: {p_value:.4f}")

    if p_value < alpha:
        print(f"Result: Reject the Null Hypothesis (p < {alpha}). There IS a statistically significant difference.")
    else:
        print(f"Result: Fail to reject the Null Hypothesis (p >= {alpha}). There is NO statistically significant difference.")
    print("------------------------------------------------------------------")
    return p_value

def perform_z_test_proportions(count1, nobs1, count2, nobs2, alpha: float = ALPHA, hypothesis_text: str = ""):
    """
    Performs a Z-test for comparing two independent proportions.
    Suitable for comparing Claim Frequencies (proportions of claims).

    Args:
        count1 (int): Number of successes (claims) in group 1.
        nobs1 (int): Total number of observations in group 1.
        count2 (int): Number of successes (claims) in group 2.
        nobs2 (int): Total number of observations in group 2.
        alpha (float): Significance level.
        hypothesis_text (str): Descriptive text for the hypothesis being tested.
    """
    if nobs1 == 0 or nobs2 == 0:
        print(f"\n--- Hypothesis Test: {hypothesis_text} (Z-Test for Proportions) ---")
        print("Warning: One or both group sizes are zero. Cannot perform Z-test.")
        print("------------------------------------------------------------------")
        return np.nan

    from statsmodels.stats.proportion import proportions_ztest
    
    # Ensure counts are integers
    counts = np.array([int(count1), int(count2)])
    nobs = np.array([int(nobs1), int(nobs2)])

    stat, p_value = proportions_ztest(counts, nobs)

    prop1 = count1 / nobs1
    prop2 = count2 / nobs2

    print(f"\n--- Hypothesis Test: {hypothesis_text} (Z-Test for Proportions) ---")
    print(f"Group 1 Proportion: {prop1:.4f} (Claims: {count1}, Total: {nobs1})")
    print(f"Group 2 Proportion: {prop2:.4f} (Claims: {count2}, Total: {nobs2})")
    print(f"Z-statistic: {stat:.4f}")
    print(f"P-value: {p_value:.4f}")

    if p_value < alpha:
        print(f"Result: Reject the Null Hypothesis (p < {alpha}). There IS a statistically significant difference in proportions.")
    else:
        print(f"Result: Fail to reject the Null Hypothesis (p >= {alpha}). There is NO statistically significant difference in proportions.")
    print("------------------------------------------------------------------")
    return p_value


if __name__ == "__main__":
    print("This is the statistical_testing.py module. It defines functions for hypothesis testing.")
    print("It should not be run directly. Please use scripts/run_hypothesis.py to execute tests.")

