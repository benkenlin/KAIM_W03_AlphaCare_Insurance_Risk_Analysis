# src/hypothesis_testing.py
import pandas as pd
import numpy as np
from scipy import stats
from statsmodels.stats.proportion import proportions_ztest
from src.config import (
    TOTAL_PREMIUM_COL, TOTAL_CLAIMS_COL, HAS_CLAIM_COL,
    CLAIM_SEVERITY_COL, MARGIN_COL
)

def calculate_kpis(df):
    """
    Calculates Claim Frequency, Claim Severity, and Margin per policy.
    Assumes 'HasClaim' column (0/1) exists from preprocessing.
    """
    df_kpis = df.copy()

    # Ensure numeric types for calculations and handle potential NaNs from coerce
    df_kpis[TOTAL_PREMIUM_COL] = pd.to_numeric(df_kpis[TOTAL_PREMIUM_COL], errors='coerce').fillna(0)
    df_kpis[TOTAL_CLAIMS_COL] = pd.to_numeric(df_kpis[TOTAL_CLAIMS_COL], errors='coerce').fillna(0)
    df_kpis[HAS_CLAIM_COL] = pd.to_numeric(df_kpis[HAS_CLAIM_COL], errors='coerce').fillna(0).astype(int)


    # Claim Severity: the average amount of a claim, given a claim occurred
    # For calculation per policy: TotalClaims / 1 if HasClaim, else 0 or NaN
    # We will use TOTAL_CLAIMS_COL directly and filter for HasClaim=1 during test.
    # For now, initialize CLAIM_SEVERITY_COL for all policies; its mean will be taken on filtered data.
    df_kpis[CLAIM_SEVERITY_COL] = np.where(
        df_kpis[HAS_CLAIM_COL] == 1,
        df_kpis[TOTAL_CLAIMS_COL], # If there's a claim, the claim amount is the severity for that policy
        0 # If no claim, severity is 0 for that policy.
    )

    # Margin: (TotalPremium - TotalClaims)
    df_kpis[MARGIN_COL] = df_kpis[TOTAL_PREMIUM_COL] - df_kpis[TOTAL_CLAIMS_COL]

    return df_kpis

def perform_t_test(group1_data, group2_data, alpha=0.05):
    """
    Performs an independent two-sample t-test.
    Returns t-statistic, p-value, and whether to reject H0.
    """
    # Filter out NaNs if any, before testing
    group1_data = group1_data.dropna()
    group2_data = group2_data.dropna()

    if len(group1_data) < 2 or len(group2_data) < 2:
        return np.nan, np.nan, "Not enough data for test in one or both groups"

    t_stat, p_value = stats.ttest_ind(group1_data, group2_data, equal_var=False) # Welch's t-test, more robust
    reject_h0 = p_value < alpha
    return t_stat, p_value, reject_h0

def perform_chi_squared_test(group1_successes, group1_nobs, group2_successes, group2_nobs, alpha=0.05):
    """
    Performs a two-sample proportions z-test (approximated by Chi-squared for binary outcomes).
    Returns z-statistic, p-value, and whether to reject H0.
    """
    count = np.array([group1_successes, group2_successes])
    nobs = np.array([group1_nobs, group2_nobs])

    # Check for zero observations or success counts that make the test invalid
    if nobs[0] == 0 or nobs[1] == 0 or success_A > nobs[0] or success_B > nobs[1]:
        return np.nan, np.nan, "Not enough valid observations/successes in one or both groups for chi-squared"

    try:
        stat, p_value = proportions_ztest(count, nobs)
    except ValueError as e: # Catch potential errors like division by zero if all success/nobs are 0
        print(f"Error in proportions_ztest: {e}. Counts: {count}, Nobs: {nobs}")
        return np.nan, np.nan, "Error during test calculation"

    reject_h0 = p_value < alpha
    return stat, p_value, reject_h0

def run_hypothesis_test(df, feature_col, group_A_val, group_B_val, metric_type, alpha=0.05):
    """
    Performs a hypothesis test for a given feature and metric.
    Args:
        df (pd.DataFrame): The input DataFrame with KPIs calculated.
        feature_col (str): The column representing the feature to test (e.g., 'Province', 'Gender').
        group_A_val: The value defining Group A (control).
        group_B_val: The value defining Group B (test).
        metric_type (str): 'ClaimFrequency', 'ClaimSeverity', or 'Margin'.
        alpha (float): Significance level.
    Returns:
        dict: Results including means, p-value, and H0 rejection status.
    """
    results = {
        "hypothesis": f"H0: No difference in {metric_type} between {feature_col}={group_A_val} and {feature_col}={group_B_val}",
        "group_A": group_A_val,
        "group_B": group_B_val,
        "metric": metric_type
    }

    group_A = df[df[feature_col] == group_A_val]
    group_B = df[df[feature_col] == group_B_val]

    if group_A.empty or group_B.empty:
        results["error"] = "One or both groups are empty for the specified feature values. Cannot perform test."
        return results

    if metric_type == "ClaimFrequency":
        n_A = len(group_A)
        success_A = group_A[HAS_CLAIM_COL].sum()
        n_B = len(group_B)
        success_B = group_B[HAS_CLAIM_COL].sum()

        mean_A = success_A / n_A if n_A > 0 else 0
        mean_B = success_B / n_B if n_B > 0 else 0

        stat, p_value, reject_h0 = perform_chi_squared_test(success_A, n_A, success_B, n_B, alpha)

    elif metric_type == "ClaimSeverity":
        # For severity, only consider policies that actually had claims
        severity_A_data = group_A[group_A[HAS_CLAIM_COL] == 1][TOTAL_CLAIMS_COL]
        severity_B_data = group_B[group_B[HAS_CLAIM_COL] == 1][TOTAL_CLAIMS_COL]

        mean_A = severity_A_data.mean() if not severity_A_data.empty else np.nan
        mean_B = severity_B_data.mean() if not severity_B_data.empty else np.nan

        stat, p_value, reject_h0 = perform_t_test(severity_A_data, severity_B_data, alpha)

    elif metric_type == "Margin":
        mean_A = group_A[MARGIN_COL].mean()
        mean_B = group_B[MARGIN_COL].mean()
        stat, p_value, reject_h0 = perform_t_test(group_A[MARGIN_COL], group_B[MARGIN_COL], alpha)

    else:
        results["error"] = f"Unknown metric type: {metric_type}"
        return results

    results["mean_A"] = mean_A
    results["mean_B"] = mean_B
    results["test_stat"] = stat
    results["p_value"] = p_value
    results["reject_h0"] = reject_h0
    results["significance_level"] = alpha

    return results