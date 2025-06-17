# scripts/run_hypothesis_tests_pipeline.py
import pandas as pd
import os
from src.utils import load_data
from src.hypothesis_testing import (
    segment_data_by_feature,
    perform_hypothesis_test,
    generate_hypothesis_report_entry,
    check_group_equivalence,
    calculate_loss_ratio
)
from src.config import (
    RAW_DATA_PATH, HYPOTHESIS_REPORTS_DIR, ALPHA_HYPOTHESIS_TESTING,
    PROVINCE_COL, POSTAL_CODE_COL, GENDER_COL, TOTAL_CLAIMS_COL, TOTAL_PREMIUM_COL,
    NUMERICAL_FEATURES, CATEGORICAL_FEATURES
)

def run_hypothesis_tests():
    print("--- Starting Hypothesis Testing Pipeline ---")
    
    # Adjust 'sep' if your .txt file uses a different delimiter (e.g., ',')
    df = load_data(RAW_DATA_PATH, sep='\t')
    
    # Ensure necessary columns are numeric and fill NaNs for calculations
    for col in [TOTAL_CLAIMS_COL, TOTAL_PREMIUM_COL]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0) # Fill NaNs with 0 for calculations

    full_report_str = "## Hypothesis Testing Results Report\n\n"
    
    # Define features for equivalence check (exclude the feature being tested itself)
    # Filter NUMERICAL_FEATURES to exclude TOTAL_PREMIUM_COL and TOTAL_CLAIMS_COL
    # as these are outcome variables for the tests.
    equiv_numerical_cols = [col for col in NUMERICAL_FEATURES if col not in [TOTAL_PREMIUM_COL, TOTAL_CLAIMS_COL]]
    equiv_categorical_cols = [col for col in CATEGORICAL_FEATURES]


    # --- Hypothesis 1: No risk differences across provinces ---
    if PROVINCE_COL in df.columns and df[PROVINCE_COL].nunique() > 1:
        print("\nTesting Hypothesis 1: No risk differences across provinces")
        provinces = df[PROVINCE_COL].dropna().unique()
        # Limit the number of province pairs for demonstration/computation time
        if len(provinces) > 5: # Arbitrary limit
            print(f"Too many provinces ({len(provinces)}). Testing a subset of 5 for demonstration.")
            provinces_sample = df[PROVINCE_COL].value_counts().index[:5].tolist()
        else:
            provinces_sample = provinces

        for i in range(len(provinces_sample)):
            for j in range(i + 1, len(provinces_sample)):
                province1 = provinces_sample[i]
                province2 = provinces_sample[j]
                
                group_A_df, group_B_df, name_A, name_B = segment_data_by_feature(df, PROVINCE_COL, province1, province2)
                
                if group_A_df.empty or group_B_df.empty:
                    print(f"  Skipping {name_A} vs {name_B} due to empty groups.")
                    continue

                # Check group equivalence (important for validity)
                print(f"\nChecking equivalence for {name_A} vs {name_B}:")
                current_equiv_cat_cols = [col for col in equiv_categorical_cols if col != PROVINCE_COL]
                check_group_equivalence(group_A_df, group_B_df, equiv_numerical_cols, current_equiv_cat_cols)

                # Claim Frequency
                p_val_freq, test_stat_freq, decision_freq, interpretation_freq = perform_hypothesis_test(group_A_df, group_B_df, metric_type='claim_frequency', alpha=ALPHA_HYPOTHESIS_TESTING)
                
                # Claim Severity
                p_val_sev, test_stat_sev, decision_sev, interpretation_sev = perform_hypothesis_test(group_A_df, group_B_df, metric_type='claim_severity', alpha=ALPHA_HYPOTHESIS_TESTING)
                
                # Business Implication Calculation
                lr_A = calculate_loss_ratio(group_A_df[TOTAL_CLAIMS_COL].sum(), group_A_df[TOTAL_PREMIUM_COL].sum())
                lr_B = calculate_loss_ratio(group_B_df[TOTAL_CLAIMS_COL].sum(), group_B_df[TOTAL_PREMIUM_COL].sum())
                
                business_imp_freq = ""
                if decision_freq == "Reject Null Hypothesis":
                    diff_pct = (lr_A - lr_B) / lr_B * 100 if lr_B != 0 else (float('inf') if lr_A > 0 else 0)
                    business_imp_freq = f"{name_A} has a {'higher' if diff_pct > 0 else 'lower'} loss ratio ({lr_A:.2f}) than {name_B} ({lr_B:.2f}) by approximately {abs(diff_pct):.2f}% for claim frequency."
                    business_imp_freq += " This suggests regional risk adjustment to premiums may be warranted for claim frequency."

                business_imp_sev = ""
                # Ensure there are claims in both groups for severity calculation
                sev_A_df = group_A_df[group_A_df[TOTAL_CLAIMS_COL] > 0]
                sev_B_df = group_B_df[group_B_df[TOTAL_CLAIMS_COL] > 0]
                if not sev_A_df.empty and not sev_B_df.empty:
                    sev_A = sev_A_df[TOTAL_CLAIMS_COL].mean()
                    sev_B = sev_B_df[TOTAL_CLAIMS_COL].mean()
                    if sev_B != 0:
                        diff_pct = (sev_A - sev_B) / sev_B * 100
                    else:
                        diff_pct = float('inf')
                    
                    if decision_sev == "Reject Null Hypothesis":
                        business_imp_sev = f"Average claim severity in {name_A} ({sev_A:.2f}) is {'higher' if diff_pct > 0 else 'lower'} than in {name_B} ({sev_B:.2f}) by approx. {abs(diff_pct):.2f}%."
                        business_imp_sev += " This indicates significant regional differences in claim costs, supporting severity-based premium adjustments."

                full_report_str += generate_hypothesis_report_entry(
                    f"H1: Risk (Claim Frequency) Difference between {name_A} and {name_B}",
                    p_val_freq, test_stat_freq, decision_freq, interpretation_freq,
                    business_implication=business_imp_freq
                )
                full_report_str += generate_hypothesis_report_entry(
                    f"H1: Risk (Claim Severity) Difference between {name_A} and {name_B}",
                    p_val_sev, test_stat_sev, decision_sev, interpretation_sev,
                    business_implication=business_imp_sev
                )
    else:
        print(f"\nSkipping Hypothesis 1: '{PROVINCE_COL}' column not found or insufficient unique values.")

    # --- Hypothesis 2 & 3: Risk and Margin differences between zip codes ---
    if POSTAL_CODE_COL in df.columns and df[POSTAL_CODE_COL].nunique() > 1:
        print("\nTesting Hypotheses 2 & 3: Risk and Margin differences between zip codes")
        zip_codes_to_test = df[POSTAL_CODE_COL].value_counts().index[:5].tolist() # Top 5 for demonstration
        
        for i in range(len(zip_codes_to_test)):
            for j in range(i + 1, len(zip_codes_to_test)):
                zip1 = zip_codes_to_test[i]
                zip2 = zip_codes_to_test[j]

                group_A_df, group_B_df, name_A, name_B = segment_data_by_feature(df, POSTAL_CODE_COL, zip1, zip2)
                
                if group_A_df.empty or group_B_df.empty:
                    print(f"  Skipping {name_A} vs {name_B} due to empty groups.")
                    continue

                # Check group equivalence
                print(f"\nChecking equivalence for {name_A} vs {name_B}:")
                current_equiv_cat_cols = [col for col in equiv_categorical_cols if col != POSTAL_CODE_COL]
                check_group_equivalence(group_A_df, group_B_df, equiv_numerical_cols, current_equiv_cat_cols)

                # Claim Frequency
                p_val_freq, test_stat_freq, decision_freq, interpretation_freq = perform_hypothesis_test(group_A_df, group_B_df, metric_type='claim_frequency', alpha=ALPHA_HYPOTHESIS_TESTING)
                
                # Claim Severity
                p_val_sev, test_stat_sev, decision_sev, interpretation_sev = perform_hypothesis_test(group_A_df, group_B_df, metric_type='claim_severity', alpha=ALPHA_HYPOTHESIS_TESTING)
                
                # Margin
                p_val_margin, test_stat_margin, decision_margin, interpretation_margin = perform_hypothesis_test(group_A_df, group_B_df, metric_type='margin', alpha=ALPHA_HYPOTHESIS_TESTING)
                
                # Business Implications (simplified for example)
                business_imp_zip = ""
                if decision_freq == "Reject Null Hypothesis" or decision_sev == "Reject Null Hypothesis" or decision_margin == "Reject Null Hypothesis":
                    business_imp_zip = f"Significant risk or margin differences found between Postal Codes {name_A} and {name_B}, suggesting potential for micro-segmentation in pricing and marketing strategies."

                full_report_str += generate_hypothesis_report_entry(
                    f"H2: Risk (Claim Frequency) Difference between Zip {name_A} and Zip {name_B}",
                    p_val_freq, test_stat_freq, decision_freq, interpretation_freq,
                    business_implication=business_imp_zip
                )
                full_report_str += generate_hypothesis_report_entry(
                    f"H2: Risk (Claim Severity) Difference between Zip {name_A} and Zip {name_B}",
                    p_val_sev, test_stat_sev, decision_sev, interpretation_sev,
                    business_implication=business_imp_zip
                )
                full_report_str += generate_hypothesis_report_entry(
                    f"H3: Margin Difference between Zip {name_A} and Zip {name_B}",
                    p_val_margin, test_stat_margin, decision_margin, interpretation_margin,
                    business_implication=business_imp_zip
                )
    else:
        print(f"\nSkipping Hypotheses 2 & 3: '{POSTAL_CODE_COL}' column not found or insufficient unique values.")


    # --- Hypothesis 4: No significant risk difference between Women and Men ---
    if GENDER_COL in df.columns and df[GENDER_COL].nunique() == 2: # Assumes exactly two unique genders
        gender_values = df[GENDER_COL].dropna().unique()
        if len(gender_values) == 2:
            gender1, gender2 = gender_values[0], gender_values[1]
            print(f"\nTesting Hypothesis 4: No significant risk difference between {gender1} and {gender2}")
            group_A_df, group_B_df, name_A, name_B = segment_data_by_feature(df, GENDER_COL, gender1, gender2)
            
            if group_A_df.empty or group_B_df.empty:
                print(f"  Skipping {name_A} vs {name_B} due to empty groups.")
            else:
                # Check group equivalence
                print(f"\nChecking equivalence for {name_A} vs {name_B}:")
                current_equiv_cat_cols = [col for col in equiv_categorical_cols if col != GENDER_COL]
                check_group_equivalence(group_A_df, group_B_df, equiv_numerical_cols, current_equiv_cat_cols)

                # Claim Frequency
                p_val_freq, test_stat_freq, decision_freq, interpretation_freq = perform_hypothesis_test(group_A_df, group_B_df, metric_type='claim_frequency', alpha=ALPHA_HYPOTHESIS_TESTING)
                
                # Claim Severity
                p_val_sev, test_stat_sev, decision_sev, interpretation_sev = perform_hypothesis_test(group_A_df, group_B_df, metric_type='claim_severity', alpha=ALPHA_HYPOTHESIS_TESTING)
                
                # Business Implication
                business_imp_gender = ""
                if decision_freq == "Reject Null Hypothesis" or decision_sev == "Reject Null Hypothesis":
                    business_imp_gender = "Gender exhibits significant differences in risk, which can inform targeted marketing and product design, subject to regulatory compliance."

                full_report_str += generate_hypothesis_report_entry(
                    f"H4: Risk (Claim Frequency) Difference between {name_A} and {name_B}",
                    p_val_freq, test_stat_freq, decision_freq, interpretation_freq,
                    business_implication=business_imp_gender
                )
                full_report_str += generate_hypothesis_report_entry(
                    f"H4: Risk (Claim Severity) Difference between {name_A} and {name_B}",
                    p_val_sev, test_stat_sev, decision_sev, interpretation_sev,
                    business_implication=business_imp_gender
                )
        else:
            print(f"\nSkipping Hypothesis 4: '{GENDER_COL}' column does not have exactly two unique values for comparison.")
    else:
        print(f"\nSkipping Hypothesis 4: '{GENDER_COL}' column not found or not binary.")


    # Save the full report
    report_filepath = os.path.join(HYPOTHESIS_REPORTS_DIR, 'hypothesis_testing_report.txt')
    with open(report_filepath, 'w') as f:
        f.write(full_report_str)
    print(f"\n--- Hypothesis Testing Pipeline Completed. Report saved to {report_filepath} ---")

if __name__ == "__main__":
    run_hypothesis_tests()