# scripts/run_modeling_pipeline.py
import pandas as pd
import os
import joblib # For loading preprocessor and saving models

from src.utils import load_data, save_dataframe
from src.data_preprocessing import prepare_data_for_modeling, split_data, get_feature_names_after_preprocessing
from src.modeling import (
    train_linear_regression, train_decision_tree_regressor, train_random_forest_regressor, train_xgboost_regressor,
    train_decision_tree_classifier, train_random_forest_classifier, train_xgboost_classifier,
    save_model
)
from src.model_evaluation import evaluate_regression_model, evaluate_classification_model, compare_models
from src.model_interpretability import get_feature_importance_tree_models, explain_model_shap, report_influential_features
from src.config import (
    RAW_DATA_PATH, MODELS_DIR, INTERPRETATIONS_DIR, TEST_SIZE, RANDOM_STATE,
    NUMERICAL_FEATURES, CATEGORICAL_FEATURES, DATE_FEATURES, COLUMNS_TO_DROP, # <-- Added
    CLAIM_SEVERITY_TARGET, CLAIM_PROBABILITY_TARGET,
    RF_N_ESTIMATORS, XGB_N_ESTIMATORS, XGB_LEARNING_RATE,
    TOTAL_CLAIMS_COL, TOTAL_PREMIUM_COL, CUSTOM_VALUE_ESTIMATE_COL # Added for interpretability reporting
)

def main():
    print("--- Starting Predictive Modeling Pipeline ---")
    
    # Load raw data
    # Adjust 'sep' if your .txt file uses a different delimiter (e.g., ',')
    df = load_data(RAW_DATA_PATH, sep='\t')
    print(f"Loaded data from {RAW_DATA_PATH} with {len(df)} rows.")

    # Ensure TOTAL_CLAIMS_COL and TOTAL_PREMIUM_COL are numeric for calculations
    for col in [TOTAL_CLAIMS_COL, TOTAL_PREMIUM_COL]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0) # Fill NaNs with 0

    # --- Claim Severity Prediction (Regression) ---
    print("\n## Claim Severity Prediction (Regression)")
    X_sev, y_sev, preprocessor_sev, num_feats_sev_final, cat_feats_sev_final = prepare_data_for_modeling(
        df.copy(), # Work on a copy to avoid modifying original df
        numerical_cols=NUMERICAL_FEATURES,
        categorical_cols=CATEGORICAL_FEATURES,
        date_cols=DATE_FEATURES, # Pass date features
        cols_to_drop=COLUMNS_TO_DROP, # Pass columns to drop
        target_col=CLAIM_SEVERITY_TARGET,
        problem_type='regression'
    )

    if X_sev.empty or y_sev.empty:
        print("Skipping severity model: Not enough data after filtering for claims.")
    else:
        # Split data before fitting preprocessor to avoid data leakage
        X_train_sev, X_test_sev, y_train_sev, y_test_sev = split_data(
            X_sev, y_sev, test_size=TEST_SIZE, random_state=RANDOM_STATE
        )

        # Fit preprocessor on training data and transform both train/test
        preprocessor_sev.fit(X_train_sev)
        X_train_sev_processed = preprocessor_sev.transform(X_train_sev)
        X_test_sev_processed = preprocessor_sev.transform(X_test_sev)

        # Get feature names after preprocessing
        all_severity_feature_names = get_feature_names_after_preprocessing(
            preprocessor_sev, num_feats_sev_final, cat_feats_sev_final
        )
        # Convert processed arrays to DataFrames for better compatibility with some models (e.g., XGBoost, SHAP)
        X_train_sev_processed_df = pd.DataFrame(X_train_sev_processed, columns=all_severity_feature_names, index=X_train_sev.index)
        X_test_sev_processed_df = pd.DataFrame(X_test_sev_processed, columns=all_severity_feature_names, index=X_test_sev.index)


        # Train and Evaluate Regression Models
        regression_models = {}
        regression_results = {}

        print("\nTraining Linear Regression...")
        lr_model = train_linear_regression(X_train_sev_processed_df, y_train_sev)
        lr_preds = lr_model.predict(X_test_sev_processed_df)
        regression_models['Linear Regression'] = lr_model
        regression_results['Linear Regression'] = evaluate_regression_model(y_test_sev, lr_preds)

        print("\nTraining Decision Tree Regressor...")
        dt_model = train_decision_tree_regressor(X_train_sev_processed_df, y_train_sev)
        dt_preds = dt_model.predict(X_test_sev_processed_df)
        regression_models['Decision Tree'] = dt_model
        regression_results['Decision Tree'] = evaluate_regression_model(y_test_sev, dt_preds)
        get_feature_importance_tree_models(dt_model, all_severity_feature_names, model_name="Decision_Tree_Severity")

        print("\nTraining Random Forest Regressor...")
        rf_model = train_random_forest_regressor(X_train_sev_processed_df, y_train_sev, n_estimators=RF_N_ESTIMATORS)
        rf_preds = rf_model.predict(X_test_sev_processed_df)
        regression_models['Random Forest'] = rf_model
        regression_results['Random Forest'] = evaluate_regression_model(y_test_sev, rf_preds)
        get_feature_importance_tree_models(rf_model, all_severity_feature_names, model_name="Random_Forest_Severity")

        print("\nTraining XGBoost Regressor...")
        xgb_reg_model = train_xgboost_regressor(X_train_sev_processed_df, y_train_sev, n_estimators=XGB_N_ESTIMATORS, learning_rate=XGB_LEARNING_RATE)
        xgb_reg_preds = xgb_reg_model.predict(X_test_sev_processed_df)
        regression_models['XGBoost Regressor'] = xgb_reg_model
        regression_results['XGBoost Regressor'] = evaluate_regression_model(y_test_sev, xgb_reg_preds)
        get_feature_importance_tree_models(xgb_reg_model, all_severity_feature_names, model_name="XGBoost_Severity")

        best_reg_model_name, best_reg_metrics = compare_models(regression_results, problem_type='regression')
        if best_reg_model_name:
            best_reg_model = regression_models[best_reg_model_name]
            save_model(best_reg_model, os.path.join(MODELS_DIR, f'{best_reg_model_name.replace(" ", "_")}_severity_model.joblib'))
            save_model(preprocessor_sev, os.path.join(MODELS_DIR, 'preprocessor_severity.joblib'))

            # Model Interpretability for best regression model
            shap_importance_severity = explain_model_shap(best_reg_model, X_test_sev_processed_df, all_severity_feature_names, model_name=best_reg_model_name.replace(" ", "_") + "_Severity")
            if shap_importance_severity is not None:
                report_sev_str = report_influential_features(shap_importance_severity, prediction_type='claim severity')
                with open(os.path.join(INTERPRETATIONS_DIR, f'{best_reg_model_name.replace(" ", "_")}_severity_interpretations.txt'), 'w') as f:
                    f.write(report_sev_str)
        else:
            print("No best regression model found.")

    # --- Claim Probability Prediction (Classification) ---
    print("\n## Claim Probability Prediction (Classification)")
    X_class, y_class, preprocessor_class, num_feats_class_final, cat_feats_class_final = prepare_data_for_modeling(
        df.copy(),
        numerical_cols=NUMERICAL_FEATURES,
        categorical_cols=CATEGORICAL_FEATURES,
        date_cols=DATE_FEATURES, # Pass date features
        cols_to_drop=COLUMNS_TO_DROP, # Pass columns to drop
        target_col=CLAIM_PROBABILITY_TARGET,
        problem_type='classification'
    )
    
    if X_class.empty or y_class.empty or y_class.nunique() < 2:
        print("Skipping probability model: Not enough data or target classes for classification.")
    else:
        # Split data with stratification for classification
        X_train_class, X_test_class, y_train_class, y_test_class = split_data(
            X_class, y_class, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y_class
        )
        
        preprocessor_class.fit(X_train_class)
        X_train_class_processed = preprocessor_class.transform(X_train_class)
        X_test_class_processed = preprocessor_class.transform(X_test_class)

        all_class_feature_names = get_feature_names_after_preprocessing(
            preprocessor_class, num_feats_class_final, cat_feats_class_final
        )
        X_train_class_processed_df = pd.DataFrame(X_train_class_processed, columns=all_class_feature_names, index=X_train_class.index)
        X_test_class_processed_df = pd.DataFrame(X_test_class_processed, columns=all_class_feature_names, index=X_test_class.index)

        # Train and Evaluate Classification Models
        classification_models = {}
        classification_results = {}

        print("\nTraining Decision Tree Classifier...")
        dt_clf_model = train_decision_tree_classifier(X_train_class_processed_df, y_train_class)
        dt_clf_preds = dt_clf_model.predict(X_test_class_processed_df)
        dt_clf_proba = dt_clf_model.predict_proba(X_test_class_processed_df)
        classification_models['Decision Tree Classifier'] = dt_clf_model
        classification_results['Decision Tree Classifier'] = evaluate_classification_model(y_test_class, dt_clf_preds, dt_clf_proba)
        get_feature_importance_tree_models(dt_clf_model, all_class_feature_names, model_name="Decision_Tree_Classifier")

        print("\nTraining Random Forest Classifier...")
        rf_clf_model = train_random_forest_classifier(X_train_class_processed_df, y_train_class, n_estimators=RF_N_ESTIMATORS)
        rf_clf_preds = rf_clf_model.predict(X_test_class_processed_df)
        rf_clf_proba = rf_clf_model.predict_proba(X_test_class_processed_df)
        classification_models['Random Forest Classifier'] = rf_clf_model
        classification_results['Random Forest Classifier'] = evaluate_classification_model(y_test_class, rf_clf_preds, rf_clf_proba)
        get_feature_importance_tree_models(rf_clf_model, all_class_feature_names, model_name="Random_Forest_Classifier")

        print("\nTraining XGBoost Classifier...")
        xgb_clf_model = train_xgboost_classifier(X_train_class_processed_df, y_train_class, n_estimators=XGB_N_ESTIMATORS, learning_rate=XGB_LEARNING_RATE)
        xgb_clf_preds = xgb_clf_model.predict(X_test_class_processed_df)
        xgb_clf_proba = xgb_clf_model.predict_proba(X_test_class_processed_df)
        classification_models['XGBoost Classifier'] = xgb_clf_model
        classification_results['XGBoost Classifier'] = evaluate_classification_model(y_test_class, xgb_clf_preds, xgb_clf_proba)
        get_feature_importance_tree_models(xgb_clf_model, all_class_feature_names, model_name="XGBoost_Classifier")

        best_clf_model_name, best_clf_metrics = compare_models(classification_results, problem_type='classification')
        if best_clf_model_name:
            best_clf_model = classification_models[best_clf_model_name]
            save_model(best_clf_model, os.path.join(MODELS_DIR, f'{best_clf_model_name.replace(" ", "_")}_probability_model.joblib'))
            save_model(preprocessor_class, os.path.join(MODELS_DIR, 'preprocessor_probability.joblib'))

            # Model Interpretability for best classification model
            shap_importance_probability = explain_model_shap(best_clf_model, X_test_class_processed_df, all_class_feature_names, model_name=best_clf_model_name.replace(" ", "_") + "_Probability")
            if shap_importance_probability is not None:
                report_prob_str = report_influential_features(shap_importance_probability, prediction_type='claim probability')
                with open(os.path.join(INTERPRETATIONS_DIR, f'{best_clf_model_name.replace(" ", "_")}_probability_interpretations.txt'), 'w') as f:
                    f.write(report_prob_str)
        else:
            print("No best classification model found.")


    print("\n--- Predictive Modeling Pipeline Completed ---")

    # --- Premium Optimization (Conceptual Framework Example) ---
    print("\n## Premium Optimization Framework (Conceptual)")
    print("This framework combines the predicted probability of a claim and the predicted claim severity to calculate a risk-based premium.")
    print("Premium = (Predicted Probability of Claim * Predicted Claim Severity) + Expense Loading + Profit Margin")
    print("\nExample Scenario for a new policy:")
    # To demonstrate this, you'd load new data, preprocess it, then run predictions.
    # For simplicity, we'll use a hypothetical example:
    hypothetical_prob = 0.05 # Example from best_clf_model.predict_proba for a new policy
    hypothetical_severity = 1200 # Example from best_reg_model.predict for a new policy
    expense_loading = 50 # Hypothetical fixed expense per policy
    profit_margin = 20 # Hypothetical desired profit per policy

    optimized_premium = (hypothetical_prob * hypothetical_severity) + expense_loading + profit_margin
    print(f"  Hypothetical Predicted Probability of Claim: {hypothetical_prob:.4f}")
    print(f"  Hypothetical Predicted Claim Severity: ${hypothetical_severity:.2f}")
    print(f"  Expense Loading: ${expense_loading:.2f}")
    print(f"  Profit Margin: ${profit_margin:.2f}")
    print(f"  Calculated Optimal Premium: ${optimized_premium:.2f}")
    print("\nThis calculated premium should be compared against existing premiums and business rules.")

if __name__ == "__main__":
    main()