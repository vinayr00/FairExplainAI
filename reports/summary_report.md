# FairExplainAI Research Project Summary Report
This report summarizes the experimental evaluation of the baseline model against leakage-aware models with fairness mitigations.

## 1. Dataset & Feature Engineering
- **Raw Shape**: 7214, 53
- **Baseline Processed Shape**: 7214, 19
- **Extended Processed Shape**: 6172, 19
- **Scientific Justification**: Dropped target leakage variables (like `is_recid`, `decile_score`) and administrative variables to avoid target-feedback circularity and administrative bias.


## 2. Model Performance Evaluation
| Model Name | Accuracy | F1-Score | ROC-AUC | Training Time (s) | Inference Time (s) |
|---|---|---|---|---|---|
| RandomForest_Baseline | 0.6355 | 0.5772 | 0.6694 | 1.2140 | 0.0328 |
| RandomForest_Extended | 0.6389 | 0.5840 | 0.6857 | 1.0657 | 0.0263 |
| logistic_regression_Tuned | 0.6923 | 0.6296 | 0.7340 | 0.5000 | 0.0019 |
| decision_tree_Tuned | 0.6753 | 0.5945 | 0.7159 | 0.5000 | 0.0023 |
| random_forest_Tuned | 0.6996 | 0.6402 | 0.7360 | 0.5000 | 0.0116 |
| xgboost_Tuned | 0.7061 | 0.6553 | 0.7391 | 0.5000 | 0.0055 |
| lightgbm_Tuned | 0.7101 | 0.6616 | 0.7387 | 0.5000 | 0.0038 |
| catboost_Tuned | 0.7004 | 0.6442 | 0.7421 | 0.5000 | 0.0025 |
| random_forest_Mitigated | 0.6713 | 0.5972 | 0.0000 | 1.0000 | 0.0500 |
| xgboost_Mitigated | 0.6688 | 0.6079 | 0.0000 | 1.0000 | 0.0500 |
| logistic_regression_Mitigated | 0.6688 | 0.6258 | 0.0000 | 1.0000 | 0.0500 |


## 3. Bias Assessment & Fairness Mitigation
| Model Status | Demographic Parity Diff | Equalized Odds Diff | Equal Opportunity Diff | Disparate Impact Ratio |
|---|---|---|---|---|
| RandomForest Baseline | 0.2018 | 0.6111 | 0.6111 | 0.5998 |
| RandomForest Leakage-Aware (Unmitigated) | 0.5304 | 0.6855 | 0.6855 | 0.0000 |
| random_forest Mitigated (DP) | 0.4255 | 1.0000 | 1.0000 | 0.0000 |
| xgboost Mitigated (DP) | 0.4681 | 1.0000 | 1.0000 | 0.0000 |
| logistic_regression Mitigated (DP) | 0.4681 | 1.0000 | 1.0000 | 0.0000 |


## 4. Multi-Criteria Recommendation Engine
- **Recommended Model**: **decision_tree_Tuned**
- **Overall Utility Score**: 0.6459
- **Justification**: The model 'decision_tree_Tuned' is recommended as the optimal choice. It achieved a combined utility score of 0.6459 (Accuracy: 0.6753, Demographic Parity Difference: 0.4555, Equalized Odds Difference: 0.6291, Training Time: 0.5000s).

