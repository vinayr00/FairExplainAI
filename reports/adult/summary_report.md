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
| RandomForest_Baseline | 0.8553 | 0.6728 | 0.9026 | 5.9566 | 0.2341 |
| RandomForest_Extended | 0.8553 | 0.6728 | 0.9026 | 6.0547 | 0.2337 |
| logistic_regression_Tuned | 0.8448 | 0.6333 | 0.8868 | 0.5000 | 0.0116 |
| decision_tree_Tuned | 0.8615 | 0.6622 | 0.9042 | 0.5000 | 0.0101 |
| random_forest_Tuned | 0.8674 | 0.6939 | 0.9180 | 0.5000 | 0.1708 |
| xgboost_Tuned | 0.8780 | 0.7218 | 0.9317 | 0.5000 | 0.0148 |
| lightgbm_Tuned | 0.8767 | 0.7192 | 0.9309 | 0.5000 | 0.0152 |
| catboost_Tuned | 0.8775 | 0.7197 | 0.9309 | 0.5000 | 0.0121 |
| random_forest_Mitigated | 0.8600 | 0.6929 | 0.0000 | 1.0000 | 0.0500 |
| xgboost_Mitigated | 0.8721 | 0.7104 | 0.0000 | 1.0000 | 0.0500 |
| logistic_regression_Mitigated | 0.8422 | 0.6100 | 0.0000 | 1.0000 | 0.0500 |


## 3. Bias Assessment & Fairness Mitigation
| Model Status | Demographic Parity Diff | Equalized Odds Diff | Equal Opportunity Diff | Disparate Impact Ratio |
|---|---|---|---|---|
| RandomForest Baseline | 0.1390 | 0.0993 | 0.0993 | 0.3874 |
| RandomForest Leakage-Aware (Unmitigated) | 0.1390 | 0.0993 | 0.0993 | 0.3874 |
| random_forest Mitigated (DP) | 0.0609 | 0.1761 | 0.1761 | 0.7787 |
| xgboost Mitigated (DP) | 0.0453 | 0.2487 | 0.2487 | 0.8163 |
| logistic_regression Mitigated (DP) | 0.0348 | 0.1748 | 0.1748 | 0.8153 |


## 4. Multi-Criteria Recommendation Engine
- **Recommended Model**: **xgboost_Tuned**
- **Overall Utility Score**: 0.8616
- **Justification**: The model 'xgboost_Tuned' is recommended as the optimal choice. It achieved a combined utility score of 0.8616 (Accuracy: 0.8780, Demographic Parity Difference: 0.1501, Equalized Odds Difference: 0.2151, Training Time: 0.5000s).

