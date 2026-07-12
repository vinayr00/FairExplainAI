# ⚖️ FairExplainAI: Auditing and Mitigating Bias in Recidivism Prediction

**FairExplainAI** is a publication-grade, end-to-end framework designed to audit, mitigate, and explain machine learning biases in criminal recidivism predictions using the **COMPAS dataset**. 

Rather than proposing new models, the framework focuses on reproducing a baseline Random Forest classifier, identifying target leakage in features, extending evaluation to multiple model architectures, applying post-processing fairness adjustments, and explaining models globally and locally.

---

## 🚀 What exactly the project does

The framework implements a complete, structured data-science pipeline organized into **8 core experiments (A through H)**:

*   **Experiment A: Baseline Paper Reproduction**
    *   Replicates the original Random Forest baseline model using ProPublica's filtering rules and raw features.
    *   Establishes baseline performance metrics (accuracy, F1-score) and demographic parity disparities across sensitive attributes (race, sex).
*   **Experiment B: Leakage-Aware Preprocessing & Cleaning**
    *   Exposes and mitigates target leakage by stripping out internal proprietary COMPAS outputs (e.g., `decile_score`, `score_text`, `v_decile_score`).
    *   Removes non-recidivism administrative markers and performs rigorous preprocessing (handling of screening/arrest offsets, degree filtering).
*   **Experiment C: Multi-Model Evaluation**
    *   Trains and evaluates six machine learning architectures under the leakage-free dataset: Logistic Regression, Decision Trees, Random Forests, XGBoost, LightGBM, and CatBoost.
*   **Experiment D: Fairness Audit & Post-Processing Mitigation**
    *   Computes group-fairness metrics (Demographic Parity Difference, Equalized Odds Difference).
    *   Applies **Fairlearn's Post-processing Threshold Optimizer** to enforce Demographic Parity constraints.
*   **Experiment E: SHAP Explanations**
    *   Generates global feature importance summary plots and local force/waterfall explanation plots to clarify model decision boundaries.
*   **Experiment F: DiCE Counterfactual Explanations**
    *   Generates Diverse Counterfactual Explanations (DiCE) showing the minimal feature changes required to flip a model's prediction from high-risk to low-risk.
*   **Experiment G: Hyperparameter Tuning**
    *   Performs automated Grid Search cross-validation for optimal classifier parameter sets.
*   **Experiment H: Multi-Criteria Recommendation Engine**
    *   Defines a custom utility function scoring models based on Accuracy, Fairness (Demographic Parity/Equalized Odds), and Training efficiency to recommend the optimal model for production deployment.

---

## 🛠️ Key Improvements & Engineering Revisions

The project codebase has been upgraded to support **Python 3.13** and **Pandas 2.x**, resolving strict-casting and library interface deprecations:

### 1. Robust Type Casting for DiCE Explainer
*   **The Bug**: Pandas 2.x raises `LossySetitemError` / `TypeError: Invalid value '0.0' for dtype 'float64'` when DiCE internally assigns categorical values (represented as strings `'0.0'`, `'1.0'`) into query instance frames containing numeric floats.
*   **The Fix**: 
    - Dynamically detects numeric categorical columns.
    - Safely casts these columns in the data interface and query instances to `str` before invoking DiCE.
    - Implemented a `ModelWrapper` class that intercepts requests from DiCE and converts strings back to `float64` before invoking model predictions.

### 2. Strict-Casting Alignment for Fairlearn Mitigation
*   **The Bug**: XGBoost, LightGBM, and CatBoost output predictions as `float32` arrays. When Fairlearn assigns its own `float64` interpolated predictions into slices of the series, Pandas 2.x throws strict casting exceptions.
*   **The Fix**:
    - Created `Float64PredictProbaWrapper` to wrap base estimators, forcing all probability outputs to `float64`.
    - Added a custom `__sklearn_clone__` method so that when scikit-learn clones the estimator, the wrapper is preserved.
    - Guarded `__getattr__` against infinite recursion errors during pickle-based model serialization/unserialization.

### 3. DiCE JSON Serialization Compatibility
*   **The Bug**: The installed version of the `dice_ml` package raises a `TypeError` because `CounterfactualExamples.to_json()` is missing the required positional argument `serialization_version`.
*   **The Fix**: Explicitly specified `serialization_version="2.0"` in the pipeline serialization script.

---

## 📂 Project Structure

```text
FairExplainAI/
│
├── configs/
│   └── config.py       # Centralized configurations, directories, and random seeds
│
├── data/
│   ├── raw/            # Original COMPAS dataset (compas-scores-two-years.csv)
│   └── processed/      # Baseline and extended leakage-free CSV datasets
│
├── src/                # Modular Source Code
│   ├── preprocessing/  # Data filtering, leakage removal, and split modules
│   ├── models/         # Model factory and tuning grids
│   ├── fairness/       # Group fairness calculations and post-processing optimizer
│   ├── explainability/ # SHAP generators and robust DiCE model wrap
│   ├── evaluation/     # Metric logging and utility recommendation engine
│   └── dashboard/      # Multi-page interactive Streamlit dashboard
│
├── reports/            # Generated run metrics, JSON, and HTML summary reports
├── figures/            # Output SHAP and training graphs
├── saved_models/       # Serialized baseline, improved, and mitigated pickle models
├── logs/               # Application run output logs
├── tests/              # Pytest unit tests for all pipeline modules
│
├── requirements.txt    # Library dependencies
├── main.py             # Main entry point to run full experiments A-H
└── README.md           # Project documentation
```

---

## 🚀 Execution Guide

### 1. Installation
Ensure Python 3.8+ is installed:
```bash
pip install -r requirements.txt
```

### 2. Run the Full ML Pipeline
To clean data, tune models, run fairness mitigations, generate SHAP/DiCE plots, and output decision recommendations:
```bash
$env:PYTHONPATH="."
python main.py
```

### 3. Launch the Interactive Dashboard
To explore charts, run custom mitigations, and view explanations interactively:
```bash
$env:PYTHONPATH="."
streamlit run src/dashboard/app.py
```

### 4. Run Unit Tests
To verify all modules and wrappers function correctly:
```bash
$env:PYTHONPATH="."
pytest
```
