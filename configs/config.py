from pathlib import Path

# General Settings
RANDOM_STATE = 42
TEST_SIZE = 0.2

# Active Dataset Selection
DATASET = "compas"  # "compas" or "adult"

DATASET_CONFIG = {
    "compas": {
        "dataset_path": "data/raw/compas-scores-two-years.csv",
        "target": "two_year_recid",
        "protected": ["race", "sex"],
        "sensitive": "race"
    },
    "adult": {
        "dataset_path": "data/raw/adult/",
        "target": "income",
        "protected": ["race", "sex"],
        "sensitive": "race"
    }
}

# Dynamic Lookups based on selected dataset
def __getattr__(name):
    if name == "DATASET_PATH":
        return DATASET_CONFIG[DATASET]["dataset_path"]
    elif name == "TARGET_COLUMN":
        return DATASET_CONFIG[DATASET]["target"]
    elif name == "PROTECTED_ATTRIBUTES":
        return DATASET_CONFIG[DATASET]["protected"]
    elif name == "SENSITIVE_ATTRIBUTE":
        return DATASET_CONFIG[DATASET]["sensitive"]
    raise AttributeError(f"module {__name__} has no attribute {name}")

PROCESSED_PATH = "data/processed/"

# Base Directory of the Project
BASE_DIR = Path(__file__).resolve().parent.parent

# Output Directories
LOGS_DIR = BASE_DIR / "logs"
RESULTS_DIR = BASE_DIR / "results" / DATASET
FIGURES_DIR = BASE_DIR / "figures" / DATASET
REPORTS_DIR = BASE_DIR / "reports" / DATASET
SAVED_MODELS_DIR = BASE_DIR / "saved_models" / DATASET

# Ensure runtime directories exist
for directory in [
    LOGS_DIR,
    BASE_DIR / "reports",
    BASE_DIR / "saved_models",
    BASE_DIR / "results",
    BASE_DIR / "figures",
    REPORTS_DIR, 
    SAVED_MODELS_DIR, 
    SAVED_MODELS_DIR / "baseline", 
    SAVED_MODELS_DIR / "improved",
    RESULTS_DIR, 
    RESULTS_DIR / "baseline", 
    RESULTS_DIR / "improved", 
    RESULTS_DIR / "comparison",
    FIGURES_DIR, 
    FIGURES_DIR / "eda", 
    FIGURES_DIR / "fairness", 
    FIGURES_DIR / "explainability", 
    FIGURES_DIR / "paper"
]:
    directory.mkdir(parents=True, exist_ok=True)


# List of Models to run
MODELS_LIST = [
    "logistic_regression",
    "decision_tree",
    "random_forest",
    "xgboost",
    "lightgbm",
    "catboost"
]

# Hyperparameter Tuning Search Grids
TUNING_GRIDS = {
    "logistic_regression": {
        "C": [0.01, 0.1, 1.0, 10.0],
        "penalty": ["l2"],
        "solver": ["lbfgs"]
    },
    "decision_tree": {
        "max_depth": [3, 5, 10, None],
        "min_samples_split": [2, 5, 10],
        "criterion": ["gini", "entropy"]
    },
    "random_forest": {
        "n_estimators": [50, 100, 200],
        "max_depth": [5, 10, None],
        "min_samples_leaf": [1, 2, 4]
    },
    "xgboost": {
        "n_estimators": [50, 100, 200],
        "max_depth": [3, 5, 7],
        "learning_rate": [0.01, 0.1, 0.2]
    },
    "lightgbm": {
        "n_estimators": [50, 100, 200],
        "max_depth": [3, 5, 7],
        "learning_rate": [0.01, 0.1, 0.2],
        "num_leaves": [15, 31, 63]
    },
    "catboost": {
        "iterations": [50, 100, 200],
        "depth": [4, 6, 8],
        "learning_rate": [0.01, 0.1, 0.2]
    }
}

# Fairness Settings
FAIRNESS_METRICS_LIST = [
    "demographic_parity",
    "equalized_odds",
    "equal_opportunity",
    "disparate_impact"
]

# Recommendation Priorities (Weights sum to 1.0)
RECOMMENDATION_PRIORITIES = {
    "accuracy_weight": 0.40,
    "fairness_weight": 0.40,
    "efficiency_weight": 0.20
}

# Subweights for recommendation evaluation
RECOMMENDATION_SUBWEIGHTS = {
    # Under fairness
    "demographic_parity_weight": 0.5,
    "equalized_odds_weight": 0.5,
    # Under performance
    "accuracy_weight": 0.5,
    "f1_weight": 0.3,
    "roc_auc_weight": 0.2,
    # Under efficiency
    "training_time_weight": 0.5,
    "inference_time_weight": 0.5
}