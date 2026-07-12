"""
Feature Configuration
---------------------
This file defines which columns should be
used or removed during preprocessing.
"""
from configs.config import DATASET, TARGET_COLUMN, PROTECTED_ATTRIBUTES

# -----------------------------
# Identifier Columns
# -----------------------------
IDENTIFIER_COLUMNS = [
    "id",
    "name",
    "first",
    "last",
    "c_case_number"
]

# -----------------------------
# Date Columns
# -----------------------------
DATE_COLUMNS = [
    "dob",
    "compas_screening_date",
    "c_jail_in",
    "c_jail_out",
    "c_offense_date",
    "c_arrest_date",
    "in_custody",
    "out_custody",
    "start",
    "end",
    "screening_date",
"v_screening_date"
]

# -----------------------------
# Leakage Columns
# -----------------------------
LEAKAGE_COLUMNS = [
    "is_recid",
    "event",
    "r_case_number",
    "r_charge_degree",
    "r_charge_desc",
    "r_offense_date",
    "r_jail_in",
    "r_jail_out",
    "r_days_from_arrest"
]

# -----------------------------
# COMPAS Generated Outputs
# -----------------------------
COMPAS_COLUMNS = [
    "decile_score",
    "decile_score.1",
    "score_text",
    "v_decile_score",
    "v_score_text",
    "type_of_assessment",
"v_type_of_assessment"
]

# -----------------------------
# Violence Related
# -----------------------------
VIOLENCE_COLUMNS = [
    "violent_recid",
    "vr_case_number",
    "vr_charge_degree",
    "vr_charge_desc",
    "vr_offense_date",
    "is_violent_recid",
]

# -----------------------------
# Duplicate Columns
# -----------------------------
DUPLICATE_COLUMNS = [
    "priors_count.1"
]

# -----------------------------
# Target & Sensitive attributes (derived from config)
# -----------------------------
# (Imports TARGET_COLUMN and PROTECTED_ATTRIBUTES from configs.config)

# -----------------------------
# Dynamic Features for Model Input
# -----------------------------
FEATURE_CONFIG = {
    "compas": {
        "categorical": ["sex", "age_cat", "race", "c_charge_degree"],
        "numerical": ["age", "priors_count", "juv_fel_count", "juv_misd_count", "juv_other_count"]
    },
    "adult": {
        "categorical": ["workclass", "education", "marital-status", "occupation", "relationship", "race", "sex", "native-country"],
        "numerical": ["age", "fnlwgt", "education-num", "capital-gain", "capital-loss", "hours-per-week"]
    }
}

def __getattr__(name):
    if name == "CATEGORICAL_FEATURES":
        import configs.config as cfg
        return FEATURE_CONFIG[cfg.DATASET]["categorical"]
    elif name == "NUMERICAL_FEATURES":
        import configs.config as cfg
        return FEATURE_CONFIG[cfg.DATASET]["numerical"]
    raise AttributeError(f"module {__name__} has no attribute {name}")

# -----------------------------
# Columns to drop during loading / cleaning
# -----------------------------
ADMINISTRATIVE_COLUMNS = [
    "c_charge_desc",
    "c_days_from_compas",
    "days_b_screening_arrest"
]

COLUMNS_TO_DROP = (
    IDENTIFIER_COLUMNS
    + DATE_COLUMNS
    + LEAKAGE_COLUMNS
    + COMPAS_COLUMNS
    + VIOLENCE_COLUMNS
    + DUPLICATE_COLUMNS
    + ADMINISTRATIVE_COLUMNS
)