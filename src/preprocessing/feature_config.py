"""
Feature Configuration
---------------------
This file defines which columns should be
used or removed during preprocessing.
"""

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
# Target
# -----------------------------
TARGET_COLUMN = "two_year_recid"

# -----------------------------
PROTECTED_ATTRIBUTES = [
    "race",
    "sex"
]

# -----------------------------
# Categorical Features for Model Input
# -----------------------------
CATEGORICAL_FEATURES = [
    "sex",
    "age_cat",
    "race",
    "c_charge_degree",
]

# -----------------------------
# Numerical Features for Model Input
# -----------------------------
NUMERICAL_FEATURES = [
    "age",
    "priors_count",
    "juv_fel_count",
    "juv_misd_count",
    "juv_other_count",
]

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