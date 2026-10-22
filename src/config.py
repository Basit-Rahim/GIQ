"""Paths, model inputs, targets and all modelling parameters."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# Paths
RAW_DIR = ROOT / "data" / "raw"              # PSLM download goes here (not in git)
INTERIM_DIR = ROOT / "data" / "interim"      # built from the PSLM download (not in git)
PROCESSED_DIR = ROOT / "data" / "processed"
MODELS_DIR = ROOT / "models"
OUTPUTS_DIR = ROOT / "outputs"

PSLM_ZIP = RAW_DIR / "stata_data_pslm_2019_20.zip"
TRAINING_TABLE = PROCESSED_DIR / "households.csv"
DISTANCE_TIME_TABLE = PROCESSED_DIR / "distance_time_lookup.csv"
PSLM_TABLE = INTERIM_DIR / "pslm_lahore_households.csv"
MODEL_BUNDLE = MODELS_DIR / "model.joblib"

PRESET = "urban"
SEED = 42
SCHOOL_AGE = (5, 16)

# Model inputs (household level, raw units)
FEATURE_COLUMNS = [
    "monthly_income",     # PKR per month, sum over household members
    "travel_mode",        # most common mode among children in school
    "route_safe",         # 1 = every child's route reported safe, 0 = any unsafe
    "read_write",         # household head can read and write
    "solve_math",         # household head can solve simple sums
    "school_facilities",  # weighted facility index, 0-5
    "min_distance",       # km, nearest school among the household's children
    "max_distance",       # km, farthest school among the household's children
    "min_time",           # minutes, shortest journey
    "max_time",           # minutes, longest journey
]
# Too few unsafe routes to estimate an effect; route safety is applied after the model.
STAGE2_FACTUAL_COLUMNS = [c for c in FEATURE_COLUMNS if c != "route_safe"]

# Stage 1 targets: four-point scale, 0 = not important/helpful ... 3 = very
STAGE1_TARGETS = {
    "perceived_cost": "b2_1",         # cost of schooling
    "perceived_distance": "b2_3",     # distance to school
    "perceived_safety": "b2_4",       # road safety for the child
    "perceived_facilities": "b2_14",  # basic school facilities
    "stipend_helpfulness": "b3_1",    # financial support (fee waivers, scholarships)
}
N_LIKERT_CLASSES = 4

# Stage 2 target: share of the household's school-age children in school.
TARGET_COLUMN = "enrolled_share"
# A household counts as enrolled if at least half of its school-age children attend.
ENROLLED_THRESHOLD = 0.5

# Model parameters
STAGE1_HIDDEN = (32, 16)
STAGE1_EPOCHS = 300
STAGE1_LR = 1e-3
STAGE1_WEIGHT_DECAY = 1e-3
LATENT_DIM = 3
AE_HIDDEN = 4
AE_EPOCHS = 500
AE_LR = 1e-2
CV_FOLDS = 5
CV_REPEATS = 3

# Route-safety adjustment applied after the model (an assumption, not an estimate).
UNSAFE_ROUTE_MULTIPLIER = 0.8

# PSLM validation: inputs PSLM lacks, or records only for children in school,
# are held at one value for every household.
PSLM_MIN_DISTANCE_KM = 0.1
PSLM_MAX_DISTANCE_KM = 1.0
# Route safety is not recorded: each route is equally likely safe or unsafe,
# so the prediction is the average of the two cases.
PSLM_UNSAFE_ROUTE_SHARE = 0.5
