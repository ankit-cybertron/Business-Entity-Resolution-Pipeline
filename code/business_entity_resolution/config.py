from pathlib import Path
import os

# VM has 80 GB RAM. Keep a hard safety ceiling well below that.
MEMORY_HARD_LIMIT_GB = 60.0
MEMORY_WARN_LIMIT_GB = 50.0

RANDOM_SEED = 42

# Training subset. Increase only after confirming RSS stays safe.
TRAIN_LIMIT_S1 = 100_000
FIXED_VAL_SIZE = 25_000
MAX_TRAIN_PAIRS = 15_000_000

# Feature generation
FEATURE_CHUNK_PAIRS = 2_000
FEATURE_THREADS = 4
N_FEATURES = 37

# Candidate generation is intentionally NOT implemented/used yet.
# Later we can add blocking.py without changing the feature/model APIs.

# LightGBM
LGBM_PARAMS = {
    "objective": "binary",
    "metric": "binary_logloss",
    "learning_rate": 0.05,
    "num_leaves": 127,
    "min_child_samples": 30,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "lambda_l1": 0.1,
    "lambda_l2": 0.1,
    "max_bin": 255,
    "random_state": RANDOM_SEED,
    "verbose": -1,
    "device_type": "cpu",
    "force_col_wise": True,
    "n_jobs": max(1, min(8, os.cpu_count() or 1)),
}
NUM_BOOST_ROUND = 2000
EARLY_STOPPING_ROUNDS = 50
DEFAULT_THRESHOLD = 0.45

BASE_DIR = Path(os.environ.get("ER_HOME", Path.home() / "er")).expanduser()
DATASET_DIR = BASE_DIR / "dataset"
TRAIN_DIR = DATASET_DIR / "train"
TEST_DIR = DATASET_DIR / "test"

CACHE_DIR = BASE_DIR / "cache"
NORMALIZED_DIR = CACHE_DIR / "normalized"
CANDIDATE_DIR = CACHE_DIR / "candidates"
FEATURE_DIR = CACHE_DIR / "features"

MODEL_DIR = BASE_DIR / "models"
OUTPUT_DIR = BASE_DIR / "output"
LOG_DIR = OUTPUT_DIR / "logs"

CODE_DIR = BASE_DIR / "code" / "business_entity_resolution"
SRC_DIR = CODE_DIR / "src"
SCRIPTS_DIR = CODE_DIR / "scripts"

for p in (
    CACHE_DIR, NORMALIZED_DIR, CANDIDATE_DIR, FEATURE_DIR,
    MODEL_DIR, OUTPUT_DIR, LOG_DIR, SRC_DIR, SCRIPTS_DIR
):
    p.mkdir(parents=True, exist_ok=True)

TRAIN_SOURCES = {
    1: TRAIN_DIR / "train_source1.tsv",
    2: TRAIN_DIR / "train_source2.tsv",
    3: TRAIN_DIR / "train_source3.tsv",
}
TEST_SOURCES = {
    1: TEST_DIR / "test_source1.tsv",
    2: TEST_DIR / "test_source2.tsv",
    3: TEST_DIR / "test_source3.tsv",
}
TRAIN_GT = TRAIN_DIR / "train_ground_truth.tsv"

MODEL_PATH = MODEL_DIR / "entity_resolution_lgbm.txt"
THRESH_PATH = MODEL_DIR / "threshold.txt"
CASCADE_META_PATH = MODEL_DIR / "cascade_meta.txt"

MATCHING_TSV = OUTPUT_DIR / "matching_results.tsv"
CANDIDATE_TSV = OUTPUT_DIR / "candidate_pairs.tsv"
