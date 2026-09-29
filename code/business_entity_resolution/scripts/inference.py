import sys
from pathlib import Path

import os
import sys

CODE_DIR = Path(__file__).resolve().parents[1]
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))
import lightgbm as lgb
import polars as pl

from config import *
from utils import memory_status

def main():
    print("=" * 80)
    print("INFERENCE")
    print("=" * 80)

    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

    if not CANDIDATE_TSV.exists():
        raise FileNotFoundError(
            f"{CANDIDATE_TSV} not found.\n"
            "Candidate generation is intentionally disabled for this first "
            "version. We will add the blocking/candidate stage separately."
        )

    print("Model:", MODEL_PATH)
    print("Candidates:", CANDIDATE_TSV)
    print("This stage will score candidates in small batches; it will never "
          "load the full test Cartesian product.")
    memory_status("inference start")

if __name__ == "__main__":
    main()
