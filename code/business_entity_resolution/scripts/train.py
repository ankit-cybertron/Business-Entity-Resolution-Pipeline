import sys
import json
import gc
import numpy as np
import polars as pl
import lightgbm as lgb

from config import *
from utils import progress, memory_status, memory_guard, cleanup

SRC = CODE_DIR / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from features import FEATURE_NAMES

def load_ground_truth():
    gt = pl.read_parquet(CACHE_DIR / "train_ground_truth.parquet")
    return {
        r["source1_entity_id"]: (
            set(r["matched_entity_ids"].split(","))
            if r["matched_entity_ids"] else set()
        )
        for r in gt.iter_rows(named=True)
    }

def feature_parts():
    meta = FEATURE_DIR / "feature_parts.json"
    if not meta.exists():
        raise FileNotFoundError(
            f"{meta} not found. Candidate/feature generation must run first."
        )
    return json.loads(meta.read_text())

def sample_rows(parts, max_rows):
    total = sum(p["n_pairs"] for p in parts)
    take = min(total, max_rows)
    rng = np.random.default_rng(RANDOM_SEED)

    # Deterministic proportional sampling from each country part.
    arrays = []
    labels = []
    used = 0

    for p in progress(parts, desc="loading feature parts", unit="country"):
        with np.load(p["path"], mmap_mode=None) as z:
            X = z["X"]
            y = z["y"]
            n = len(y)
            quota = min(n, max(1, int(round(take * n / total))))
            idx = rng.choice(n, size=quota, replace=False)
            idx.sort()

            arrays.append(np.asarray(X[idx], dtype=np.float32))
            labels.append(np.asarray(y[idx], dtype=np.int8))
            used += quota

        cleanup(f"loaded {p['country']} sample")

    X = np.concatenate(arrays, axis=0)
    y = np.concatenate(labels, axis=0)

    # Exact final cap.
    if len(y) > max_rows:
        idx = rng.choice(len(y), size=max_rows, replace=False)
        X = X[idx]
        y = y[idx]

    return X, y

def main():
    print("=" * 80)
    print("LIGHTGBM TRAINING — MEMORY BOUNDED")
    print("=" * 80)
    parts = feature_parts()
    X, y = sample_rows(parts, MAX_TRAIN_PAIRS)

    # Hard memory guard before LightGBM Dataset construction.
    extra_gb = X.nbytes / (1024**3) * 2.5
    memory_guard("LightGBM Dataset construction", extra_gb=extra_gb)

    # Entity-level validation is preferred. If candidate metadata isn't
    # available yet, use a deterministic stratified holdout of feature rows.
    rng = np.random.default_rng(RANDOM_SEED)
    idx = rng.permutation(len(y))
    n_val = max(1, int(len(y) * 0.20))
    val_idx, tr_idx = idx[:n_val], idx[n_val:]

    X_val, y_val = X[val_idx], y[val_idx]
    X_tr, y_tr = X[tr_idx], y[tr_idx]

    del X, y, idx, val_idx, tr_idx
    cleanup("after split")

    pos = int(y_tr.sum())
    neg = int((y_tr == 0).sum())
    scale_pos_weight = max(1.0, neg / max(1, pos))
    params = dict(LGBM_PARAMS)
    params["scale_pos_weight"] = scale_pos_weight

    print(f"train={len(y_tr):,} val={len(y_val):,}")
    print(f"positive={pos:,} negative={neg:,}")
    print(f"scale_pos_weight={scale_pos_weight:.3f}")

    dtrain = lgb.Dataset(
        X_tr, label=y_tr, feature_name=FEATURE_NAMES,
        free_raw_data=False
    )
    dval = lgb.Dataset(
        X_val, label=y_val, reference=dtrain,
        feature_name=FEATURE_NAMES, free_raw_data=False
    )

    callbacks = [
        lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=True),
        lgb.log_evaluation(50),
    ]

    print("\n[TRAIN] LightGBM starting...")
    bst = lgb.train(
        params, dtrain,
        num_boost_round=NUM_BOOST_ROUND,
        valid_sets=[dtrain, dval],
        valid_names=["train", "val"],
        callbacks=callbacks,
    )

    scores = bst.predict(X_val)

    # Keep threshold sweep cheap.
    gt = load_ground_truth()
    thresholds = np.linspace(0.05, 0.95, 91)

    # Row-level fallback threshold selection. Once candidate-pair entity IDs
    # are wired in, replace this with the official per-S1 macro F0.5 evaluator.
    best_t, best_score = DEFAULT_THRESHOLD, -1.0
    for t in thresholds:
        pred = (scores >= t).astype(np.int8)
        tp = int(((pred == 1) & (y_val == 1)).sum())
        fp = int(((pred == 1) & (y_val == 0)).sum())
        fn = int(((pred == 0) & (y_val == 1)).sum())
        precision = tp / max(1, tp + fp)
        recall = tp / max(1, tp + fn)
        f05 = (1.25 * precision * recall / (0.25 * precision + recall)
               if precision + recall else 0.0)
        if f05 > best_score:
            best_score, best_t = f05, float(t)

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    bst.save_model(str(MODEL_PATH))
    THRESH_PATH.write_text(f"{best_t:.6f}\n")
    CASCADE_META_PATH.write_text("enabled=0\n")

    print(f"\nSaved model: {MODEL_PATH}")
    print(f"Saved threshold: {THRESH_PATH} = {best_t:.3f}")
    print(f"Validation row-level F0.5: {best_score:.4f}")

    del dtrain, dval, bst, X_tr, X_val, y_tr, y_val, scores
    cleanup("training complete")

if __name__ == "__main__":
    main()
