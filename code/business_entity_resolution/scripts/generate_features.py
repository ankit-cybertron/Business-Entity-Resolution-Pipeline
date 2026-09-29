import json
import gc
from pathlib import Path

import numpy as np
import polars as pl
from tqdm.auto import tqdm

import sys
sys.path.insert(0, str(Path.home() / "er" / "code" / "business_entity_resolution"))

from src.features import build_precomp, compute_features_batch, FEATURE_NAMES


BASE = Path.home() / "er"

LABELED = BASE / "cache/candidates/train_labeled.parquet"
NORM_DIR = BASE / "cache/normalized"
FEATURE_DIR = BASE / "cache/features"

CHUNK_SIZE = 100_000
WORKERS = 4

FEATURE_DIR.mkdir(parents=True, exist_ok=True)


print("=" * 70)
print("GENERATE TRAINING FEATURES")
print("=" * 70)


# ============================================================
# 1. LOAD NORMALIZED DATA
# ============================================================

print("\n[1/4] Loading normalized records...")

s1 = pl.read_parquet(
    NORM_DIR / "train_source1_norm.parquet"
)

s2 = pl.read_parquet(
    NORM_DIR / "train_source2_norm.parquet"
)

s3 = pl.read_parquet(
    NORM_DIR / "train_source3_norm.parquet"
)

print(f"S1: {len(s1):,}")
print(f"S2: {len(s2):,}")
print(f"S3: {len(s3):,}")


# ============================================================
# 2. BUILD PRECOMPUTED REPRESENTATIONS
# ============================================================

print("\n[2/4] Building precomputed representations...")

rec_map = {}

columns = [
    "entity_id",
    "business_name",
    "business_address",
    "country",
    "n_core",
    "n_ph",
    "n_sk",
    "a_core",
]

for df in (s1, s2, s3):
    for row in df.select(columns).iter_rows():
        (
            eid,
            name,
            addr,
            country,
            nc,
            nph,
            nsk,
            ac,
        ) = row

        rec_map[eid] = (
            name,
            addr,
            country,
            nc,
            nph,
            nsk,
            ac,
        )

print(f"Precomp records: {len(rec_map):,}")

precomp = build_precomp(rec_map)

# Raw normalized tables are no longer needed.
del s1, s2, s3, rec_map
gc.collect()


# ============================================================
# 3. PROCESS LABELED PAIRS IN CHUNKS
# ============================================================

print("\n[3/4] Generating feature chunks...")

df = pl.read_parquet(
    LABELED,
    columns=[
        "s1_id",
        "matched_id",
        "label",
    ],
)

total = len(df)

print(f"Labeled pairs: {total:,}")
print(f"Chunk size: {CHUNK_SIZE:,}")
print(f"Features: {len(FEATURE_NAMES)}")


# Remove old feature chunks from previous runs.
for p in FEATURE_DIR.glob("features_*.npz"):
    p.unlink()

for p in FEATURE_DIR.glob("labels_*.npy"):
    p.unlink()


parts = []

for start in tqdm(
    range(0, total, CHUNK_SIZE),
    desc="Feature chunks",
):
    end = min(start + CHUNK_SIZE, total)

    chunk = df.slice(start, end - start)

    pairs = list(
        zip(
            chunk["s1_id"].to_list(),
            chunk["matched_id"].to_list(),
        )
    )

    X, ids = compute_features_batch(
        pairs,
        precomp,
    )

    y = chunk["label"].to_numpy().astype(np.int8)

    if X.shape != (len(chunk), len(FEATURE_NAMES)):
        raise RuntimeError(
            f"Unexpected feature shape: {X.shape}; "
            f"expected {(len(chunk), len(FEATURE_NAMES))}"
        )

    part_id = len(parts)

    feature_path = FEATURE_DIR / f"features_{part_id:05d}.npz"
    label_path = FEATURE_DIR / f"labels_{part_id:05d}.npy"

    np.savez_compressed(
        feature_path,
        X=X.astype(np.float32, copy=False),
    )

    np.save(
        label_path,
        y,
    )

    parts.append({
        "features": str(feature_path),
        "labels": str(label_path),
        "rows": int(len(chunk)),
        "n_features": int(X.shape[1]),
    })

    del chunk, pairs, X, y, ids
    gc.collect()


# ============================================================
# 4. WRITE METADATA
# ============================================================

print("\n[4/4] Writing feature metadata...")

meta = {
    "n_parts": len(parts),
    "total_rows": total,
    "n_features": len(FEATURE_NAMES),
    "feature_names": FEATURE_NAMES,
    "parts": parts,
}

meta_path = FEATURE_DIR / "feature_parts.json"

with open(meta_path, "w") as f:
    json.dump(meta, f, indent=2)

print("\n" + "=" * 70)
print("FEATURE GENERATION COMPLETE")
print("=" * 70)
print(f"Rows:       {total:,}")
print(f"Features:   {len(FEATURE_NAMES)}")
print(f"Parts:      {len(parts)}")
print(f"Metadata:   {meta_path}")
print("=" * 70)
