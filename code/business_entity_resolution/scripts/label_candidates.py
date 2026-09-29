import gc
from pathlib import Path

import polars as pl

BASE = Path.home() / "er"

CAND = BASE / "cache/candidates/train_candidates.parquet"
GT = BASE / "cache/train_ground_truth.parquet"
OUT = BASE / "cache/candidates/train_labeled.parquet"

# Number of negatives retained per S1/source.
NEG_PER_S1_SOURCE = 3

print("=" * 70)
print("LABEL + REDUCE TRAINING CANDIDATES")
print("=" * 70)

# ============================================================
# 1. LOAD GROUND TRUTH
# ============================================================

print("\n[1/4] Loading ground truth...")

gt = (
    pl.read_parquet(GT)
    .with_columns(
        pl.col("matched_entity_ids")
        .fill_null("")
        .str.split(",")
        .alias("matched_list")
    )
    .explode("matched_list")
    .rename({
        "source1_entity_id": "s1_id",
        "matched_list": "matched_id",
    })
    .filter(
        pl.col("matched_id").is_not_null()
        & (pl.col("matched_id") != "")
    )
    .select(["s1_id", "matched_id"])
    .unique()
    .with_columns(
        pl.lit(1, dtype=pl.Int8).alias("is_positive")
    )
)

print(f"Ground-truth positive pairs: {len(gt):,}")

# ============================================================
# 2. LABEL ALL CANDIDATES
# ============================================================

print("\n[2/4] Labeling candidate pairs...")

cand = pl.scan_parquet(CAND)

labeled = (
    cand
    .join(
        gt.lazy(),
        on=["s1_id", "matched_id"],
        how="left",
    )
    .with_columns([
        pl.col("is_positive")
        .fill_null(0)
        .cast(pl.Int8)
        .alias("label"),

        # train_candidates.parquet contains only s1_id + matched_id.
        # Derive the source from the matched entity ID.
        pl.when(pl.col("matched_id").str.starts_with("S2-"))
        .then(pl.lit("s2"))
        .when(pl.col("matched_id").str.starts_with("S3-"))
        .then(pl.lit("s3"))
        .otherwise(pl.lit("unknown"))
        .alias("source"),
    ])
    .select([
        "s1_id",
        "matched_id",
        "source",
        "label",
    ])
)

# ============================================================
# 3. KEEP ALL POSITIVES + SMALL NEGATIVE SAMPLE
# ============================================================

print("\n[3/4] Reducing negatives...")

# Deterministic hash.
# We first keep ~10% of negatives so we don't materialize
# all candidate negatives.
sampled = (
    labeled
    .with_columns(
        pl.struct(["s1_id", "matched_id"])
        .hash(seed=42)
        .alias("_hash")
    )
    .filter(
        (pl.col("label") == 1)
        |
        ((pl.col("_hash") % 10) == 0)
    )
    .drop("_hash")
)

print("Collecting reduced candidate pool...")

df = sampled.collect(engine="streaming")

print(f"Reduced pool: {len(df):,}")

# ============================================================
# 4. KEEP ALL POSITIVES + 3 NEGATIVES PER S1/SOURCE
# ============================================================

print("\n[4/4] Applying per-entity negative cap...")

positives = (
    df
    .filter(pl.col("label") == 1)
)

negatives = (
    df
    .filter(pl.col("label") == 0)
    .sort(["s1_id", "source", "matched_id"])
    .group_by(
        ["s1_id", "source"],
        maintain_order=True
    )
    .head(NEG_PER_S1_SOURCE)
)

print(f"Positive pairs retained: {len(positives):,}")
print(f"Negative pairs retained: {len(negatives):,}")

positives = positives.select([
    "s1_id",
    "matched_id",
    "source",
    "label",
])

negatives = negatives.select([
    "s1_id",
    "matched_id",
    "source",
    "label",
])

result = (
    pl.concat([
        positives,
        negatives,
    ])
    .unique(["s1_id", "matched_id", "source"])
    .sort(
        ["s1_id", "source", "label"],
        descending=[False, False, True]
    )
)

result.write_parquet(
    OUT,
    compression="zstd",
)

print(f"\nWrote: {OUT}")
print(f"Final labeled rows: {len(result):,}")
print("\nDone.")
