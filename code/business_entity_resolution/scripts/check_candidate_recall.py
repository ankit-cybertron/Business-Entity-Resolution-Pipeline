import polars as pl
from pathlib import Path

BASE = Path.home() / "er"
CAND_DIR = BASE / "cache" / "candidates"
GT_PATH = BASE / "cache" / "train_ground_truth.parquet"

print("Loading ground truth...")

gt = (
    pl.read_parquet(GT_PATH)
    .select(["source1_entity_id", "matched_entity_ids"])
    .filter(
        pl.col("matched_entity_ids").is_not_null()
        & (pl.col("matched_entity_ids").str.len_chars() > 0)
    )
    .with_columns(
        pl.col("matched_entity_ids")
        .str.split(",")
        .alias("matched_list")
    )
    .explode("matched_list")
    .rename({"matched_list": "matched_id"})
    .select(["source1_entity_id", "matched_id"])
    .unique()
)

total_gt = gt.height

print(f"Ground-truth positive pairs: {total_gt:,}")
print()

def check(source, path):
    print("=" * 60)
    print(f"Checking S1 -> {source}")
    print(f"Candidate file: {path}")

    cand = (
        pl.read_parquet(path)
        .select(["s1_id", "matched_id"])
        .unique()
    )

    # Keep only GT pairs belonging to this source.
    gt_source = gt.filter(
        pl.col("matched_id").str.starts_with(f"S{source}-")
    )

    total = gt_source.height

    found = (
        gt_source
        .join(
            cand,
            left_on=["source1_entity_id", "matched_id"],
            right_on=["s1_id", "matched_id"],
            how="inner",
        )
        .height
    )

    missing = total - found
    recall = found / total if total else 0.0

    print(f"Ground-truth positives: {total:,}")
    print(f"Candidates:              {cand.height:,}")
    print(f"Found positives:         {found:,}")
    print(f"Missing positives:       {missing:,}")
    print(f"Candidate recall:        {recall:.6%}")
    print()

    del cand
    return total, found, missing

s2_total, s2_found, s2_missing = check(
    2,
    CAND_DIR / "train_s2_candidates.parquet"
)

s3_total, s3_found, s3_missing = check(
    3,
    CAND_DIR / "train_s3_candidates.parquet"
)

total = s2_total + s3_total
found = s2_found + s3_found
missing = total - found
recall = found / total if total else 0.0

print("=" * 60)
print("FINAL CANDIDATE RECALL")
print("=" * 60)
print(f"Ground-truth positives: {total:,}")
print(f"Found in candidates:    {found:,}")
print(f"Missing:                {missing:,}")
print(f"Candidate recall:       {recall:.6%}")
print("=" * 60)
