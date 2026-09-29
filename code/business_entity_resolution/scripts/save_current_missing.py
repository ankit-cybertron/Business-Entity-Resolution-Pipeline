import polars as pl
from pathlib import Path

BASE = Path.home() / "er"
CAND = BASE / "cache" / "candidates"
GT = BASE / "cache" / "train_ground_truth.parquet"

OUT = CAND / "missing_positives_current.parquet"

print("Loading ground truth...")

gt = (
    pl.read_parquet(GT)
    .select(["source1_entity_id", "matched_entity_ids"])
    .filter(
        pl.col("matched_entity_ids").is_not_null()
        & (pl.col("matched_entity_ids").str.len_chars() > 0)
    )
    .with_columns(
        pl.col("matched_entity_ids").str.split(",").alias("matched_list")
    )
    .explode("matched_list")
    .rename({"matched_list": "matched_id"})
    .select(["source1_entity_id", "matched_id"])
    .unique()
)

print(f"GT positives: {gt.height:,}")

def missing_for(path):
    print(f"\nReading {path.name}...")

    cand = (
        pl.read_parquet(path)
        .select(["s1_id", "matched_id"])
        .unique()
    )

    print(f"Candidates: {cand.height:,}")

    # Only GT pairs belonging to this source.
    prefix = "S2-" if "s2_" in path.name else "S3-"

    gt_source = gt.filter(
        pl.col("matched_id").str.starts_with(prefix)
    )

    found = (
        gt_source
        .join(
            cand,
            left_on=["source1_entity_id", "matched_id"],
            right_on=["s1_id", "matched_id"],
            how="inner",
        )
        .select([
            pl.col("source1_entity_id").alias("s1_id"),
            "matched_id",
        ])
        .unique()
    )

    missing = gt_source.rename({
        "source1_entity_id": "s1_id"
    }).join(
        found,
        on=["s1_id", "matched_id"],
        how="anti",
    )

    print(f"GT:       {gt_source.height:,}")
    print(f"Found:    {found.height:,}")
    print(f"Missing:  {missing.height:,}")

    return missing

m2 = missing_for(CAND / "train_s2_candidates.parquet")
m3 = missing_for(CAND / "train_s3_candidates.parquet")

missing = pl.concat([m2, m3]).unique()

missing.write_parquet(OUT, compression="zstd")

print("\n" + "=" * 60)
print(f"CURRENT MISSING POSITIVES: {missing.height:,}")
print(f"Saved: {OUT}")
print("=" * 60)
