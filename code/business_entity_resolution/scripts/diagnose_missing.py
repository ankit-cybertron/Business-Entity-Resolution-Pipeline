import polars as pl
from pathlib import Path

BASE = Path.home() / "er"
NORM = BASE / "cache" / "normalized"
MISS = BASE / "cache" / "candidates" / "missing_positives_current.parquet"

print("Loading missing positives...")

missing = pl.read_parquet(MISS)

print(f"Missing positives: {missing.height:,}")

m2 = missing.filter(pl.col("matched_id").str.starts_with("S2-"))
m3 = missing.filter(pl.col("matched_id").str.starts_with("S3-"))

print(f"Missing S2: {m2.height:,}")
print(f"Missing S3: {m3.height:,}")


def find_col(df, candidates, label):
    for c in candidates:
        if c in df.columns:
            return c
    raise RuntimeError(
        f"Could not find {label} ID column.\n"
        f"Available columns: {df.columns}"
    )


def inspect(missing_df, source):

    print(f"\nLoading normalized S1/S{source}...")

    s1 = pl.read_parquet(NORM / "train_source1_norm.parquet")
    sx = pl.read_parquet(NORM / f"train_source{source}_norm.parquet")

    print("S1 columns:", s1.columns)
    print(f"S{source} columns:", sx.columns)

    s1_id = find_col(
        s1,
        ["source1_entity_id", "entity_id", "id"],
        "S1"
    )

    sx_id = find_col(
        sx,
        ["entity_id", f"source{source}_entity_id", "id"],
        f"S{source}"
    )

    wanted_s1 = [
        s1_id,
        "country",
        "n_core",
        "n_ph",
        "n_sk",
        "postal_code",
        "address",
    ]

    wanted_sx = [
        sx_id,
        "country",
        "n_core",
        "n_ph",
        "n_sk",
        "postal_code",
        "address",
    ]

    wanted_s1 = [c for c in wanted_s1 if c in s1.columns]
    wanted_sx = [c for c in wanted_sx if c in sx.columns]

    s1 = s1.select(wanted_s1)
    sx = sx.select(wanted_sx)

    # Rename IDs to stable names.
    s1 = s1.rename({s1_id: "_s1_id"})
    sx = sx.rename({sx_id: "_matched_id"})

    print(f"\nJoining {missing_df.height:,} missing S1->S{source} pairs...")

    pairs = (
        missing_df
        .join(
            s1,
            left_on="s1_id",
            right_on="_s1_id",
            how="inner",
        )
        .join(
            sx,
            left_on="matched_id",
            right_on="_matched_id",
            how="inner",
            suffix="_r",
        )
    )

    print(f"Joined pairs: {pairs.height:,}")

    # Helper: only evaluate a signal if its columns exist.
    def check(name, expr):
        n = pairs.filter(expr).height
        print(f"{name:25s}: {n:>10,} ({n / pairs.height:.2%})")

    print("\nExact signals:")

    if "country" in pairs.columns and "country_r" in pairs.columns:
        check(
            "country_match",
            pl.col("country") == pl.col("country_r")
        )

    if "n_core" in pairs.columns and "n_core_r" in pairs.columns:
        check(
            "name_exact",
            pl.col("n_core") == pl.col("n_core_r")
        )

    if "n_ph" in pairs.columns and "n_ph_r" in pairs.columns:
        check(
            "phonetic_exact",
            pl.col("n_ph") == pl.col("n_ph_r")
        )

    if "n_sk" in pairs.columns and "n_sk_r" in pairs.columns:
        check(
            "skeleton_exact",
            pl.col("n_sk") == pl.col("n_sk_r")
        )

    if "postal_code" in pairs.columns and "postal_code_r" in pairs.columns:
        check(
            "postal_exact",
            pl.col("postal_code") == pl.col("postal_code_r")
        )

    print("\nCombined signals:")

    if all(c in pairs.columns for c in ["country", "country_r",
                                         "postal_code", "postal_code_r"]):
        check(
            "country+postal",
            (pl.col("country") == pl.col("country_r"))
            & (pl.col("postal_code") == pl.col("postal_code_r"))
        )

    if all(c in pairs.columns for c in ["country", "country_r",
                                         "n_core", "n_core_r"]):
        check(
            "country+name",
            (pl.col("country") == pl.col("country_r"))
            & (pl.col("n_core") == pl.col("n_core_r"))
        )

    if all(c in pairs.columns for c in ["country", "country_r",
                                         "n_ph", "n_ph_r"]):
        check(
            "country+phonetic",
            (pl.col("country") == pl.col("country_r"))
            & (pl.col("n_ph") == pl.col("n_ph_r"))
        )

    if all(c in pairs.columns for c in ["country", "country_r",
                                         "n_sk", "n_sk_r"]):
        check(
            "country+skeleton",
            (pl.col("country") == pl.col("country_r"))
            & (pl.col("n_sk") == pl.col("n_sk_r"))
        )

    if all(c in pairs.columns for c in ["n_core", "n_core_r",
                                         "postal_code", "postal_code_r"]):
        check(
            "name+postal",
            (pl.col("n_core") == pl.col("n_core_r"))
            & (pl.col("postal_code") == pl.col("postal_code_r"))
        )

    if all(c in pairs.columns for c in ["n_ph", "n_ph_r",
                                         "postal_code", "postal_code_r"]):
        check(
            "phonetic+postal",
            (pl.col("n_ph") == pl.col("n_ph_r"))
            & (pl.col("postal_code") == pl.col("postal_code_r"))
        )

    if all(c in pairs.columns for c in ["n_sk", "n_sk_r",
                                         "postal_code", "postal_code_r"]):
        check(
            "skeleton+postal",
            (pl.col("n_sk") == pl.col("n_sk_r"))
            & (pl.col("postal_code") == pl.col("postal_code_r"))
        )


inspect(m2, 2)
inspect(m3, 3)
