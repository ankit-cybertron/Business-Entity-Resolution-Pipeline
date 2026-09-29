import polars as pl
import re
from pathlib import Path

BASE = Path.home() / "er"
NORM = BASE / "cache" / "normalized"
MISS = BASE / "cache" / "candidates" / "missing_positives_current.parquet"

missing = pl.read_parquet(MISS)

print(f"Missing positives: {missing.height:,}")


def add_address_keys(df):
    return (
        df.with_columns([
            pl.col("business_address")
              .fill_null("")
              .str.extract(r"(\d+)", 1)
              .fill_null("")
              .alias("_street_num"),

            pl.col("business_address")
              .fill_null("")
              .str.extract(r"(\d{5,6})", 1)
              .fill_null("")
              .alias("_postal_num"),

            pl.col("a_core")
              .fill_null("")
              .str.replace_all(r"\s+", " ")
              .alias("_addr_key"),
        ])
    )


def inspect(source):

    miss = missing.filter(
        pl.col("matched_id").str.starts_with(f"S{source}-")
    )

    s1 = pl.read_parquet(
        NORM / "train_source1_norm.parquet"
    )

    sx = pl.read_parquet(
        NORM / f"train_source{source}_norm.parquet"
    )

    s1 = add_address_keys(s1).select([
        "entity_id",
        "country",
        "n_core",
        "n_ph",
        "n_sk",
        "a_core",
        "business_address",
        "_street_num",
        "_postal_num",
    ])

    sx = add_address_keys(sx).select([
        "entity_id",
        "country",
        "n_core",
        "n_ph",
        "n_sk",
        "a_core",
        "business_address",
        "_street_num",
        "_postal_num",
    ])

    pairs = (
        miss
        .join(
            s1,
            left_on="s1_id",
            right_on="entity_id",
            how="inner",
        )
        .join(
            sx,
            left_on="matched_id",
            right_on="entity_id",
            how="inner",
            suffix="_r",
        )
    )

    print("\n" + "=" * 70)
    print(f"S1 -> S{source}")
    print("=" * 70)
    print(f"Missing pairs: {pairs.height:,}")

    def test(label, expr):
        n = pairs.filter(expr).height
        print(f"{label:32s}: {n:>10,} ({n/pairs.height:.2%})")

    # Address exactness.
    print("\nADDRESS SIGNALS")

    test(
        "address_core_exact",
        (pl.col("a_core") != "")
        & (pl.col("a_core") == pl.col("a_core_r"))
    )

    test(
        "street_number_exact",
        (pl.col("_street_num") != "")
        & (pl.col("_street_num") == pl.col("_street_num_r"))
    )

    test(
        "postal_numeric_exact",
        (pl.col("_postal_num") != "")
        & (pl.col("_postal_num") == pl.col("_postal_num_r"))
    )

    # Combined signals.
    print("\nCOMBINED SIGNALS")

    test(
        "name_prefix4 + street_number",
        (pl.col("n_core").str.slice(0, 4)
         == pl.col("n_core_r").str.slice(0, 4))
        & (pl.col("_street_num") != "")
        & (pl.col("_street_num") == pl.col("_street_num_r"))
    )

    test(
        "name_prefix6 + street_number",
        (pl.col("n_core").str.slice(0, 6)
         == pl.col("n_core_r").str.slice(0, 6))
        & (pl.col("_street_num") != "")
        & (pl.col("_street_num") == pl.col("_street_num_r"))
    )

    test(
        "phonetic + street_number",
        (pl.col("n_ph") != "")
        & (pl.col("n_ph") == pl.col("n_ph_r"))
        & (pl.col("_street_num") != "")
        & (pl.col("_street_num") == pl.col("_street_num_r"))
    )

    test(
        "skeleton + street_number",
        (pl.col("n_sk") != "")
        & (pl.col("n_sk") == pl.col("n_sk_r"))
        & (pl.col("_street_num") != "")
        & (pl.col("_street_num") == pl.col("_street_num_r"))
    )

    test(
        "name_prefix4 + postal",
        (pl.col("n_core").str.slice(0, 4)
         == pl.col("n_core_r").str.slice(0, 4))
        & (pl.col("_postal_num") != "")
        & (pl.col("_postal_num") == pl.col("_postal_num_r"))
    )

    test(
        "name_prefix6 + postal",
        (pl.col("n_core").str.slice(0, 6)
         == pl.col("n_core_r").str.slice(0, 6))
        & (pl.col("_postal_num") != "")
        & (pl.col("_postal_num") == pl.col("_postal_num_r"))
    )

    test(
        "address_core + name_prefix4",
        (pl.col("a_core") != "")
        & (pl.col("a_core") == pl.col("a_core_r"))
        & (pl.col("n_core").str.slice(0, 4)
           == pl.col("n_core_r").str.slice(0, 4))
    )

    # Print a few hard examples.
    print("\nSAMPLE MISSING PAIRS")

    print(
        pairs.select([
            "s1_id",
            "matched_id",
            "business_address",
            "business_address_r",
            "n_core",
            "n_core_r",
        ])
        .head(10)
    )


inspect(2)
inspect(3)
