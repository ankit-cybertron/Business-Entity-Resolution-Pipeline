import polars as pl
from pathlib import Path

BASE = Path.home() / "er"
NORM = BASE / "cache" / "normalized"
MISS = BASE / "cache" / "candidates" / "missing_positives_current.parquet"

missing = pl.read_parquet(MISS)

def add_keys(df):
    return df.with_columns([
        pl.col("business_address")
        .fill_null("")
        .str.extract(r"(\d+)", 1)
        .fill_null("")
        .alias("_num"),

        pl.col("business_address")
        .fill_null("")
        .str.extract(r"(\d{5,6})", 1)
        .fill_null("")
        .alias("_postal"),

        pl.col("business_address")
        .fill_null("")
        .str.to_lowercase()
        .str.replace_all(r"[^a-z0-9]+", " ")
        .str.split(" ")
        .list.eval(
            pl.element().filter(pl.element().str.len_chars() >= 3)
        )
        .alias("_addr_tokens"),
    ])

def check(source):
    miss = missing.filter(
        pl.col("matched_id").str.starts_with(f"S{source}-")
    )

    s1 = add_keys(
        pl.read_parquet(NORM / "train_source1_norm.parquet")
        .select([
            "entity_id", "country", "n_core",
            "business_address"
        ])
    )

    sx = add_keys(
        pl.read_parquet(NORM / f"train_source{source}_norm.parquet")
        .select([
            "entity_id", "country", "n_core",
            "business_address"
        ])
    )

    pairs = (
        miss
        .join(s1, left_on="s1_id", right_on="entity_id")
        .join(
            sx,
            left_on="matched_id",
            right_on="entity_id",
            suffix="_r"
        )
    )

    print("\n" + "=" * 65)
    print(f"S1 -> S{source}")
    print(f"Missing pairs: {pairs.height:,}")
    print("=" * 65)

    def test(name, expr):
        n = pairs.filter(expr).height
        print(f"{name:38s}: {n:>10,} ({n/pairs.height:.2%})")

    # Same country + street number
    test(
        "country + street_number",
        (pl.col("country") == pl.col("country_r"))
        & (pl.col("_num") != "")
        & (pl.col("_num") == pl.col("_num_r"))
    )

    # Same country + street number + postal
    test(
        "country + number + postal",
        (pl.col("country") == pl.col("country_r"))
        & (pl.col("_num") != "")
        & (pl.col("_num") == pl.col("_num_r"))
        & (pl.col("_postal") != "")
        & (pl.col("_postal") == pl.col("_postal_r"))
    )

    # Same country + number + first 4 name chars
    test(
        "country + number + name_prefix4",
        (pl.col("country") == pl.col("country_r"))
        & (pl.col("_num") != "")
        & (pl.col("_num") == pl.col("_num_r"))
        & (
            pl.col("n_core").str.slice(0, 4)
            == pl.col("n_core_r").str.slice(0, 4)
        )
    )

    # Same country + number + first 6 name chars
    test(
        "country + number + name_prefix6",
        (pl.col("country") == pl.col("country_r"))
        & (pl.col("_num") != "")
        & (pl.col("_num") == pl.col("_num_r"))
        & (
            pl.col("n_core").str.slice(0, 6)
            == pl.col("n_core_r").str.slice(0, 6)
        )
    )

    # Same country + number + address token overlap
    test(
        "country + number + address token overlap",
        (pl.col("country") == pl.col("country_r"))
        & (pl.col("_num") != "")
        & (pl.col("_num") == pl.col("_num_r"))
        & (
            pl.col("_addr_tokens").list.set_intersection(
                pl.col("_addr_tokens_r")
            ).list.len() >= 2
        )
    )

check(2)
check(3)
