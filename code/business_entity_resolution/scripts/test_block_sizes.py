import polars as pl
from pathlib import Path

BASE = Path.home() / "er"
NORM = BASE / "cache" / "normalized"

def prepare(path):
    df = pl.read_parquet(path).select([
        "entity_id",
        "country",
        "n_core",
        "business_address",
    ])

    return df.with_columns([
        pl.col("business_address")
        .fill_null("")
        .str.extract(r"(\d+)", 1)
        .fill_null("")
        .alias("_num"),

        pl.col("business_address")
        .fill_null("")
        .str.to_lowercase()
        .str.replace_all(r"[^a-z0-9]+", " ")
        .str.split(" ")
        .list.eval(
            pl.element().filter(pl.element().str.len_chars() >= 3)
        )
        .alias("_tokens"),
    ]).with_columns([
        pl.col("n_core").str.slice(0, 4).alias("_p4"),
        pl.col("n_core").str.slice(0, 6).alias("_p6"),
        pl.col("_tokens").list.sort().list.join("_").alias("_addr_sig"),
    ])

def analyze(s1, sx, source):

    print("\n" + "=" * 70)
    print(f"S1 -> S{source}")
    print("=" * 70)

    for name, key in [
        ("number+p4", ["country", "_num", "_p4"]),
        ("number+p6", ["country", "_num", "_p6"]),
        ("number+addr_sig", ["country", "_num", "_addr_sig"]),
    ]:

        l = (
            s1.filter(
                (pl.col("_num") != "") &
                (pl.col("_p4") != "")
            )
            .group_by(key)
            .len()
            .rename({"len": "_ln"})
        )

        r = (
            sx.filter(
                (pl.col("_num") != "") &
                (pl.col("_p4") != "")
            )
            .group_by(key)
            .len()
            .rename({"len": "_rn"})
        )

        blocks = l.join(r, on=key, how="inner")

        print(f"\n[{name}]")

        for cap in [20, 50, 100, 200, 500]:
            valid = blocks.filter(
                (pl.col("_ln") <= cap) &
                (pl.col("_rn") <= cap)
            )

            print(
                f"  cap={cap:3d} | "
                f"valid blocks={valid.height:,} | "
                f"potential pairs="
                f"{(valid['_ln'] * valid['_rn']).sum():,}"
            )

s1 = prepare(NORM / "train_source1_norm.parquet")
s2 = prepare(NORM / "train_source2_norm.parquet")
s3 = prepare(NORM / "train_source3_norm.parquet")

analyze(s1, s2, 2)
analyze(s1, s3, 3)
