import os
import re
import polars as pl

BASE = os.path.expanduser("~/er")
NORM = f"{BASE}/cache/normalized"
OUT = f"{BASE}/cache/candidates/address_retrieval"

os.makedirs(OUT, exist_ok=True)

MAX_BLOCK = 100
TOP_K = 100

GENERIC = {
    "road", "rd", "street", "st", "avenue", "ave",
    "lane", "ln", "drive", "dr", "route", "highway",
    "hwy", "boulevard", "blvd", "parkway", "pkwy",
    "place", "pl", "court", "ct", "way",
    "city", "town", "county", "state",
    "north", "south", "east", "west",
    "n", "s", "e", "w",
    "india", "indian",
    "united", "states", "usa", "america",
    "maharashtra", "california", "texas", "florida",
}


def informative_tokens(x):
    if x is None:
        return []

    tokens = re.findall(r"[a-z0-9]+", str(x).lower())

    return list({
        t
        for t in tokens
        if len(t) >= 3 and t not in GENERIC
    })


def build_index(path, id_name):

    df = pl.read_parquet(path)

    return (
        df
        .select([
            pl.col("entity_id").alias(id_name),
            "country",
            "a_core",
        ])
        .with_columns(
            pl.col("a_core")
            .map_elements(
                informative_tokens,
                return_dtype=pl.List(pl.String),
                skip_nulls=False,
            )
            .alias("tokens")
        )
        .explode("tokens")
        .filter(
            pl.col("tokens").is_not_null() &
            (pl.col("tokens").str.len_chars() > 0)
        )
        .with_columns(
            (
                pl.col("country")
                + "|"
                + pl.col("tokens")
            ).alias("key")
        )
        .select([
            id_name,
            "key",
        ])
        .unique()
    )


def retrieve(s1, target):

    print("Building target index...")

    target_index = (
        target
        .group_by("key")
        .agg(
            pl.col("target_id")
            .alias("target_ids")
        )
    )

    print(
        f"Target token blocks: "
        f"{target_index.height:,}"
    )

    # Keep only manageable blocks.
    target_index = (
        target_index
        .with_columns(
            pl.col("target_ids")
            .list.len()
            .alias("block_size")
        )
        .filter(
            pl.col("block_size") <= MAX_BLOCK
        )
        .drop("block_size")
    )

    print(
        f"Usable target blocks: "
        f"{target_index.height:,}"
    )

    # Join S1 tokens to target token index.
    hits = (
        s1
        .join(
            target_index,
            on="key",
            how="inner",
        )
        .explode("target_ids")
        .rename({
            "target_ids": "matched_id"
        })
    )

    print(
        f"Raw token hits: "
        f"{hits.height:,}"
    )

    # Count how many informative tokens are shared
    # by each S1-target pair.
    scored = (
        hits
        .group_by([
            "s1_id",
            "matched_id",
        ])
        .agg(
            pl.len().alias("token_overlap")
        )
        .sort(
            [
                "s1_id",
                "token_overlap",
            ],
            descending=[False, True],
        )
    )

    # Keep top K retrieved candidates per S1.
    result = (
        scored
        .group_by("s1_id", maintain_order=True)
        .head(TOP_K)
        .select([
            "s1_id",
            "matched_id",
            "token_overlap",
        ])
    )

    return result


def main():

    print("=" * 80)
    print("ADDRESS TOKEN RETRIEVAL")
    print("=" * 80)
    print(f"MAX_BLOCK = {MAX_BLOCK}")
    print(f"TOP_K     = {TOP_K}")

    s1_path = f"{NORM}/train_source1_norm.parquet"

    s1 = build_index(
        s1_path,
        "s1_id",
    )

    print(
        f"S1 token rows: "
        f"{s1.height:,}"
    )

    for source in [2, 3]:

        print("\n" + "=" * 80)
        print(f"S1 -> S{source}")
        print("=" * 80)

        target = build_index(
            f"{NORM}/train_source{source}_norm.parquet",
            "target_id",
        )

        result = retrieve(
            s1,
            target,
        )

        out = (
            pl.DataFrame(result)
            .select([
                "s1_id",
                "matched_id",
            ])
        )

        path = (
            f"{OUT}/"
            f"train_s{source}_address_retrieval.parquet"
        )

        out.write_parquet(
            path,
            compression="zstd",
        )

        print(
            f"Retrieved candidates: "
            f"{out.height:,}"
        )

        print(f"Saved: {path}")


if __name__ == "__main__":
    main()
