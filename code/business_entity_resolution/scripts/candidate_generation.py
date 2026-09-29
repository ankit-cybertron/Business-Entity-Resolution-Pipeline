import os
import re
import polars as pl

BASE = os.path.expanduser("~/er")

NORM = f"{BASE}/cache/normalized"
OUT = f"{BASE}/cache/candidates"

os.makedirs(OUT, exist_ok=True)

ORIGINAL_MAX_BLOCK = 500
FALLBACK_MAX_BLOCK = 100

SOURCES = [2, 3]


def postal(x):
    if not x:
        return ""
    nums = re.findall(r"\d+", x)
    return nums[-1] if nums else ""


def street_num(x):
    if not x:
        return ""
    m = re.match(r"^\s*(\d+)", x)
    return m.group(1) if m else ""


def first_token(x):
    return x.split()[0] if x else ""


def last_token(x):
    return x.split()[-1] if x else ""


def sorted_name(x):
    return " ".join(sorted(x.split())) if x else ""


def prepare(df, side):
    return (
        df
        .with_columns([
            pl.col("business_address")
            .map_elements(postal, return_dtype=pl.String)
            .alias("_postal"),

            pl.col("business_address")
            .map_elements(street_num, return_dtype=pl.String)
            .alias("_street"),

            pl.col("n_core")
            .map_elements(first_token, return_dtype=pl.String)
            .alias("_first"),

            pl.col("n_core")
            .map_elements(last_token, return_dtype=pl.String)
            .alias("_last"),

            pl.col("n_core")
            .map_elements(sorted_name, return_dtype=pl.String)
            .alias("_sorted"),

            pl.col("business_address")
            .fill_null("")
            .str.to_lowercase()
            .str.replace_all(r"[^a-z0-9]+", " ")
            .str.split(" ")
            .list.eval(
                pl.element().filter(
                    pl.element().str.len_chars() >= 3
                )
            )
            .list.sort()
            .list.join("_")
            .alias("_addr_sig"),
        ])
        .rename({"entity_id": f"{side}_id"})
    )


def generate_rule(left, right, rule):
    rules = {
        # Existing blockers
        "exact_name": (
            pl.col("country") + "|" + pl.col("n_core")
        ),

        "phonetic_name": (
            pl.col("country") + "|" + pl.col("n_ph")
        ),

        "skeleton_name": (
            pl.col("country") + "|" + pl.col("n_sk")
        ),

        "name_postal": (
            pl.col("country") + "|" +
            pl.col("n_core").str.slice(0, 6) + "|" +
            pl.col("_postal")
        ),

        "phonetic_postal": (
            pl.col("country") + "|" +
            pl.col("n_ph").str.slice(0, 8) + "|" +
            pl.col("_postal")
        ),

        "name_street": (
            pl.col("country") + "|" +
            pl.col("n_core").str.slice(0, 6) + "|" +
            pl.col("_street")
        ),

        "skeleton_street": (
            pl.col("country") + "|" +
            pl.col("n_sk").str.slice(0, 8) + "|" +
            pl.col("_street")
        ),

        # Selective fallback blockers
        "sorted_name": (
            pl.col("country") + "|" +
            pl.col("_sorted")
        ),

        "last_street": (
            pl.col("country") + "|" +
            pl.col("_last") + "|" +
            pl.col("_street")
        ),

        "first_street": (
            pl.col("country") + "|" +
            pl.col("_first") + "|" +
            pl.col("_street")
        ),

        "prefix4_street": (
            pl.col("country") + "|" +
            pl.col("n_core").str.slice(0, 4) + "|" +
            pl.col("_street")
        ),
    }

    lkey = rules[rule].alias("_key")

    rkey = rules[rule].alias("_key")

    l = (
        left
        .select([
            "s1_id",
            "country",
            lkey
        ])
        .filter(pl.col("_key").str.len_chars() > 0)
    )

    r = (
        right
        .select([
            "s2_id",
            "country",
            rkey
        ])
        .filter(pl.col("_key").str.len_chars() > 0)
    )

    # Count block sizes first.
    lc = (
        l.group_by("_key")
        .agg(pl.len().alias("_ln"))
    )

    rc = (
        r.group_by("_key")
        .agg(pl.len().alias("_rn"))
    )

    valid = (
        lc.join(rc, on="_key", how="inner")
        .filter(
            (pl.col("_ln") <= ORIGINAL_MAX_BLOCK) &
            (pl.col("_rn") <= ORIGINAL_MAX_BLOCK)
        )
        .select("_key")
    )

    l = l.join(valid, on="_key", how="inner")
    r = r.join(valid, on="_key", how="inner")

    pairs = (
        l.join(r, on="_key", how="inner")
        .select([
            "s1_id",
            "s2_id"
        ])
    )

    return pairs


def main():

    print("=" * 80)
    print("SELECTIVE CANDIDATE GENERATION")
    print("=" * 80)
    print(f"ORIGINAL_MAX_BLOCK = {ORIGINAL_MAX_BLOCK}")
    print(f"FALLBACK_MAX_BLOCK = {FALLBACK_MAX_BLOCK}")

    s1 = prepare(
        pl.read_parquet(
            f"{NORM}/train_source1_norm.parquet"
        ),
        "s1"
    )

    print(f"S1 rows: {s1.height:,}")

    all_outputs = []

    for source in SOURCES:

        print("\n" + "=" * 80)
        print(f"S1 -> S{source}")
        print("=" * 80)

        target = prepare(
            pl.read_parquet(
                f"{NORM}/train_source{source}_norm.parquet"
            ),
            f"s{source}"
        )

        # Rename target ID uniformly.
        target = target.rename({
            f"s{source}_id": "target_id"
        })

        # Rules are generated separately because target ID differs.
        rules = [
            "exact_name",
            "phonetic_name",
            "skeleton_name",
            "name_postal",
            "phonetic_postal",
            "name_street",
            "skeleton_street",
            "sorted_name",
            "last_street",
            "first_street",
            "prefix4_street",
            "number_addr_sig",
        ]

        rule_outputs = []

        for rule in rules:

            print(f"\n[{rule}]")

            # Build keys directly.
            key_expr = {
                "exact_name":
                    pl.col("country") + "|" + pl.col("n_core"),

                "phonetic_name":
                    pl.col("country") + "|" + pl.col("n_ph"),

                "skeleton_name":
                    pl.col("country") + "|" + pl.col("n_sk"),

                "name_postal":
                    pl.col("country") + "|" +
                    pl.col("n_core").str.slice(0, 6) + "|" +
                    pl.col("_postal"),

                "phonetic_postal":
                    pl.col("country") + "|" +
                    pl.col("n_ph").str.slice(0, 8) + "|" +
                    pl.col("_postal"),

                "name_street":
                    pl.col("country") + "|" +
                    pl.col("n_core").str.slice(0, 6) + "|" +
                    pl.col("_street"),

                "skeleton_street":
                    pl.col("country") + "|" +
                    pl.col("n_sk").str.slice(0, 8) + "|" +
                    pl.col("_street"),

                "sorted_name":
                    pl.col("country") + "|" +
                    pl.col("_sorted"),

                "last_street":
                    pl.col("country") + "|" +
                    pl.col("_last") + "|" +
                    pl.col("_street"),

                "first_street":
                    pl.col("country") + "|" +
                    pl.col("_first") + "|" +
                    pl.col("_street"),

                "prefix4_street":
                    pl.col("country") + "|" +
                    pl.col("n_core").str.slice(0, 4) + "|" +
                    pl.col("_street"),

                "number_addr_sig":
                    pl.col("country") + "|" +
                    pl.col("_street") + "|" +
                    pl.col("_addr_sig"),
            }[rule]

            l = (
                s1
                .select([
                    "s1_id",
                    key_expr.alias("_key")
                ])
                .filter(pl.col("_key").str.len_chars() > 0)
            )

            r = (
                target
                .select([
                    "target_id",
                    key_expr.alias("_key")
                ])
                .filter(pl.col("_key").str.len_chars() > 0)
            )

            lc = l.group_by("_key").agg(
                pl.len().alias("_ln")
            )

            rc = r.group_by("_key").agg(
                pl.len().alias("_rn")
            )

            valid = (
                lc
                .join(rc, on="_key", how="inner")
                .filter(
                    (pl.col("_ln") <= (ORIGINAL_MAX_BLOCK if rule in [
                        "exact_name", "phonetic_name", "skeleton_name",
                        "name_postal", "phonetic_postal",
                        "name_street", "skeleton_street"
                    ] else FALLBACK_MAX_BLOCK)) &
                    (pl.col("_rn") <= (ORIGINAL_MAX_BLOCK if rule in [
                        "exact_name", "phonetic_name", "skeleton_name",
                        "name_postal", "phonetic_postal",
                        "name_street", "skeleton_street"
                    ] else FALLBACK_MAX_BLOCK))
                )
                .select("_key")
            )

            l = l.join(valid, on="_key", how="inner")
            r = r.join(valid, on="_key", how="inner")

            pairs = (
                l.join(r, on="_key", how="inner")
                .select([
                    "s1_id",
                    "target_id"
                ])
                .rename({"target_id": "matched_id"})
            )

            print(
                f"  candidates: {pairs.height:,}"
            )

            if pairs.height:
                rule_outputs.append(pairs)

        # Combine rules and deduplicate.
        combined = (
            pl.concat(rule_outputs)
            .unique(
                subset=["s1_id", "matched_id"],
                maintain_order=False
            )
        )

        print(
            f"\nFINAL S1->S{source}: "
            f"{combined.height:,}"
        )

        path = (
            f"{OUT}/train_s{source}_candidates.parquet"
        )

        combined.write_parquet(
            path,
            compression="zstd"
        )

        print(f"Saved: {path}")

        all_outputs.append(combined)

    # Combined candidate set.
    final = (
        pl.concat(all_outputs)
        .unique(
            subset=["s1_id", "matched_id"],
            maintain_order=False
        )
    )

    final_path = f"{OUT}/train_candidates.parquet"

    final.write_parquet(
        final_path,
        compression="zstd"
    )

    print("\n" + "=" * 80)
    print("COMPLETE")
    print("=" * 80)
    print(f"Total candidates: {final.height:,}")
    print(f"Saved: {final_path}")


if __name__ == "__main__":
    main()
