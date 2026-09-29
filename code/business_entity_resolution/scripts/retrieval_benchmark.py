import os
import re
import time
import gc

import numpy as np
import polars as pl

from rapidfuzz import process, fuzz
from sklearn.feature_extraction.text import TfidfVectorizer


BASE = os.path.expanduser("~/er")
NORM = f"{BASE}/cache/normalized"

S1_SAMPLE = 1_000
TOP_K = 100


def clean(x):
    if x is None:
        return ""
    return re.sub(r"[^a-z0-9]+", " ", str(x).lower()).strip()


def load_s1():
    return (
        pl.read_parquet(
            f"{NORM}/train_source1_norm.parquet"
        )
        .select([
            "entity_id",
            "country",
            "n_core",
            "a_core",
        ])
        .head(S1_SAMPLE)
    )


def load_target(source):
    return (
        pl.read_parquet(
            f"{NORM}/train_source{source}_norm.parquet"
        )
        .select([
            "entity_id",
            "country",
            "n_core",
            "a_core",
        ])
    )


def ground_truth(s1_ids, source):

    gt = pl.read_parquet(
        f"{BASE}/cache/train_ground_truth.parquet"
    )

    return (
        gt
        .select([
            "source1_entity_id",
            "matched_entity_ids",
        ])
        .filter(
            pl.col("source1_entity_id")
            .is_in(s1_ids.implode())
        )
        .with_columns(
            pl.col("matched_entity_ids")
            .fill_null("")
            .str.split(",")
            .alias("matches")
        )
        .explode("matches")
        .rename({
            "source1_entity_id": "s1_id",
            "matches": "matched_id",
        })
        .filter(
            pl.col("matched_id")
            .str.starts_with(f"S{source}-")
        )
        .select([
            "s1_id",
            "matched_id",
        ])
        .unique()
    )


def evaluate(retrieved, gt):

    found = (
        gt
        .join(
            retrieved,
            on=["s1_id", "matched_id"],
            how="inner",
        )
        .height
    )

    total = gt.height

    return (
        found,
        total,
        found / total if total else 0.0,
    )


# ============================================================
# FUZZY NAME RETRIEVAL
# ============================================================

def fuzzy_name_retrieval(s1, target):

    print()
    print("=" * 80)
    print("FUZZY NAME RETRIEVAL")
    print("=" * 80)

    target_names = [
        clean(x)
        for x in target["n_core"].to_list()
    ]

    target_ids = target["entity_id"].to_list()

    rows = []

    start = time.time()

    for row in s1.iter_rows(named=True):

        query = clean(row["n_core"])

        if not query:
            continue

        # RapidFuzz returns the strongest TOP_K
        # target strings.
        matches = process.extract(
            query,
            target_names,
            scorer=fuzz.WRatio,
            limit=TOP_K,
            score_cutoff=45,
        )

        for _, score, idx in matches:

            rows.append({
                "s1_id": row["entity_id"],
                "matched_id": target_ids[idx],
                "score": score,
            })

    result = pl.DataFrame(rows)

    elapsed = time.time() - start

    print(f"Candidates: {result.height:,}")
    print(f"Time:       {elapsed:.1f}s")

    return result


# ============================================================
# CHARACTER TF-IDF NAME RETRIEVAL
# ============================================================

def tfidf_name_retrieval(s1, target):

    print()
    print("=" * 80)
    print("CHARACTER TF-IDF NAME RETRIEVAL")
    print("=" * 80)

    start = time.time()

    s1_text = [
        clean(x)
        for x in s1["n_core"].to_list()
    ]

    target_text = [
        clean(x)
        for x in target["n_core"].to_list()
    ]

    vectorizer = TfidfVectorizer(
        analyzer="char",
        ngram_range=(3, 5),
        min_df=2,
        max_features=100_000,
        dtype=np.float32,
    )

    print("Fitting TF-IDF...")

    target_matrix = vectorizer.fit_transform(
        target_text
    )

    query_matrix = vectorizer.transform(
        s1_text
    )

    print(
        f"Target matrix: {target_matrix.shape}"
    )

    rows = []

    batch_size = 100

    for start_idx in range(
        0,
        query_matrix.shape[0],
        batch_size,
    ):

        end_idx = min(
            start_idx + batch_size,
            query_matrix.shape[0],
        )

        scores = (
            query_matrix[start_idx:end_idx]
            @ target_matrix.T
        )

        for local in range(
            scores.shape[0]
        ):

            row = scores.getrow(local)

            if row.nnz == 0:
                continue

            data = row.data
            indices = row.indices

            if len(data) > TOP_K:

                keep = np.argpartition(
                    data,
                    -TOP_K,
                )[-TOP_K:]

                data = data[keep]
                indices = indices[keep]

            order = np.argsort(
                data
            )[::-1]

            s1_idx = start_idx + local

            for pos in order:

                rows.append({
                    "s1_id":
                        s1["entity_id"][s1_idx],

                    "matched_id":
                        target["entity_id"][
                            indices[pos]
                        ],

                    "score":
                        float(data[pos]),
                })

    result = pl.DataFrame(rows)

    elapsed = time.time() - start

    print(f"Candidates: {result.height:,}")
    print(f"Time:       {elapsed:.1f}s")

    return result


# ============================================================
# MAIN BENCHMARK
# ============================================================

def main():

    print("=" * 80)
    print("ENTITY RESOLUTION RETRIEVAL BENCHMARK")
    print("=" * 80)

    s1 = load_s1()

    print(
        f"S1 sample: {s1.height:,}"
    )

    for source in [2, 3]:

        print()
        print("#" * 80)
        print(f"S1 -> S{source}")
        print("#" * 80)

        target = load_target(source)

        print(
            f"Target rows: {target.height:,}"
        )

        gt = ground_truth(
            s1["entity_id"],
            source,
        )

        print(
            f"Ground-truth pairs: {gt.height:,}"
        )

        # ----------------------------------------------------
        # 1. FUZZY NAME
        # ----------------------------------------------------

        fuzzy = fuzzy_name_retrieval(
            s1,
            target,
        )

        found, total, recall = evaluate(
            fuzzy.select([
                "s1_id",
                "matched_id",
            ]),
            gt,
        )

        print()
        print("FUZZY NAME")
        print(f"Found:  {found:,} / {total:,}")
        print(f"Recall: {recall:.4%}")

        del fuzzy
        gc.collect()

        # ----------------------------------------------------
        # 2. CHARACTER TF-IDF NAME
        # ----------------------------------------------------

        tfidf = tfidf_name_retrieval(
            s1,
            target,
        )

        found, total, recall = evaluate(
            tfidf.select([
                "s1_id",
                "matched_id",
            ]),
            gt,
        )

        print()
        print("CHAR TF-IDF NAME")
        print(f"Found:  {found:,} / {total:,}")
        print(f"Recall: {recall:.4%}")

        del tfidf
        del target

        gc.collect()


if __name__ == "__main__":
    main()
