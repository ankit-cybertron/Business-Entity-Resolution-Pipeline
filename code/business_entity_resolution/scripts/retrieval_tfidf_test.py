import os
import re
import time
import gc
from sparse_dot_topn import sp_matmul_topn

import numpy as np
import polars as pl
from sklearn.feature_extraction.text import TfidfVectorizer


BASE = os.path.expanduser("~/er")
NORM = f"{BASE}/cache/normalized"

S1_SAMPLE = 1_000
TOP_K = 100
MAX_FEATURES = 100_000


def clean(x):
    if x is None:
        return ""
    return re.sub(
        r"[^a-z0-9]+",
        " ",
        str(x).lower(),
    ).strip()


def load_data(source):

    s1 = (
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

    target = (
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

    return s1, target


def get_ground_truth(s1_ids, source):

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


def retrieve(s1, target, field):

    print()
    print("=" * 80)
    print(f"CHARACTER TF-IDF: {field}")
    print("=" * 80)

    start = time.time()

    query_text = [
        clean(x)
        for x in s1[field].to_list()
    ]

    target_text = [
        clean(x)
        for x in target[field].to_list()
    ]

    vectorizer = TfidfVectorizer(
        analyzer="char",
        ngram_range=(3, 5),
        min_df=2,
        max_features=MAX_FEATURES,
        dtype=np.float32,
        sublinear_tf=True,
    )

    print("Fitting vectorizer...")

    target_matrix = vectorizer.fit_transform(
        target_text
    )

    print(
        f"Target matrix: {target_matrix.shape}"
    )

    query_matrix = vectorizer.transform(
        query_text
    )

    print(
        f"Query matrix: {query_matrix.shape}"
    )

    results = []

    batch_size = 50

    for start_idx in range(
        0,
        query_matrix.shape[0],
        batch_size,
    ):

        end_idx = min(
            start_idx + batch_size,
            query_matrix.shape[0],
        )

        scores = sp_matmul_topn(
            query_matrix[start_idx:end_idx],
            target_matrix.T,
            top_n=TOP_K,
            threshold=0.0,
            sort=True,
            n_threads=8,
        )

        for local in range(
            scores.shape[0]
        ):

            row = scores.getrow(local)

            if row.nnz == 0:
                continue

            data = row.data
            indices = row.indices

            k = min(TOP_K, len(data))

            if len(data) > k:
                keep = np.argpartition(
                    data,
                    -k,
                )[-k:]

                data = data[keep]
                indices = indices[keep]

            order = np.argsort(
                data
            )[::-1]

            s1_idx = start_idx + local

            for pos in order:

                results.append((
                    s1["entity_id"][s1_idx],
                    target["entity_id"][int(indices[pos])],
                ))

        if start_idx % 250 == 0:
            print(
                f"Processed "
                f"{min(end_idx, query_matrix.shape[0]):,}"
                f"/{query_matrix.shape[0]:,} S1"
            )

    result = pl.DataFrame(
        results,
        schema=[
            "s1_id",
            "matched_id",
        ],
        orient="row",
    ).unique()

    elapsed = time.time() - start

    print(
        f"Retrieved pairs: {result.height:,}"
    )

    print(
        f"Runtime: {elapsed:.1f}s"
    )

    return result


def main():

    print("=" * 80)
    print("TF-IDF RETRIEVAL TEST")
    print("=" * 80)
    print(f"S1 sample:    {S1_SAMPLE:,}")
    print(f"TOP_K:        {TOP_K}")
    print(f"MAX_FEATURES: {MAX_FEATURES}")
    print("=" * 80)

    for source in [2, 3]:

        print()
        print(
            "#" * 80
        )
        print(
            f"S1 -> S{source}"
        )
        print(
            "#" * 80
        )

        s1, target = load_data(source)

        print(
            f"S1:     {s1.height:,}"
        )
        print(
            f"S{source}: {target.height:,}"
        )

        gt = get_ground_truth(
            s1["entity_id"],
            source,
        )

        print(
            f"Ground-truth pairs: {gt.height:,}"
        )

        for field in [
            "n_core",
            "a_core",
        ]:

            result = retrieve(
                s1,
                target,
                field,
            )

            found = (
                gt
                .join(
                    result,
                    on=[
                        "s1_id",
                        "matched_id",
                    ],
                    how="inner",
                )
                .height
            )

            recall = (
                found / gt.height
                if gt.height
                else 0
            )

            print()
            print(
                f"{field} RECALL"
            )
            print(
                f"Found:  {found:,}"
            )
            print(
                f"Total:  {gt.height:,}"
            )
            print(
                f"Recall: {recall:.4%}"
            )

            del result
            gc.collect()


if __name__ == "__main__":
    main()
