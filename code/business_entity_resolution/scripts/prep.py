import os
import sys
from pathlib import Path

CODE_DIR = Path(__file__).resolve().parents[1]
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))
import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq

from config import *
from src.utils import progress, memory_status, cleanup

# Allow "python scripts/prep.py" from CODE_DIR/scripts.
SRC = CODE_DIR / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from normalization import normalize_record

SCHEMA = {
    "entity_id": pl.String,
    "business_name": pl.String,
    "business_address": pl.String,
    "country": pl.String,
}

def cache_path(name):
    return CACHE_DIR / f"{name}.parquet"

def count_rows(path):
    return pl.scan_parquet(path).select(pl.len()).collect().item()

def prep_to_parquet(tsv_path, cache_name, is_gt=False):
    out = cache_path(cache_name)
    if out.exists():
        n = count_rows(out)
        print(f"[SKIP] {cache_name}: {n:,} rows already cached")
        return out

    print(f"\n[1/2] TSV -> Parquet: {tsv_path}")
    if is_gt:
        lf = pl.scan_csv(
            tsv_path, separator="\t",
            schema_overrides={
                "source1_entity_id": pl.String,
                "matched_entity_ids": pl.String,
            },
            null_values=["", "NULL", "null", "None"],
        )
        lf.sink_parquet(out, compression="zstd", compression_level=3)
    else:
        # Polars streams batches; we never materialize the whole TSV as a list.
        lf = pl.scan_csv(
            tsv_path, separator="\t",
            schema_overrides=SCHEMA,
            null_values=["", "NULL", "null", "None"],
        )
        # Polars' sink_parquet performs the scan lazily and does not build a
        # Python list of all rows in RAM.
        lf.sink_parquet(out, compression="zstd", compression_level=3)

    n = count_rows(out)
    print(f"[OK] {cache_name}: {n:,} rows -> {out}")
    memory_status(cache_name)
    cleanup(cache_name)
    return out

def prep_norm(cache_name, batch_size=50_000):
    src = cache_path(cache_name)
    out = NORMALIZED_DIR / f"{cache_name}_norm.parquet"
    if out.exists():
        n = count_rows(out)
        print(f"[SKIP] normalized {cache_name}: {n:,} rows already cached")
        return out

    print(f"\n[2/2] Normalizing: {cache_name}")
    writer = None
    total = 0

    try:
        df = pl.read_parquet(src)
        n_rows = df.height

        for start in progress(
            range(0, n_rows, batch_size),
            total=(n_rows + batch_size - 1) // batch_size,
            desc=f"  normalize {cache_name}",
            unit="batch",
        ):
            stop = min(start + batch_size, n_rows)
            rows = df.slice(start, stop - start).select(
                ["entity_id", "country", "business_name", "business_address"]
            ).iter_rows()

            normalized = [normalize_record(r) for r in rows]

            table = pa.Table.from_pylist([
                {
                    "entity_id": r[0],
                    "country": r[1],
                    "business_name": r[2],
                    "business_address": r[3],
                    "n_core": r[4],
                    "n_ph": r[5],
                    "n_sk": r[6],
                    "a_core": r[7],
                }
                for r in normalized
            ])

            if writer is None:
                writer = pq.ParquetWriter(
                    out, table.schema, compression="zstd"
                )
            writer.write_table(table)
            total += len(normalized)

            del normalized, rows, table
            cleanup(f"{cache_name} batch")

        del df
    finally:
        if writer is not None:
            writer.close()

    print(f"[OK] normalized {cache_name}: {total:,} rows -> {out}")
    memory_status(f"normalized {cache_name}")
    return out

def main():
    print("=" * 80)
    print("ENTITY RESOLUTION PREPARATION")
    print("=" * 80)

    for i in (1, 2, 3):
        prep_to_parquet(TRAIN_SOURCES[i], f"train_source{i}")
    prep_to_parquet(TRAIN_GT, "train_ground_truth", is_gt=True)

    for i in (1, 2, 3):
        prep_norm(f"train_source{i}")

    for i in (1, 2, 3):
        prep_to_parquet(TEST_SOURCES[i], f"test_source{i}")
        prep_norm(f"test_source{i}")

    print("\nPREPARATION COMPLETE")
    memory_status("final")

if __name__ == "__main__":
    main()
