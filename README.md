# Amazon ML Challenge 2026 — Entity Resolution

This VM project is based on the supplied challenge notebook and problem statement.

## Current design

- 37 numeric pairwise features.
- LightGBM as the tabular matching model.
- RapidFuzz batch similarity.
- Polars + Parquet for disk-backed preprocessing.
- Progress bars for long-running operations.
- Explicit RSS memory guard with a hard ceiling of 60 GB.
- No `blocking.py` is included or executed yet, by request.

## Important next stage

The challenge requires a candidate set before final matching. The official
`candidate_pairs.tsv` is the final candidate list actually scored by the model.
Every final match must be present in that candidate set.

We will add the candidate-generation/blocking stage separately after the
preprocessing and feature/model code is verified on the VM.

## Data layout

dataset/train/
- train_source1.tsv
- train_source2.tsv
- train_source3.tsv
- train_ground_truth.tsv

dataset/test/
- test_source1.tsv
- test_source2.tsv
- test_source3.tsv

## Commands

From `~/er/code/business_entity_resolution/scripts`:

```bash
python prep.py
python train.py
python inference.py
python evaluate.py
```

Use the virtual environment:

```bash
source ~/er/.venv/bin/activate
```

## Memory safety

The VM has 80 GB RAM, but the code intentionally treats 60 GB RSS as a hard
stop and 50 GB as a warning level. Feature computation is chunked, and the
training script caps the number of feature rows at `MAX_TRAIN_PAIRS`.

Do not increase these limits until actual RSS has been measured.

## Challenge-specific rules

The supplied problem statement says:
- all test Source-1 entities must appear in the final output;
- empty match lists are valid for singletons;
- final matches must be a subset of candidate pairs;
- output is TSV;
- country labels must remain open-set because France occurs in test;
- external entity lookup/data augmentation/geocoding is prohibited.
