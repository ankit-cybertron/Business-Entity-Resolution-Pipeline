# Business Entity Resolution Pipeline

A large-scale entity resolution pipeline for matching business records across multiple noisy data sources.

## Current design

- 37 numeric pairwise features.
- LightGBM as the tabular matching model.
- RapidFuzz batch similarity.
- Polars + Parquet for disk-backed preprocessing.
- Progress bars for long-running operations.
- Explicit RSS memory guard with a hard ceiling of 60 GB.
- Candidate generation and blocking are used to reduce the pairwise search space.

## Candidate generation

Candidate generation reduces the number of record pairs that need to be
processed by creating a restricted set of plausible candidate matches.

The pipeline uses multiple blocking strategies based on normalized business
names, phonetic representations, address information, street information,
token relationships, and numeric address signatures.

Final matching is performed only on generated candidate pairs.

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


## Output requirements

The final output is validated before it is produced.

The pipeline ensures that:

- All required Source-1 entities are represented in the final output.
- Empty match lists are supported for entities with no resolved matches.
- Final matches are restricted to generated candidate pairs.
- The output is written in TSV format.
- Country values are treated as open-set categorical information.
- External entity lookup, data augmentation, and geocoding are not part of the pipeline.

## Project status

The pipeline is organized as independent stages so that preprocessing,
candidate generation, feature engineering, model training, inference, and
validation can be executed and evaluated separately.

The project is designed to support further improvements in candidate
generation, feature engineering, machine learning models, and large-scale
inference.
