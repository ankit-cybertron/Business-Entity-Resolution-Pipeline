from config import *
import polars as pl

def main():
    print("=" * 80)
    print("OUTPUT VALIDATION")
    print("=" * 80)

    test_s1 = pl.read_csv(
        TEST_SOURCES[1], separator="\t",
        schema_overrides={"entity_id": pl.String}
    )["entity_id"].to_list()

    valid_s23 = set()
    for i in (2, 3):
        ids = pl.read_csv(
            TEST_SOURCES[i], separator="\t",
            schema_overrides={"entity_id": pl.String}
        )["entity_id"].to_list()
        valid_s23.update(ids)

    if not MATCHING_TSV.exists():
        raise FileNotFoundError(MATCHING_TSV)
    if not CANDIDATE_TSV.exists():
        raise FileNotFoundError(CANDIDATE_TSV)

    matches = pl.read_csv(
        MATCHING_TSV, separator="\t", has_header=True,
        schema_overrides={"source1_entity_id": pl.String,
                          "matched_entity_ids": pl.String}
    ).fill_null("")

    candidates = pl.read_csv(
        CANDIDATE_TSV, separator="\t", has_header=True,
        schema_overrides={"source1_entity_id": pl.String,
                          "candidate_entity_ids": pl.String}
    ).fill_null("")

    errors = []

    if matches.height != len(test_s1):
        errors.append(
            f"matching_results rows={matches.height:,}, expected={len(test_s1):,}"
        )

    if candidates.height != len(test_s1):
        errors.append(
            f"candidate_pairs rows={candidates.height:,}, expected={len(test_s1):,}"
        )

    m_ids = matches["source1_entity_id"].to_list()
    c_ids = candidates["source1_entity_id"].to_list()

    if len(set(m_ids)) != len(m_ids):
        errors.append("duplicate source1_entity_id in matching_results.tsv")
    if len(set(c_ids)) != len(c_ids):
        errors.append("duplicate source1_entity_id in candidate_pairs.tsv")

    c_map = dict(zip(c_ids, candidates["candidate_entity_ids"].to_list()))

    for s1, raw in zip(m_ids, matches["matched_entity_ids"].to_list()):
        mids = [x for x in raw.split(",") if x]
        cand = set(x for x in c_map.get(s1, "").split(",") if x)

        if len(mids) != len(set(mids)):
            errors.append(f"{s1}: duplicate matched ID")
        for x in mids:
            if not (x.startswith("S2-") or x.startswith("S3-")):
                errors.append(f"{s1}: invalid prefix {x}")
            if x not in valid_s23:
                errors.append(f"{s1}: unknown test ID {x}")
            if x not in cand:
                errors.append(f"{s1}: match {x} absent from candidate list")

    missing = set(test_s1) - set(m_ids)
    if missing:
        errors.append(f"missing {len(missing):,} Source-1 test entities")

    if errors:
        print(f"FAILED: {len(errors)} issue(s)")
        for e in errors[:50]:
            print(" -", e)
        raise SystemExit(1)

    print("PASS: output structure satisfies the documented rules.")
    print("Every test Source-1 entity has one row; matched IDs are S2/S3 IDs "
          "and are contained in candidate_pairs.tsv.")

if __name__ == "__main__":
    main()
