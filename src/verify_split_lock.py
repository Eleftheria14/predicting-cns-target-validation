#!/usr/bin/env python3
"""Independently reproduce the homolog-safe split-lock fingerprint from the gold
dataset and check it against the value recorded in the manuscript and in
data/splits/per_split_id_lists.json.

The fingerprint binds ROW IDENTITY (ensembl_gene_id | efo_id) to fold assignment
(fold_seqclust), so any pipeline that reconstructs the same split over the same
rows gets the same hash; a reordered, subset, or augmented row set changes it.
This is what makes the split lock enforceable rather than advisory.

Usage:
    python src/verify_split_lock.py
Exit code 0 on match, 1 on mismatch.
"""
import hashlib
import json
import sys
from pathlib import Path

import pandas as pd

EXPECTED = "sha256:ffa3855aef4eb3901b0e6fdc8e2e9d5cf1088921b039a3608c8da57c54432731"
GOLD = Path("data/gold/cns_cvs_gold_modeling_dataset.parquet")
SPLIT_JSON = Path("data/splits/per_split_id_lists.json")


def p_split_fingerprint(row_ids, folds, n_splits, seed):
    """Content hash binding row identity to fold assignment.

    This is the exact algorithm used by the training and comparison harnesses
    (kernel.p_split_fingerprint); reproduced here so the check has no dependency
    on the harness package.
    """
    pairs = sorted((str(r), int(f)) for r, f in zip(row_ids, folds, strict=False))
    blob = f"n_splits={n_splits};seed={seed};" + ";".join(f"{r}:{f}" for r, f in pairs)
    return "sha256:" + hashlib.sha256(blob.encode()).hexdigest()


def main():
    df = pd.read_parquet(GOLD, columns=["ensembl_gene_id", "efo_id", "fold_seqclust"])
    row_ids = [f"{g}|{d}" for g, d in
               zip(df["ensembl_gene_id"].astype(str), df["efo_id"].astype(str), strict=False)]
    seqfold = df["fold_seqclust"].astype(int).values
    n_splits = len(set(seqfold.tolist()))

    if len(set(row_ids)) != len(row_ids):
        print("FAIL: row identifiers are not unique", file=sys.stderr)
        return 1

    fp = p_split_fingerprint(row_ids, seqfold, n_splits, 0)
    print(f"rows                 : {len(row_ids)} ({n_splits}-fold homolog-safe split)")
    print(f"recomputed fingerprint: {fp}")
    print(f"expected  fingerprint : {EXPECTED}")

    ok = fp == EXPECTED

    # cross-check against the value stored in the split JSON, if present
    if SPLIT_JSON.exists():
        stored = json.loads(SPLIT_JSON.read_text()).get("split_lock_fingerprint", "")
        print(f"stored in split JSON  : {stored}")
        ok = ok and (stored == EXPECTED)

    print("MATCH" if ok else "MISMATCH")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
