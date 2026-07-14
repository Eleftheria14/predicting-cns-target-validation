#!/usr/bin/env python3
"""
Tests for cvs_train_harness.py — validate the harness contracts on a tiny synthetic frame,
runnable locally (CPU, no Spark, no GPU). Focus: the invariants that make the comparison valid.

Run:  pytest test_cvs_train_harness.py -q
The harness is import-guarded (see conftest note): we exercise its pure functions by loading
the module with monkeypatched globals so no real data / GPU is needed.
"""
import importlib
import pathlib
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent / "training_harness"))


# --------------------------------------------------------------------------------------
# Load the harness as a LIBRARY. The module is import-clean (no side effects at import); we
# point its CONFIG at synthetic fixtures on disk, then call load_data() so it derives its OWN
# globals (interp_cols, emb_cols, splits, y_perm, ...) exactly as in production. The heavy
# main() execution loop and output writes are never invoked. This replaces the old approach of
# string-splitting the source at a magic comment — which coupled the test to a literal line.
# --------------------------------------------------------------------------------------
def load_harness_funcs(gold_path=None, manifest_path=None, mono_path=None, pca_k=4):
    import cvs_train_harness as H
    importlib.reload(H)                      # fresh module state per fixture (globals re-derived)
    if gold_path:
        H.GOLD, H.MANIFEST, H.MONO, H.PCA_K = gold_path, manifest_path, mono_path, pca_k
    H.load_data()
    return H


# --------------------------------------------------------------------------------------
# Synthetic frame: 60 genes x up-to-3 diseases, a homolog cluster that bridges 2 genes,
# 6 interpretable cols + 8 "embedding" cols, a binary label. Written to disk so the harness
# module loads it through its own GOLD/MANIFEST/MONO paths and derives its real globals.
# --------------------------------------------------------------------------------------
@pytest.fixture(scope="module")
def synth(tmp_path_factory):
    rng = np.random.default_rng(0)
    rows = []
    for gi in range(80):
        g = f"ENSG{gi:05d}"
        clust = "C_BRIDGE" if gi in (0, 1) else f"C{gi}"   # genes 0 & 1 share a homolog cluster
        # spread pairs across 8 diseases so GroupKFold(5) on disease has enough groups
        for di in rng.choice(8, size=int(rng.integers(1, 5)), replace=False):
            rows.append({"ensembl_gene_id": g, "efo_id": f"D{di}", "gene_symbol": g,
                         "homol_cluster": clust, "fold_seqclust": gi % 5})
    df = pd.DataFrame(rows)
    n = len(df)
    for j in range(6):
        df[f"iv{j}"] = rng.normal(size=n)
    for j in range(8):
        df[f"emb{j}"] = rng.normal(size=n)
    df["label_consensus"] = (df["iv0"] + rng.normal(scale=0.5, size=n) > 0).astype(int)
    man = pd.DataFrame(
        [{"column": f"iv{j}", "block": "intrinsic_84"} for j in range(6)] +
        [{"column": f"emb{j}", "block": "esmc6b"} for j in range(8)])
    d = tmp_path_factory.mktemp("synth")
    gp, mp, kp = d / "gold.parquet", d / "manifest.csv", d / "mono.json"
    df.to_parquet(gp); man.to_csv(mp, index=False); kp.write_text("{}")
    return df, man, str(gp), str(mp), str(kp)


@pytest.fixture(scope="module")
def H(synth):
    _df, _man, gp, mp, kp = synth
    return load_harness_funcs(gold_path=gp, manifest_path=mp, mono_path=kp, pca_k=4)


def test_module_imports_without_side_effects():
    # The harness must be importable as a library WITHOUT reading data or touching the GPU,
    # so the sweep job and these tests can import it. df stays None until load_data() is called.
    import importlib

    import cvs_train_harness as H
    importlib.reload(H)
    assert H.df is None, "importing the harness must not load data (no side effects at import)"
    assert callable(H.load_data) and callable(H.main), "library API: load_data() + main()"


def test_module_loads_config(H):
    # registries and config exist and agree
    assert set(H.MODELS).issubset(set(H.REGISTRY)), "every MODELS entry must be in REGISTRY"
    assert set(H.SPLITS).issubset(set(H.SPLIT_REG)), "every SPLITS entry must be in SPLIT_REG"
    assert "knn" in H.REGISTRY, "KNN must be a registered comparable family"
    assert H.HEADLINE in H.SPLITS


def test_split_registry_shapes(H, synth, monkeypatch):
    df, man = synth[0], synth[1]
    for name, fn in H.SPLIT_REG.items():
        folds = fn(df)
        assert len(folds) >= 2, f"{name} produced <2 folds"
        for tr, te in folds:
            assert tr.dtype == bool and te.dtype == bool
            assert not (tr & te).any(), f"{name}: train/test overlap"
            assert te.sum() > 0 and tr.sum() > 0, f"{name}: empty side"


def test_group_splits_are_disjoint_by_group(H, synth):
    df, man = synth[0], synth[1]
    gene = df["ensembl_gene_id"].values
    for tr, te in H._split_gene(df):
        assert not (set(gene[tr]) & set(gene[te])), "gene split leaks a gene across train/test"
    dis = df["efo_id"].values
    for tr, te in H._split_disease(df):
        assert not (set(dis[tr]) & set(dis[te])), "disease split leaks a disease across train/test"


def test_cluster_safe_drops_homolog_bridge(H, synth):
    df, man = synth[0], synth[1]
    # gene 0 and gene 1 share cluster C_BRIDGE. In the gene split, when one is in test,
    # the other must NOT be in train (cluster-safe exclusion).
    clust = df["homol_cluster"].values
    for tr, te in H._split_gene(df):
        if "C_BRIDGE" in set(clust[te]):
            assert "C_BRIDGE" not in set(clust[tr]), "homolog bridge not excluded from train"


def test_random_split_is_not_cluster_safe(H, synth):
    df, man = synth[0], synth[1]
    # random is the optimistic reference — deliberately allows a cluster on both sides
    clust = df["homol_cluster"].values
    bridged = any("C_BRIDGE" in set(clust[te]) and "C_BRIDGE" in set(clust[tr])
                  for tr, te in H._split_random(df))
    assert bridged, "random split should NOT apply cluster exclusion (it is the optimistic ref)"


def test_build_fold_no_leakage_and_shape(H, synth):
    df, man = synth[0], synth[1]
    tr = df["fold_seqclust"].values != 0
    te = df["fold_seqclust"].values == 0
    Xtr, Xte = H.build_fold(tr, te)
    assert Xtr.shape[1] == Xte.shape[1] == len(H.interp_cols) + 4
    assert np.isfinite(Xtr).all() and np.isfinite(Xte).all(), "NaNs leaked into feature matrix"
    assert Xtr.shape[0] == tr.sum() and Xte.shape[0] == te.sum()


def test_all_families_train_and_predict(H, synth):
    df, man = synth[0], synth[1]
    tr = df["fold_seqclust"].values != 0
    te = df["fold_seqclust"].values == 0
    Xtr, Xte = H.build_fold(tr, te)
    ytr = df["label_consensus"].values[tr]
    Xi_tr = np.nan_to_num(df.loc[tr, H.interp_cols].values)
    Xi_te = np.nan_to_num(df.loc[te, H.interp_cols].values)
    edges = H.binarize_fit(Xi_tr)
    Btr, bnames = H.binarize_apply(Xi_tr, edges); Bte, _ = H.binarize_apply(Xi_te, edges)
    ctx = dict(first=True, Xi_tr=Xi_tr, Xi_te=Xi_te, Btr=Btr, Bte=Bte, bnames=bnames)
    for name in H.MODELS:
        if name in ("ga2m", "nam", "ripper"):
            continue  # optional deps (interpret-core/torch/wittgenstein); covered on the Spark run
        proba, fitted, rep = H.REGISTRY[name](Xtr, ytr, Xte, ctx)
        proba = np.asarray(proba, dtype=float)
        assert proba.shape[0] == te.sum(), f"{name}: wrong prediction length"
        assert np.isfinite(proba).all(), f"{name}: non-finite probabilities"
        assert (proba >= 0).all() and (proba <= 1).all(), f"{name}: probabilities out of [0,1]"


def test_binarizer_train_fit_applies_to_test(H, synth):
    df, man = synth[0], synth[1]
    Xi = np.nan_to_num(df[H.interp_cols].values)
    edges = H.binarize_fit(Xi[:40])
    B1, names1 = H.binarize_apply(Xi[:40], edges)
    B2, names2 = H.binarize_apply(Xi[40:], edges)   # same thresholds applied to held-out rows
    assert names1 == names2 and B1.shape[1] == B2.shape[1]
    assert set(np.unique(B1)).issubset({0, 1})


# --------------------------------------------------------------------------------------
# helper: bind the module-level globals the functions close over (interp_cols, emb_cols, ...)
# so the pure functions run against the synthetic frame.
# --------------------------------------------------------------------------------------
if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-q"]))
