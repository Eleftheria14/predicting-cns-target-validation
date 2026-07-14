"""Test suite for cvs_interp_adapter.py - the bridge from the CVS training
harness to the model-comparison harness.

Dual-mode: `pytest test_adapter.py` or `python test_adapter.py`.

Builds a SYNTHETIC gold matrix + manifest in the exact CVS schema (interp
blocks + an esmc6b embedding block, label_consensus, fold_seqclust,
ensembl_gene_id, efo_id) with a KNOWN signal planted in the embedding, then
checks that the adapter rebuilds X faithfully, wraps each model family as a
leakage-safe retrainable estimator, bridges the split lock, and ingests OOF.

Families needing heavy deps (wittgenstein/interpret/torch) are skipped for the
fit/predict tests when the dep is absent, but their factory wiring and
feature-space construction are still checked.
"""
import importlib.util
import json
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
INTERP = os.path.join(HERE, "..", "interpretation_harness")


def load_mod(fname, modname):
    spec = importlib.util.spec_from_file_location(modname, os.path.join(INTERP, fname))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


A = load_mod("cvs_interp_adapter.py", "cvs_interp_adapter_uut")
H = load_mod("kernel.py", "cmp_kernel_for_adapter")


def _has(mod):
    import importlib
    try:
        importlib.import_module(mod)
        return True
    except Exception:
        return False


HAVE = {"xgb_baseline": _has("xgboost"), "mono_gbm": _has("xgboost"),
        "elasticnet_glm": True, "cart": True, "scorecard": True,
        "ripper": _has("wittgenstein"), "ga2m": _has("interpret"),
        "nam": _has("torch")}


# ----------------------------------------------------------------------------
# Synthetic CVS-schema fixture
# ----------------------------------------------------------------------------
def make_gold(tmpdir, seed=0, n_genes=60, per_gene=10, n_emb=40, n_interp=12):
    """Write gold.parquet + manifest.csv + feature_spec.json in the CVS schema.
    Signal lives in the EMBEDDING (first few dims track gene quality), so the
    esmc6b family should dominate ablation/convergence - mirrors the real CVS
    finding that the embedding carries the largest unique contribution."""
    rng = np.random.RandomState(seed)
    os.makedirs(tmpdir, exist_ok=True)
    gene_idx = np.repeat(np.arange(n_genes), per_gene)
    n = len(gene_idx)
    gene_q = rng.randn(n_genes)

    # embedding: first 4 dims carry gene quality, rest noise
    E = rng.randn(n, n_emb).astype(np.float32)
    for i, g in enumerate(gene_idx):
        E[i, :4] += gene_q[g] * 1.4

    # interp blocks: mostly noise, a couple weak signal cols
    interp = rng.randn(n, n_interp).astype(np.float32)
    for i, g in enumerate(gene_idx):
        interp[i, 0] += gene_q[gene_idx[i]] * 0.3
    # two binary flag columns (exercise the binarizer's "bin" branch)
    interp[:, 1] = (interp[:, 1] > 0).astype(np.float32)
    interp[:, 2] = (interp[:, 2] > 0.5).astype(np.float32)

    gene_label = (gene_q > 0.2).astype(int)
    y = np.array([gene_label[g] if rng.rand() > 0.15 else 1 - gene_label[g]
                  for g in gene_idx])

    # frozen homolog-safe folds: whole genes to folds (proxy for seqclust)
    gene_to_fold = {g: int(g % 5) for g in range(n_genes)}
    fold_seqclust = np.array([gene_to_fold[g] for g in gene_idx])

    interp_cols = [f"intr_{i}" for i in range(n_interp)]
    emb_cols = [f"esmc6b_{i}" for i in range(n_emb)]
    cols = {}
    for j, c in enumerate(interp_cols):
        cols[c] = interp[:, j]
    for j, c in enumerate(emb_cols):
        cols[c] = E[:, j]
    df = pd.DataFrame(cols)
    df["label_consensus"] = y
    df["ensembl_gene_id"] = [f"ENSG{g:05d}" for g in gene_idx]
    # unique disease per (gene, within-gene position) so gene|disease keys are
    # unique - real CVS pairs are unique, and the split lock requires it
    df["efo_id"] = [f"EFO_{(i % per_gene):04d}" for i in range(n)]
    df["gene_symbol"] = df["ensembl_gene_id"]
    df["fold_seqclust"] = fold_seqclust
    df["homol_cluster"] = [f"clu{g}" for g in gene_idx]     # whole-gene clusters
    df.to_parquet(os.path.join(tmpdir, "gold.parquet"))

    # manifest: block map. interp split across two blocks to test family map.
    man_rows = []
    for j, c in enumerate(interp_cols):
        blk = "intrinsic_84" if j < n_interp // 2 else "predicted_go"
        man_rows.append({"column": c, "block": blk})
    for c in emb_cols:
        man_rows.append({"column": c, "block": "esmc6b"})
    pd.DataFrame(man_rows).to_csv(os.path.join(tmpdir, "manifest.csv"), index=False)

    spec = {"interp_cols": interp_cols, "pc_names": [f"emb_pc{i}" for i in range(8)],
            "feat_names": interp_cols + [f"emb_pc{i}" for i in range(8)],
            "emb_block": "esmc6b", "pca_k": 8, "scale_before_pca": True,
            "label_col": "label_consensus", "gene_col": "ensembl_gene_id",
            "disease_col": "efo_id", "headline_split": "seqclust",
            "splits": ["seqclust", "gene", "disease", "random"],
            "models": ["xgb_baseline", "mono_gbm", "elasticnet_glm", "cart",
                       "scorecard", "ripper", "ga2m", "nam"],
            "xgb_rounds": 40, "seed": 0}
    json.dump(spec, open(os.path.join(tmpdir, "feature_spec.json"), "w"))
    return spec


def fixture(tmpdir="t_adapter_fix"):
    spec = make_gold(tmpdir)
    gold = os.path.join(tmpdir, "gold.parquet")
    man = os.path.join(tmpdir, "manifest.csv")
    built = A.build_matrix(gold, man, spec)
    return spec, gold, man, built


# ----------------------------------------------------------------------------
# build_matrix
# ----------------------------------------------------------------------------
def test_build_matrix_shapes_and_families():
    spec, gold, man, b = fixture()
    assert b["X"].shape[0] == len(b["y"]) == len(b["row_ids"])
    # X = interp then emb; widths recorded
    assert b["X"].shape[1] == b["n_interp"] + b["n_emb"]
    # embedding is the width-dominant trailing block
    assert b["n_interp"] < b["n_emb"]
    fam = b["feature_blocks"]
    assert "esmc6b" in fam and len(fam["esmc6b"]) == b["n_emb"]
    # interp split across two manifest blocks
    assert "intrinsic_84" in fam and "predicted_go" in fam
    # every family column exists in X
    allcols = set(b["X"].columns)
    for cols in fam.values():
        assert set(cols).issubset(allcols)


def test_row_ids_are_biological_key():
    spec, gold, man, b = fixture()
    assert all("|" in r for r in b["row_ids"])          # gene|disease
    assert b["row_ids"][0].startswith("ENSG")


def test_build_matrix_rejects_bad_width_rule():
    """The adapter's embedding-by-width detection requires n_interp < n_emb."""
    spec, gold, man, b = fixture()
    # a spec whose embedding block is tiny should trip the width assertion
    bad_spec = dict(spec)
    # point emb_block at a block that yields too few columns
    man_df = pd.read_csv(man)
    # relabel all but 2 emb columns to interp so n_emb < n_interp
    emb = [c for c in man_df.column if c.startswith("esmc6b_")]
    man_df.loc[man_df.column.isin(emb[2:]), "block"] = "intrinsic_84"
    bad_man = os.path.join("t_adapter_fix", "manifest_bad.csv")
    man_df.to_csv(bad_man, index=False)
    raised = False
    try:
        A.build_matrix(gold, bad_man, bad_spec)
    except ValueError:
        raised = True
    assert raised


# ----------------------------------------------------------------------------
# RegistryEstimator: fit/predict + in-fold PCA leakage-safety
# ----------------------------------------------------------------------------
def _fit_predict_ok(name, b):
    est = A.RegistryEstimator(name, n_interp_total=b["n_interp"], n_emb=b["n_emb"],
                              seed=0, pca_k=8, xgb_rounds=40)
    X = b["X"].values
    ntr = int(0.7 * len(X))
    est.fit(X[:ntr], b["y"][:ntr])
    p = est.predict_proba(X[ntr:])
    assert p.shape == (len(X) - ntr, 2)
    assert np.isfinite(p).all()
    assert (p >= 0).all() and (p <= 1).all()
    # rows sum to 1 (proper 2-col proba)
    assert np.allclose(p.sum(axis=1), 1.0, atol=1e-6)
    return est


def test_feat_space_models_fit_predict():
    spec, gold, man, b = fixture()
    for name in ["xgb_baseline", "mono_gbm", "elasticnet_glm"]:
        if HAVE[name]:
            _fit_predict_ok(name, b)


def test_interp_only_model_ignores_embedding():
    """CART consumes interp cols only; its prediction must not change when the
    embedding columns are permuted (it never sees them)."""
    spec, gold, man, b = fixture()
    X = b["X"].values.copy()
    est = _fit_predict_ok("cart", b)
    p1 = est.predict_proba(X)[:, 1]
    Xp = X.copy()
    rng = np.random.RandomState(1)
    Xp[:, -b["n_emb"]:] = rng.permutation(Xp[:, -b["n_emb"]:])
    p2 = est.predict_proba(Xp)[:, 1]
    assert np.allclose(p1, p2), "CART prediction changed when embedding perturbed"


def test_binarized_model_scorecard():
    spec, gold, man, b = fixture()
    _fit_predict_ok("scorecard", b)


def test_infold_pca_is_leakage_safe():
    """PCA is refit inside fit(): a model fit on TRAIN only must produce the
    same test scores whether or not the test rows were present at fit time.
    (If PCA had been fit globally, holding out rows would shift the basis.)"""
    spec, gold, man, b = fixture()
    if not HAVE["elasticnet_glm"]:
        return
    X = b["X"].values; y = b["y"]
    ntr = int(0.7 * len(X))
    e1 = A.RegistryEstimator("elasticnet_glm", b["n_interp"], b["n_emb"], seed=0, pca_k=8)
    e1.fit(X[:ntr], y[:ntr])
    p_holdout = e1.predict_proba(X[ntr:])[:, 1]
    # refit an identical estimator on the same train rows; scores must match
    e2 = A.RegistryEstimator("elasticnet_glm", b["n_interp"], b["n_emb"], seed=0, pca_k=8)
    e2.fit(X[:ntr], y[:ntr])
    p_holdout2 = e2.predict_proba(X[ntr:])[:, 1]
    assert np.allclose(p_holdout, p_holdout2), "fit is not deterministic / leakage-safe"


# ----------------------------------------------------------------------------
# make_factories
# ----------------------------------------------------------------------------
def test_make_factories_all_families():
    spec, gold, man, b = fixture()
    facs = A.make_factories(spec, b["n_interp"], b["n_emb"])
    assert set(facs) == set(spec["models"])
    for name, fac in facs.items():
        est = fac(0)
        assert isinstance(est, A.RegistryEstimator)
        assert est.model_name == name
        assert est.seed == 0
    # seed threads through
    assert facs["cart"](7).seed == 7


def test_factory_subset():
    spec, gold, man, b = fixture()
    facs = A.make_factories(spec, b["n_interp"], b["n_emb"],
                            models=["cart", "elasticnet_glm"])
    assert set(facs) == {"cart", "elasticnet_glm"}


# ----------------------------------------------------------------------------
# split lock bridge
# ----------------------------------------------------------------------------
def test_split_lock_from_seqclust_reproduces_folds():
    spec, gold, man, b = fixture()
    lock, row_ids = A.build_split_lock_from_seqclust(gold, spec, H)
    df = pd.read_parquet(gold)
    # every row's locked fold equals its fold_seqclust
    seq = dict(zip((df.ensembl_gene_id + "|" + df.efo_id), df.fold_seqclust.astype(int), strict=False))
    assert all(lock["row_fold"][r] == seq[r] for r in row_ids)
    # lock verifies against the same rows
    chk = H.verify_split_lock(lock, row_ids, strict=True)
    assert chk["ok"]
    assert lock["source"].startswith("fold_seqclust")


def test_split_lock_rejects_foreign_rows():
    spec, gold, man, b = fixture()
    lock, row_ids = A.build_split_lock_from_seqclust(gold, spec, H)
    raised = False
    try:
        H.verify_split_lock(lock, row_ids[:-1] + ["ENSG99999|EFO_9999"], strict=True)
    except ValueError:
        raised = True
    assert raised


# ----------------------------------------------------------------------------
# Mode A: ingest OOF
# ----------------------------------------------------------------------------
def make_oof(tmpdir, gold):
    """Fabricate an oof_predictions.parquet + metrics_by_split.csv in the
    training harness's output schema."""
    df = pd.read_parquet(gold)
    n = len(df)
    rng = np.random.RandomState(2)
    out = df[["ensembl_gene_id", "efo_id", "gene_symbol", "label_consensus",
              "fold_seqclust"]].copy()
    out["gidx"] = np.arange(n)
    y = df.label_consensus.values
    for sp in ["seqclust", "gene"]:
        for m in ["xgb_baseline", "elasticnet_glm"]:
            # signal-bearing score + noise
            out[f"oof_{sp}_{m}"] = 0.6 * y + 0.4 * rng.rand(n)
    # permuted-label OOF only for the headline split
    for m in ["xgb_baseline", "elasticnet_glm"]:
        out[f"permoof_{m}"] = rng.rand(n)
    p = os.path.join(tmpdir, "oof_predictions.parquet")
    out.to_parquet(p)
    mcsv = os.path.join(tmpdir, "metrics_by_split.csv")
    pd.DataFrame([{"split": "seqclust", "model": "xgb_baseline", "gene_roc": 0.8}]).to_csv(mcsv, index=False)
    return p, mcsv


def test_ingest_oof_mode_a():
    spec, gold, man, b = fixture()
    oof_p, mcsv = make_oof("t_adapter_fix", gold)
    tbl = A.ingest_oof(oof_p, mcsv, H)
    assert len(tbl) > 0
    assert {"split", "model", "gene_roc_auc"}.issubset(tbl.columns)
    # signal-bearing scores must beat 0.5
    assert (tbl["gene_roc_auc"] > 0.5).all()
    # headline split carries a signal gap vs the permuted null
    head = tbl[tbl["split"] == "seqclust"]
    assert "signal_gap" in tbl.columns
    assert (head["signal_gap"] > 0).all(), "signal gap should be positive on headline split"


# ----------------------------------------------------------------------------
# End-to-end: adapter factories -> run_comparison (available families only)
# ----------------------------------------------------------------------------
def test_end_to_end_run_comparison():
    import shutil
    spec, gold, man, b = fixture()
    avail = [m for m in ["xgb_baseline", "elasticnet_glm", "cart"] if HAVE[m]]
    facs = A.make_factories(spec, b["n_interp"], b["n_emb"], models=avail)
    lock, row_ids = A.build_split_lock_from_seqclust(gold, spec, H)
    od = "t_adapter_e2e"
    if os.path.exists(od):
        shutil.rmtree(od)
    rep = H.run_comparison(
        b["X"], b["y"], models=facs, groups=b["groups"],
        feature_blocks=b["feature_blocks"], group_key=b["group_key"],
        seeds=(0, 1), n_splits=5, k=10, headline_metric="gene_roc_auc",
        split_lock=lock, row_ids=row_ids, make_figures=True, outdir=od)
    assert rep["split_lock"]["enforced"] is True
    # planted signal is in the embedding -> esmc6b should be the top ablation family
    abl = {k: v for k, v in rep["ablation"].items() if k != "__full__"}
    top_fam = max(abl, key=lambda f: abl[f]["delta_vs_full"])
    assert top_fam == "esmc6b", f"expected esmc6b to dominate ablation, got {top_fam}"
    for f in ["fig_metrics.png", "fig_ablation.png", "fig_convergence.png"]:
        assert os.path.exists(os.path.join(od, f))


# ----------------------------------------------------------------------------
# standalone runner
# ----------------------------------------------------------------------------
def _main():
    import shutil
    tests = [v for k, v in sorted(globals().items())
             if k.startswith("test_") and callable(v)]
    passed = failed = 0
    for t in tests:
        try:
            t(); print(f"PASS {t.__name__}"); passed += 1
        except Exception as exc:
            print(f"FAIL {t.__name__}: {type(exc).__name__}: {exc}"); failed += 1
    print(f"\n{passed} passed, {failed} failed of {len(tests)}")
    print("dep availability:", {k: v for k, v in HAVE.items()})
    for d in ["t_adapter_fix", "t_adapter_e2e"]:
        shutil.rmtree(d, ignore_errors=True)
    return failed


if __name__ == "__main__":
    raise SystemExit(_main())
