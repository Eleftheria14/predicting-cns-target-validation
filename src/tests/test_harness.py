"""Test suite for the model-comparison harness (kernel.py).

Dual-mode: runs under pytest (`pytest test_harness.py`) OR standalone
(`python test_harness.py`) with no pytest dependency. Each test asserts one
guarantee the harness makes. A synthetic dataset with a KNOWN signal family
lets us check that the harness recovers the planted structure, not just that
it runs.
"""
import ast
import importlib.util
import os
import shutil

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
KERNEL = os.path.join(HERE, "..", "interpretation_harness", "kernel.py")


def load_kernel():
    spec = importlib.util.spec_from_file_location("cmp_kernel_under_test", KERNEL)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


H = load_kernel()


# ----------------------------------------------------------------------------
# Synthetic data with a planted signal family. Gene-level latent quality so
# that group-aggregated labels have BOTH classes (a global row-shuffle would
# collapse them - the harness must handle this).
# ----------------------------------------------------------------------------
def make_data(seed=7, n_genes=80, per_gene=12, flip=0.15):
    rng = np.random.RandomState(seed)
    grp = np.repeat(np.arange(n_genes), per_gene)
    n = len(grp)
    gene_q = rng.randn(n_genes)
    A = np.zeros((n, 5)); B = np.zeros((n, 4)); C = rng.randn(n, 6)
    for i, g in enumerate(grp):
        A[i, 0] = gene_q[g] * 1.3 + rng.randn() * 0.5   # famA carries signal
        A[i, 1] = gene_q[g] * 0.9 + rng.randn() * 0.6
        A[i, 2:] = rng.randn(3)
        B[i, 0] = gene_q[g] * 0.4 + rng.randn() * 0.8   # famB weak
        B[i, 1:] = rng.randn(3)
    gene_label = (gene_q > 0.3).astype(int)
    y = np.array([gene_label[g] if rng.rand() > flip else 1 - gene_label[g]
                  for g in grp])
    cols = [f"A{i}" for i in range(5)] + [f"B{i}" for i in range(4)] + [f"C{i}" for i in range(6)]
    X = pd.DataFrame(np.hstack([A, B, C]), columns=cols)
    fams = {"famA": [c for c in cols if c[0] == "A"],
            "famB": [c for c in cols if c[0] == "B"],
            "famC": [c for c in cols if c[0] == "C"]}
    return X, y, grp, fams


def default_models():
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.tree import DecisionTreeClassifier
    return {
        "GLM": lambda s: LogisticRegression(max_iter=500, random_state=s),
        "Tree": lambda s: DecisionTreeClassifier(max_depth=3, random_state=s),
        "GBM": lambda s: GradientBoostingClassifier(max_depth=3, n_estimators=60, random_state=s),
    }


def preprocess(seed):
    from sklearn.preprocessing import StandardScaler
    return StandardScaler()


def _run(outdir, **kw):
    if os.path.exists(outdir):
        shutil.rmtree(outdir)
    X, y, grp, fams = make_data()
    base = dict(models=default_models(), groups=grp, feature_blocks=fams,
                seeds=(0, 1, 2, 3, 4), n_splits=5, preprocess=preprocess,
                group_key=grp, k=20, headline_metric="gene_roc_auc", outdir=outdir)
    base.update(kw)
    return H.run_comparison(X, y, **base), (X, y, grp, fams)


# ----------------------------------------------------------------------------
# Folds / leakage
# ----------------------------------------------------------------------------
def test_folds_are_group_disjoint():
    """No group may appear in more than one fold (the leakage guard)."""
    _, _, grp, _ = make_data()
    folds = H.make_folds(grp, n_splits=5, seed=0)
    by_group = {}
    for g, f in zip(grp, folds, strict=False):
        by_group.setdefault(g, set()).add(f)
    assert all(len(fs) == 1 for fs in by_group.values()), "a group spans >1 fold"


def test_folds_deterministic():
    _, _, grp, _ = make_data()
    a = H.make_folds(grp, 5, seed=0)
    b = H.make_folds(grp, 5, seed=0)
    assert np.array_equal(a, b)
    c = H.make_folds(grp, 5, seed=1)
    assert not np.array_equal(a, c), "different seed should reshuffle"


# ----------------------------------------------------------------------------
# Metrics
# ----------------------------------------------------------------------------
def test_precision_at_k():
    y = np.array([1, 1, 0, 0, 0])
    s = np.array([0.9, 0.8, 0.7, 0.2, 0.1])   # top-2 are both positive
    assert H.precision_at_k(y, s, k=2) == 1.0
    assert H.precision_at_k(y, s, k=4) == 0.5


def test_both_scoring_paths():
    """predict_proba and decision_function must both yield finite scores."""
    from sklearn.linear_model import LogisticRegression, SGDClassifier
    X, y, grp, _ = make_data()
    folds = H.make_folds(grp, 5, seed=0)
    oof_pp = H.cross_val_oof(lambda s: LogisticRegression(max_iter=300, random_state=s),
                             X, y, folds)
    oof_df = H.cross_val_oof(lambda s: SGDClassifier(loss="hinge", random_state=s),
                             X, y, folds)   # decision_function only
    assert np.isfinite(oof_pp).all() and np.isfinite(oof_df).all()


# ----------------------------------------------------------------------------
# Full run: outputs, determinism, signal recovery
# ----------------------------------------------------------------------------
def test_run_writes_all_outputs():
    rep, _ = _run("t_out_1")
    for f in ["metrics.csv", "report.json", "verdict.md",
              "fig_metrics.png", "fig_permutation.png",
              "fig_ablation.png", "fig_convergence.png"]:
        assert os.path.exists(os.path.join("t_out_1", f)), f"missing {f}"


def test_report_json_deterministic():
    """Two runs on identical inputs -> byte-identical report.json."""
    import hashlib
    h = []
    for od in ["t_det_a", "t_det_b"]:
        _run(od, make_figures=False)
        h.append(hashlib.md5(open(os.path.join(od, "report.json"), "rb").read()).hexdigest())
    assert h[0] == h[1], "report.json is not deterministic"


def test_signal_family_recovered_by_ablation():
    """Ablation must rank the planted family (famA) far above the noise ones."""
    rep, _ = _run("t_abl", make_figures=False)
    abl = rep["ablation"]
    dA = abl["famA"]["delta_vs_full"]
    dB = abl["famB"]["delta_vs_full"]
    dC = abl["famC"]["delta_vs_full"]
    assert dA > 0.05, f"planted family famA delta too small: {dA}"
    assert dA > max(dB, dC) + 0.05, "famA not clearly above noise families"


def test_permutation_gap_positive():
    """Real models must beat their group-level permutation null."""
    rep, _ = _run("t_perm", make_figures=False)
    for m, d in rep["models"].items():
        assert d["signal_gap"] > 0.03, f"{m} signal gap not positive: {d['signal_gap']}"


def test_permutation_null_is_group_level_and_defined():
    """The grouped permutation must keep the gene-level metric DEFINED (not
    NaN) - the bug that a global row-shuffle would reintroduce."""
    rep, _ = _run("t_perm2", make_figures=False)
    for m, d in rep["models"].items():
        assert d["permutation"]["gene_roc_auc"] == d["permutation"]["gene_roc_auc"], \
            f"{m} permutation null is NaN (group-level shuffle broke)"


# ----------------------------------------------------------------------------
# Rashomon set + convergence
# ----------------------------------------------------------------------------
def test_rashomon_set_membership():
    rep, _ = _run("t_rset", make_figures=False)
    rs = rep["rashomon_set"]
    assert rs["best_model"] in rep["models"]
    assert set(rs["members"]).issubset(set(rep["models"]))
    # best model is always in its own set
    assert rs["best_model"] in rs["members"]


def test_convergence_recovers_planted_family():
    """Every Rashomon-set member should put most importance on famA, so the
    consensus and agreement metrics are high."""
    rep, _ = _run("t_conv", make_figures=False)
    conv = rep["convergence"]
    ci = conv["consensus_importance"]
    assert ci["famA"] > 0.5, f"consensus importance on famA too low: {ci}"
    assert ci["famA"] > ci["famB"] and ci["famA"] > ci["famC"]
    assert conv["mean_tau"] >= 0.0   # defined and in range
    assert -1.0 <= conv["mean_tau"] <= 1.0


def test_tie_decision_band_logic():
    """A model's difference with itself is 0, which is below any band -> tied.
    Exercised indirectly: tie flags are bools and reference real bands."""
    rep, _ = _run("t_tie", make_figures=False)
    for t in rep["tie_decisions"]:
        assert isinstance(t["tied"], bool)
        assert t["diff"] >= 0.0


# ----------------------------------------------------------------------------
# Split lock
# ----------------------------------------------------------------------------
def test_split_lock_fingerprint_stable_and_order_invariant():
    _, _, grp, _ = make_data()
    row_ids = [f"g{g}|{i}" for i, g in enumerate(grp)]
    l1 = H.make_split_lock(row_ids, grp, 5, 0)
    l2 = H.make_split_lock(row_ids, grp, 5, 0)
    assert l1["fingerprint"] == l2["fingerprint"], "fingerprint not stable"
    # reordered rows still verify
    perm = np.random.RandomState(3).permutation(len(row_ids))
    chk = H.verify_split_lock(l1, [row_ids[i] for i in perm], strict=True)
    assert chk["ok"]


def test_split_lock_rejects_mismatch():
    _, _, grp, _ = make_data()
    row_ids = [f"g{g}|{i}" for i, g in enumerate(grp)]
    lock = H.make_split_lock(row_ids, grp, 5, 0)
    bad = row_ids[:-1] + ["fake|row"]
    raised = False
    try:
        H.verify_split_lock(lock, bad, strict=True)
    except ValueError:
        raised = True
    assert raised, "verify_split_lock did not reject a mismatched row set"


def test_split_lock_reproduces_canonical_split():
    """A locked run must equal the unlocked seed-0 run (lock enforces, does not
    change results)."""
    X, y, grp, fams = make_data()
    row_ids = [f"g{g}|{i}" for i, g in enumerate(grp)]
    lock = H.make_split_lock(row_ids, grp, 5, 0)
    common = dict(models=default_models(), groups=grp, feature_blocks=fams,
                  seeds=(0, 1, 2), n_splits=5, preprocess=preprocess,
                  group_key=grp, k=20, headline_metric="gene_roc_auc",
                  make_figures=False)
    locked = H.run_comparison(X, y, split_lock=lock, row_ids=row_ids,
                              outdir="t_lock_a", **common)
    free = H.run_comparison(X, y, outdir="t_lock_b", **common)
    for m in locked["models"]:
        a = locked["models"][m]["point"]["gene_roc_auc"]
        b = free["models"][m]["point"]["gene_roc_auc"]
        assert a == b, f"{m}: locked {a} != unlocked {b}"
    assert locked["split_lock"]["enforced"] is True


def test_run_refuses_mismatched_lock():
    X, y, grp, fams = make_data()
    row_ids = [f"g{g}|{i}" for i, g in enumerate(grp)]
    lock = H.make_split_lock(row_ids, grp, 5, 0)
    raised = False
    try:
        H.run_comparison(X.iloc[:-5], y[:-5], models=default_models(),
                         groups=grp[:-5], feature_blocks=fams, seeds=(0, 1),
                         n_splits=5, preprocess=preprocess, group_key=grp[:-5],
                         split_lock=lock, row_ids=row_ids[:-5],
                         make_figures=False, outdir="t_lock_bad")
    except ValueError:
        raised = True
    assert raised, "run_comparison ran on rows that don't match the lock"


# ----------------------------------------------------------------------------
# Sidecar gate: kernel.py must stay loadable as a skill sidecar
# ----------------------------------------------------------------------------
def test_kernel_is_sidecar_gate_clean():
    tree = ast.parse(open(KERNEL).read())
    problems = []
    for node in tree.body:
        if isinstance(node, ast.FunctionDef):
            if node.name.startswith("_"):
                problems.append(f"underscore def {node.name}")
            if node.decorator_list:
                problems.append(f"decorated def {node.name}")
            for d in list(node.args.defaults) + list(node.args.kw_defaults):
                if d is None:
                    continue
                if isinstance(d, ast.Constant):
                    continue
                if isinstance(d, (ast.Tuple, ast.List)) and all(isinstance(e, ast.Constant) for e in d.elts):
                    continue
                problems.append(f"non-literal default in {node.name}")
        elif isinstance(node, ast.Assign):
            if not isinstance(node.value, (ast.Constant, ast.Tuple, ast.List)):
                problems.append("non-literal top-level assign")
        elif isinstance(node, (ast.If, ast.For, ast.While, ast.ClassDef, ast.With)):
            problems.append(f"disallowed top-level {type(node).__name__}")
        elif isinstance(node, ast.Expr) and node is not tree.body[0]:
            problems.append("stray top-level expression")
    assert not problems, f"sidecar gate violations: {problems}"


# ----------------------------------------------------------------------------
# Standalone runner (no pytest dependency)
# ----------------------------------------------------------------------------
def test_full_report_md():
    """report.md is written with all 7 DS-ordered sections; the split ladder
    section appears when splits_table is supplied."""
    splits = [{"split": sp, "model": m, "gene_roc_auc": v}
              for m in default_models()
              for sp, v in [("random", 0.90), ("gene", 0.86),
                            ("disease", 0.83), ("seqclust", 0.85)]]
    od = "t_report_out"
    _run(od, make_figures=False,
         report_meta={"dataset_name": "unit-test", "n_genes": 80},
         splits_table=splits)
    rp = os.path.join(od, "report.md")
    assert os.path.exists(rp)
    txt = open(rp).read()
    for h in ["## 0. Run provenance", "## 1. Can I trust",
              "## 2. Headline result", "## 3. Is the signal real",
              "## 4. Generalization across split", "## 5. What drives",
              "## 6. Do independent model families", "## 7. Caveats"]:
        assert h in txt, f"missing report section: {h}"
    assert "Mean optimism (random" in txt        # split ladder rendered
    assert "unit of comparison" in txt           # headline framing present
    shutil.rmtree(od, ignore_errors=True)


def test_report_md_omits_split_section_without_table():
    """Without splits_table, Section 4 is skipped but the rest render."""
    od = "t_report_nosplit"
    _run(od, make_figures=False)
    txt = open(os.path.join(od, "report.md")).read()
    assert "## 4. Generalization across split" not in txt
    assert "## 7. Caveats" in txt
    shutil.rmtree(od, ignore_errors=True)


def _main():
    tests = [v for k, v in sorted(globals().items())
             if k.startswith("test_") and callable(v)]
    passed = failed = 0
    for t in tests:
        try:
            t()
            print(f"PASS {t.__name__}")
            passed += 1
        except Exception as exc:
            print(f"FAIL {t.__name__}: {type(exc).__name__}: {exc}")
            failed += 1
    print(f"\n{passed} passed, {failed} failed of {len(tests)}")
    # cleanup scratch dirs
    for d in os.listdir("."):
        if d.startswith("t_out_") or d.startswith("t_det_") or d.startswith("t_") and d.endswith(("_a", "_b", "bad")):
            shutil.rmtree(d, ignore_errors=True)
    return failed


if __name__ == "__main__":
    raise SystemExit(_main())
