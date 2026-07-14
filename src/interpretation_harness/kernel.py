"""Harness for leakage-safe, over-interpretation-resistant model comparison.

Reusable helpers for the model-comparison protocol. Plug in your own models
(as factories) and feature matrix; the harness runs a frozen-harness comparison
with plural metrics, a per-model permutation control, a seed-based Rashomon
band, remove-and-retrain ablation over feature families, and a model-agnostic
cross-model convergence assessment, then writes ONE canonical output set
(metrics CSV + JSON report + markdown verdict + one figure per stage) that is
deterministic across runs on identical inputs.

Core entry point: run_comparison(...). See SKILL.md for the full guide.

Model spec contract: each entry in `models` is a factory callable
`factory(seed) -> estimator`, where estimator exposes sklearn-style
`fit(X, y)` and `predict_proba(X)` (or `decision_function`). Passing a
factory (not a fitted model) is what lets the harness retrain cleanly across
folds and seeds — essential for an honest Rashomon band, honest ablation, and
fresh fits for the convergence layer.

Two importance questions, deliberately kept distinct (see SKILL.md):
  * ablation (remove-and-RETRAIN a family)  -> the family's MARGINAL VALUE to
    the pipeline. Grounded in ROAR (Hooker et al. 2019).
  * convergence (permutation importance on ALREADY-FIT models) -> what each
    model RELIES ON. Grounded in model-class reliance (Fisher, Rudin &
    Dominici 2018) and permutation importance (Breiman 2001). This is the
    common currency across heterogeneous model types (needs only
    predict_proba), so it is what makes cross-family agreement measurable.

Convergence voting is restricted to the epsilon-Rashomon set: only models whose
headline metric is within `epsilon` of the best model "vote" on the biology. A
model that barely works is not a dissenting opinion, it is a worse model
(Semenova, Rudin & Parr 2019). epsilon defaults to the empirical seed band of
the best model.
"""

RANDOM_SEED = 0
DEFAULT_SEEDS = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9)
FLOAT_ROUND = 4
TIE_BAND_NOTE = (
    "Two models whose metric difference is smaller than the larger of their "
    "Rashomon bands are TIED: interpret their explanations for agreement, do "
    "not narrate the score difference."
)


def make_folds(groups, n_splits=5, seed=0):
    """Leakage-safe fold assignment: whole groups (e.g. homolog clusters or
    genes) are held out together. Returns an int array of fold ids aligned to
    `groups`. Deterministic given (groups, n_splits, seed)."""
    import numpy as np
    groups = np.asarray(groups)
    uniq = np.array(sorted(set(groups.tolist())))
    rng = np.random.RandomState(seed)
    perm = rng.permutation(len(uniq))
    fold_of_group = {g: int(perm[i] % n_splits) for i, g in enumerate(uniq)}
    return np.array([fold_of_group[g] for g in groups], dtype=int)


def p_split_fingerprint(row_ids, folds, n_splits, seed):
    """Content hash binding ROW IDENTITY to fold assignment. Any harness that
    reconstructs the same split over the same rows gets the same fingerprint;
    evaluating on a different row set (reordered, subset, or augmented) changes
    it. This is what makes the lock enforceable rather than advisory."""
    import hashlib
    pairs = sorted((str(r), int(f)) for r, f in zip(row_ids, folds, strict=False))
    blob = f"n_splits={n_splits};seed={seed};" + ";".join(f"{r}:{f}" for r, f in pairs)
    return "sha256:" + hashlib.sha256(blob.encode()).hexdigest()


def make_split_lock(row_ids, groups, n_splits=5, seed=0):
    """Build the canonical shared split. row_ids is a stable per-row key (e.g.
    ensembl_gene_id + '|' + disease) — NOT the positional index, so the lock
    survives row reordering. groups is the leakage-safe grouping held out whole
    (e.g. homolog cluster). Returns a lock dict: fingerprint + the row->fold and
    row->group maps + params. Serialize with save_split_lock and share it as
    the one artifact BOTH the training harness and this comparison harness
    load."""
    import numpy as np
    row_ids = [str(r) for r in row_ids]
    if len(set(row_ids)) != len(row_ids):
        raise ValueError("row_ids must be unique (they key the fold lock)")
    folds = make_folds(groups, n_splits, seed)
    fp = p_split_fingerprint(row_ids, folds, n_splits, seed)
    return {
        "schema": "split_lock/v1", "fingerprint": fp,
        "n_splits": int(n_splits), "seed": int(seed), "n_rows": len(row_ids),
        "row_fold": dict(zip(row_ids, [int(f) for f in folds], strict=False)),
        "row_group": dict(zip(row_ids, [str(g) for g in np.asarray(groups)], strict=False)),
    }


def save_split_lock(lock, path):
    """Write the lock as canonical JSON (sorted keys) so the file itself is
    byte-deterministic across runs."""
    import json
    with open(path, "w") as fh:
        json.dump(lock, fh, indent=2, sort_keys=True)
    return path


def load_split_lock(path):
    import json
    with open(path) as fh:
        return json.load(fh)


def verify_split_lock(lock, row_ids, strict=True):
    """Enforce the lock against the rows you are about to use. Checks that the
    row_ids EXACTLY match the lock's row set and recomputes the fingerprint. On
    any mismatch (missing rows, extra rows, changed grouping) it raises when
    strict — this is the hard stop that prevents a harness from silently
    evaluating or tuning on a different split. Returns the fold vector aligned
    to the ORDER of the row_ids you passed."""
    ids = [str(r) for r in row_ids]
    lock_ids = set(lock["row_fold"])
    have = set(ids)
    missing = lock_ids - have
    extra = have - lock_ids
    problems = []
    if len(ids) != len(have):
        problems.append("row_ids not unique")
    if missing:
        problems.append(f"{len(missing)} locked rows absent (e.g. {sorted(missing)[:3]})")
    if extra:
        problems.append(f"{len(extra)} rows not in lock (e.g. {sorted(extra)[:3]})")
    folds = None
    if not missing and not extra and len(ids) == len(have):
        folds = [lock["row_fold"][r] for r in ids]
        groups = [lock["row_group"][r] for r in ids]
        fp = p_split_fingerprint(ids, folds, lock["n_splits"], lock["seed"])
        if fp != lock["fingerprint"]:
            problems.append("fingerprint mismatch (fold/group content differs)")
    if problems and strict:
        raise ValueError("split-lock verification FAILED: " + "; ".join(problems))
    return {"ok": not problems, "problems": problems,
            "folds": folds, "fingerprint": lock["fingerprint"]}


def p_scores_from_estimator(est, X):
    """Positive-class score from predict_proba or decision_function."""
    import numpy as np
    if hasattr(est, "predict_proba"):
        p = est.predict_proba(X)
        return np.asarray(p)[:, 1]
    if hasattr(est, "decision_function"):
        return np.asarray(est.decision_function(X))
    return np.asarray(est.predict(X), dtype=float)


def precision_at_k(y_true, y_score, k=20):
    """Fraction of the top-k highest-scored items that are positive. The
    decision-facing metric for a top-of-ranking triage tool; does NOT track
    ROC-AUC, so always report it alongside."""
    import numpy as np
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    k = min(int(k), len(y_score))
    if k <= 0:
        return float("nan")
    top = np.argsort(-y_score)[:k]
    return float(y_true[top].mean())


def p_metric_block(y_true, y_score, k=20):
    import numpy as np
    from sklearn.metrics import average_precision_score, roc_auc_score
    y_true = np.asarray(y_true)
    out = {}
    if len(set(y_true.tolist())) < 2:
        out["roc_auc"] = float("nan")
        out["pr_auc"] = float("nan")
    else:
        out["roc_auc"] = float(roc_auc_score(y_true, y_score))
        out["pr_auc"] = float(average_precision_score(y_true, y_score))
    out[f"p_at_{k}"] = precision_at_k(y_true, y_score, k)
    return out


def cross_val_oof(factory, X, y, folds, preprocess=None, seed=0):
    """Out-of-fold positive-class scores. `preprocess`, if given, is a callable
    `preprocess(seed) -> transformer` with fit/transform, fit on the TRAIN fold
    only (leakage guard for e.g. in-fold PCA/scaling). Returns an OOF score
    array aligned to X's rows."""
    import numpy as np
    Xv = X.values if hasattr(X, "values") else np.asarray(X)
    y = np.asarray(y)
    folds = np.asarray(folds)
    oof = np.full(len(y), np.nan, dtype=float)
    for f in sorted(set(folds.tolist())):
        tr, te = folds != f, folds == f
        Xtr, Xte = Xv[tr], Xv[te]
        if preprocess is not None:
            pp = preprocess(seed)
            Xtr = pp.fit_transform(Xtr)
            Xte = pp.transform(Xte)
        est = factory(seed)
        est.fit(Xtr, y[tr])
        oof[te] = p_scores_from_estimator(est, Xte)
    return oof


def p_aggregate(y, oof, group_key=None, k=20):
    """Pair-level metrics, plus group-level (e.g. gene-level) metrics when a
    group_key is supplied (OOF aggregated by group mean, matching a gene-level
    headline)."""
    import numpy as np
    import pandas as pd
    res = {}
    for name, val in p_metric_block(y, oof, k).items():
        res[f"pair_{name}"] = val
    if group_key is not None:
        df = pd.DataFrame({"g": np.asarray(group_key), "y": np.asarray(y), "s": oof})
        agg = df.groupby("g").agg(y=("y", "max"), s=("s", "mean")).reset_index()
        for name, val in p_metric_block(agg["y"].values, agg["s"].values, k).items():
            res[f"gene_{name}"] = val
    return res


def evaluate_model(factory, X, y, folds, preprocess=None, seed=0,
                   group_key=None, k=20):
    """One model, one seed: OOF then aggregated metrics."""
    oof = cross_val_oof(factory, X, y, folds, preprocess, seed)
    return p_aggregate(y, oof, group_key, k), oof


def permutation_control(factory, X, y, folds, preprocess=None, seed=0,
                        group_key=None, k=20):
    """Label-shuffle control: shuffle labels, rerun OOF, score against the
    SHUFFLED labels. The real-minus-null gap is the honest signal; an elevated
    null means the recipe fits chance structure.

    When group_key is given, the shuffle is done AT THE GROUP LEVEL: each
    group's label-vector is reassigned to another group, preserving every
    group's internal label pattern (and thus the group-level label
    distribution). A global row-shuffle would make max-aggregated group labels
    collapse to a single class and the group-level metric undefined — so a
    grouped metric requires a grouped permutation."""
    import numpy as np
    rng = np.random.RandomState(seed)
    y = np.asarray(y)
    if group_key is not None:
        gk = np.asarray(group_key)
        uniq = np.array(sorted(set(gk.tolist())))
        perm = rng.permutation(len(uniq))
        # map each group to a donor group; a row takes the donor group's
        # label at the same within-group position (cyclic if sizes differ)
        idx_by_group = {g: np.where(gk == g)[0] for g in uniq}
        y_perm = y.copy()
        for gi, g in enumerate(uniq):
            donor = uniq[perm[gi]]
            tgt, src = idx_by_group[g], idx_by_group[donor]
            y_perm[tgt] = y[src[np.arange(len(tgt)) % len(src)]]
    else:
        y_perm = y.copy()
        rng.shuffle(y_perm)
    oof = cross_val_oof(factory, X, y_perm, folds, preprocess, seed)
    return p_aggregate(y_perm, oof, group_key, k)


def rashomon_band(factory, X, y, groups, seeds=None, n_splits=5,
                  preprocess=None, group_key=None, k=20, metric="gene_roc_auc"):
    """Retrain the SAME model family across seeds (each seed reshuffles folds
    and re-inits the model) and report the spread of a chosen metric. The band
    (max-min, and std) is the yardstick for whether a between-model difference
    is real or noise."""
    import numpy as np
    if seeds is None:
        seeds = DEFAULT_SEEDS
    vals = []
    for s in seeds:
        folds = make_folds(groups, n_splits, seed=s)
        m, _ = evaluate_model(factory, X, y, folds, preprocess, s, group_key, k)
        if metric in m and not np.isnan(m[metric]):
            vals.append(m[metric])
    vals = np.asarray(vals, dtype=float)
    if len(vals) == 0:
        return {"metric": metric, "n": 0}
    return {
        "metric": metric, "n": int(len(vals)),
        "mean": float(vals.mean()), "std": float(vals.std(ddof=1) if len(vals) > 1 else 0.0),
        "min": float(vals.min()), "max": float(vals.max()),
        "band": float(vals.max() - vals.min()),
    }


def ablation(factory, X, y, groups, feature_blocks, seeds=(0, 1, 2), n_splits=5,
             preprocess=None, group_key=None, k=20, metric="gene_roc_auc"):
    """Remove-and-RETRAIN ablation (ROAR discipline, Hooker et al. 2019).
    For each named block, DROP those columns and refit the model from scratch
    across `seeds`; the delta vs the full model is the block's contribution.
    Naive drop-and-rescore (without retraining) is biased and is deliberately
    not offered. Requires X to be a DataFrame so blocks name columns.

    feature_blocks: {block_name: [column, ...]}. Returns per-block mean delta
    with a seed spread so a contribution can be judged against its own noise."""
    import numpy as np
    if not hasattr(X, "columns"):
        raise ValueError("ablation requires X as a pandas DataFrame (blocks name columns)")

    def _mean_metric(Xin):
        vals = []
        for s in seeds:
            folds = make_folds(groups, n_splits, seed=s)
            m, _ = evaluate_model(factory, Xin, y, folds, preprocess, s, group_key, k)
            if metric in m and not np.isnan(m[metric]):
                vals.append(m[metric])
        vals = np.asarray(vals, dtype=float)
        return vals

    full = _mean_metric(X)
    rows = {"__full__": {"metric": metric, "n": int(len(full)),
                         "mean": float(full.mean()) if len(full) else float("nan"),
                         "band": float(full.max() - full.min()) if len(full) else float("nan")}}
    for name, cols in feature_blocks.items():
        keep = [c for c in X.columns if c not in set(cols)]
        red = _mean_metric(X[keep])
        delta = (full.mean() - red.mean()) if (len(full) and len(red)) else float("nan")
        rows[name] = {
            "metric": metric, "n_dropped": len(cols),
            "reduced_mean": float(red.mean()) if len(red) else float("nan"),
            "delta_vs_full": float(delta),
            "reduced_band": float(red.max() - red.min()) if len(red) else float("nan"),
        }
    return rows


def family_permutation_importance(est, X, y, feature_blocks, k=20,
                                  metric="roc_auc", n_repeats=5, seed=0):
    """Model-agnostic importance in a COMMON currency across heterogeneous
    models: for each feature family, permute all its columns together and
    measure the drop in `metric` on the already-fit estimator. Needs only
    predict_proba, so a GBM, an EBM, a GLM, a rule set, and a prototype model
    all yield comparable family-importance vectors. This is what makes
    cross-model convergence measurable. (Breiman 2001 permutation importance;
    grouped by family so collinear columns do not deflate each other.)

    Returns {family: mean_importance} normalized to sum to 1 over families."""
    import numpy as np
    rng = np.random.RandomState(seed)
    Xv = X.values.copy() if hasattr(X, "values") else np.asarray(X).copy()
    cols = list(X.columns) if hasattr(X, "columns") else list(range(Xv.shape[1]))
    col_idx = {c: i for i, c in enumerate(cols)}

    def score(mat):
        s = p_scores_from_estimator(est, mat)
        m = p_metric_block(y, s, k)
        return m.get(metric, m.get("roc_auc"))

    base = score(Xv)
    raw = {}
    for fam, fcols in feature_blocks.items():
        idx = [col_idx[c] for c in fcols if c in col_idx]
        if not idx:
            raw[fam] = 0.0
            continue
        drops = []
        for _ in range(n_repeats):
            Xp = Xv.copy()
            perm = rng.permutation(Xp.shape[0])
            for j in idx:
                Xp[:, j] = Xp[perm, j]
            drops.append(base - score(Xp))
        raw[fam] = float(max(np.mean(drops), 0.0))
    total = sum(raw.values()) or 1.0
    return {fam: v / total for fam, v in raw.items()}


def rashomon_set(per_model_point, per_model_band, headline_metric, epsilon=None):
    """epsilon-Rashomon set: the models within `epsilon` of the best headline
    metric. These are the models allowed to "vote" on convergence; a model
    outside the set is a worse model, not a dissent (Fisher/Rudin/Dominici
    2018; Semenova/Rudin/Parr 2019). epsilon defaults to the best model's
    empirical seed band."""
    import numpy as np
    vals = {m: per_model_point[m].get(headline_metric, float("nan"))
            for m in per_model_point}
    best_model = max(vals, key=lambda m: (vals[m] if not np.isnan(vals[m]) else -1))
    best = vals[best_model]
    if epsilon is None:
        epsilon = float(per_model_band.get(best_model, {}).get("band", 0.0)) or 0.02
    members = sorted([m for m, v in vals.items()
                      if not np.isnan(v) and (best - v) <= epsilon])
    return {"best_model": best_model, "best_value": float(best),
            "epsilon": float(epsilon), "members": members}


def convergence(models, X, y, feature_blocks, folds, rset_members,
                preprocess=None, seed=0, k=20, metric="roc_auc", top_k=3):
    """Cross-model convergence on a COMMON currency (grouped permutation
    importance). Fits each Rashomon-set member on one held-in split, projects
    its importance onto feature families, then computes agreement metrics from
    the disagreement-problem literature (Krishna et al. 2022): pairwise rank
    agreement (Kendall tau) over families, and top-k family overlap (Jaccard).

    Returns the family x model importance matrix (the headline object), the
    pairwise tau matrix, and mean agreement."""
    from itertools import combinations

    import numpy as np
    Xv = X.values if hasattr(X, "values") else np.asarray(X)
    fams = sorted(feature_blocks)
    # fit each member once on fold!=0, importance measured on fold==0
    tr, te = folds != 0, folds == 0
    Xtr = X.iloc[tr] if hasattr(X, "iloc") else X[tr]
    Xte = X.iloc[te] if hasattr(X, "iloc") else X[te]
    ytr, yte = np.asarray(y)[tr], np.asarray(y)[te]
    if preprocess is not None:
        pp = preprocess(seed)
        import pandas as pd
        Xtr = pd.DataFrame(pp.fit_transform(Xtr), columns=X.columns, index=Xtr.index)
        Xte = pd.DataFrame(pp.transform(Xte), columns=X.columns, index=Xte.index)
    imp = {}
    for name in rset_members:
        est = models[name](seed)
        est.fit(Xtr.values if hasattr(Xtr, "values") else Xtr, ytr)
        fi = family_permutation_importance(est, Xte, yte, feature_blocks, k,
                                           metric, seed=seed)
        imp[name] = [fi.get(f, 0.0) for f in fams]

    def kendall_tau(a, b):
        n = len(a); c = d = 0
        for i in range(n):
            for j in range(i + 1, n):
                s = (a[i] - a[j]) * (b[i] - b[j])
                if s > 0: c += 1
                elif s < 0: d += 1
        return (c - d) / (0.5 * n * (n - 1)) if n > 1 else 1.0

    def topk(vec):
        order = sorted(range(len(fams)), key=lambda i: -vec[i])[:top_k]
        return set(order)

    tau, jac = {}, []
    for a, b in combinations(sorted(imp), 2):
        t = kendall_tau(imp[a], imp[b])
        tau[f"{a}|{b}"] = float(t)
        ja = topk(imp[a]); jb = topk(imp[b])
        jac.append(len(ja & jb) / len(ja | jb) if (ja | jb) else 1.0)
    consensus = [float(np.mean([imp[m][i] for m in imp])) for i in range(len(fams))]
    return {
        "families": fams, "members": sorted(imp),
        "importance_matrix": {m: imp[m] for m in imp},
        "consensus_importance": dict(zip(fams, consensus, strict=False)),
        "pairwise_tau": tau,
        "mean_tau": float(np.mean(list(tau.values()))) if tau else 1.0,
        "mean_topk_jaccard": float(np.mean(jac)) if jac else 1.0,
        "top_k": top_k,
    }


def p_round(obj, nd=None):
    if nd is None:
        nd = FLOAT_ROUND
    import numpy as np
    if isinstance(obj, dict):
        return {kk: p_round(vv, nd) for kk, vv in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [p_round(vv, nd) for vv in obj]
    if isinstance(obj, (float, np.floating)):
        return round(float(obj), nd)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    return obj


def emit_figures(report, outdir):
    """One figure per protocol stage, written to outdir/fig_*.png. Plain
    matplotlib (no seaborn dependency); deterministic. For a manuscript-grade
    composite, load the figure-style / figure-composer skills and rebuild from
    report.json — these are the standard-run diagnostics, not the paper figure.
    Returns the list of written paths."""
    import os

    import matplotlib
    import numpy as np
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    hm = report["headline_metric"]
    pm = report["models"]
    names = sorted(pm)
    written = []

    # Stage 2+4: metrics dot plot with Rashomon min-max whiskers + tie band
    fig, ax = plt.subplots(figsize=(7, 0.6 * len(names) + 1.5))
    ys = np.arange(len(names))
    pts = [pm[n]["point"].get(hm, np.nan) for n in names]
    lo = [pm[n]["rashomon"].get("min", np.nan) for n in names]
    hi = [pm[n]["rashomon"].get("max", np.nan) for n in names]
    best = np.nanmax(pts)
    best_band = pm[names[int(np.nanargmax(pts))]]["rashomon"].get("band", 0.0)
    ax.axvspan(best - best_band, best, color="0.85", label="best-model Rashomon band")
    for i, n in enumerate(names):
        ax.plot([lo[i], hi[i]], [i, i], color="0.4", lw=1.5, zorder=2)
        ax.plot(pts[i], i, "o", color="#2b6cb0", zorder=3)
    ax.set_yticks(ys); ax.set_yticklabels(names)
    ax.set_xlabel(hm); ax.set_title("Performance with Rashomon bands")
    ax.legend(loc="lower left", fontsize=8)
    fig.tight_layout(); p = os.path.join(outdir, "fig_metrics.png")
    fig.savefig(p, dpi=150); plt.close(fig); written.append(p)

    # Stage 3: signal gap (real - null)
    fig, ax = plt.subplots(figsize=(7, 0.6 * len(names) + 1.5))
    gaps = [pm[n]["signal_gap"] for n in names]
    ax.barh(ys, gaps, color="#38a169")
    ax.set_yticks(ys); ax.set_yticklabels(names)
    ax.set_xlabel(f"{hm}: real − permuted"); ax.set_title("Signal gap (permutation control)")
    ax.axvline(0, color="0.3", lw=0.8)
    fig.tight_layout(); p = os.path.join(outdir, "fig_permutation.png")
    fig.savefig(p, dpi=150); plt.close(fig); written.append(p)

    # Stage 5: ablation tornado
    if report.get("ablation"):
        abl = {k2: v for k2, v in report["ablation"].items() if k2 != "__full__"}
        if abl:
            fams = sorted(abl, key=lambda f: abl[f]["delta_vs_full"])
            deltas = [abl[f]["delta_vs_full"] for f in fams]
            bands = [abl[f].get("reduced_band", 0.0) for f in fams]
            fig, ax = plt.subplots(figsize=(7, 0.55 * len(fams) + 1.5))
            ax.barh(np.arange(len(fams)), deltas, xerr=bands, color="#d69e2e",
                    error_kw={"ecolor": "0.4", "elinewidth": 1})
            ax.set_yticks(np.arange(len(fams))); ax.set_yticklabels(fams)
            ax.set_xlabel(f"Δ {hm} when family removed and retrained")
            ax.set_title("Ablation (remove-and-retrain) by feature family")
            ax.axvline(0, color="0.3", lw=0.8)
            fig.tight_layout(); p = os.path.join(outdir, "fig_ablation.png")
            fig.savefig(p, dpi=150); plt.close(fig); written.append(p)

    # Stage 6: convergence heatmaps
    conv = report.get("convergence")
    if conv and conv.get("importance_matrix"):
        fams = conv["families"]; mem = conv["members"]
        M = np.array([conv["importance_matrix"][m] for m in mem]).T  # fam x model
        fig, ax = plt.subplots(figsize=(0.8 * len(mem) + 2.5, 0.5 * len(fams) + 2))
        im = ax.imshow(M, aspect="auto", cmap="viridis")
        ax.set_xticks(np.arange(len(mem))); ax.set_xticklabels(mem, rotation=45, ha="right")
        ax.set_yticks(np.arange(len(fams))); ax.set_yticklabels(fams)
        ax.set_title("Family × model importance (Rashomon-set members)")
        fig.colorbar(im, ax=ax, fraction=0.046, label="normalized importance")
        fig.tight_layout(); p = os.path.join(outdir, "fig_convergence.png")
        fig.savefig(p, dpi=150); plt.close(fig); written.append(p)

    return written


def run_comparison(X, y, models, groups, feature_blocks=None,
                   seeds=None, n_splits=5, preprocess=None,
                   group_key=None, k=20, headline_metric="gene_roc_auc",
                   epsilon=None, make_figures=True, split_lock=None,
                   row_ids=None, report_meta=None, splits_table=None,
                   outdir="model_comparison_out"):
    """Run the full comparison protocol deterministically and write ONE
    canonical output set. Steps per model: point metrics (seed 0), permutation
    control (seed 0), Rashomon band (all seeds). Then pairwise tie decisions
    (|Δ| vs the larger band), remove-and-retrain ablation over feature_blocks
    (run on the Rashomon-set best model), the epsilon-Rashomon set of adequate
    models, and a cross-model convergence assessment among those members.

    Inputs
      X            : pandas DataFrame (rows = pairs/samples, cols = features)
      y            : binary labels aligned to X
      models       : {name: factory(seed)->estimator}
      groups       : leakage-safe grouping (held out whole in CV)
      feature_blocks : {family: [cols]} for ablation + convergence (optional)
      group_key    : per-row group for gene-level aggregation (optional)
      headline_metric : metric for band/tie/Rashomon decisions
      epsilon      : Rashomon tolerance; None => best model's seed band
      make_figures : emit one PNG per stage into outdir

    Writes (deterministic, sorted, rounded): metrics.csv, report.json,
    verdict.md, and fig_*.png under outdir. Returns the report dict."""
    import json
    import os

    import numpy as np
    import pandas as pd

    os.makedirs(outdir, exist_ok=True)
    if seeds is None:
        seeds = DEFAULT_SEEDS
    model_names = sorted(models)
    per_model = {}
    # Primary split: if a split_lock is supplied, ENFORCE it (hard stop on
    # mismatch) and use its fold assignment so this harness and the upstream
    # training harness evaluate on the byte-identical split. Otherwise derive
    # the canonical seed-0 split locally.
    lock_report = None
    if split_lock is not None:
        if row_ids is None:
            raise ValueError("split_lock requires row_ids to bind rows to folds")
        chk = verify_split_lock(split_lock, row_ids, strict=True)
        folds0 = np.asarray(chk["folds"])
        n_splits = int(split_lock["n_splits"])
        lock_report = {"fingerprint": chk["fingerprint"], "enforced": True}
    else:
        folds0 = make_folds(groups, n_splits, seed=RANDOM_SEED)
    for name in model_names:
        fac = models[name]
        point, _ = evaluate_model(fac, X, y, folds0, preprocess, RANDOM_SEED, group_key, k)
        perm = permutation_control(fac, X, y, folds0, preprocess, RANDOM_SEED, group_key, k)
        band = rashomon_band(fac, X, y, groups, seeds, n_splits, preprocess,
                             group_key, k, headline_metric)
        per_model[name] = {
            "point": point,
            "permutation": {kk: perm[kk] for kk in perm},
            "rashomon": band,
            "signal_gap": (point.get(headline_metric, float("nan"))
                           - perm.get(headline_metric, float("nan"))),
        }

    # pairwise tie decisions
    ties = []
    for i in range(len(model_names)):
        for j in range(i + 1, len(model_names)):
            a, b = model_names[i], model_names[j]
            va = per_model[a]["point"].get(headline_metric, float("nan"))
            vb = per_model[b]["point"].get(headline_metric, float("nan"))
            ba = per_model[a]["rashomon"].get("band", float("nan"))
            bb = per_model[b]["rashomon"].get("band", float("nan"))
            bands = [x for x in (ba, bb) if not np.isnan(x)]
            larger = float(max(bands)) if bands else float("nan")
            diff = abs(va - vb)
            ties.append({"model_a": a, "model_b": b,
                         "diff": float(diff), "larger_band": larger,
                         "tied": bool(diff < larger)})

    # epsilon-Rashomon set: which models are adequate enough to "vote"
    point_map = {m: per_model[m]["point"] for m in model_names}
    band_map = {m: per_model[m]["rashomon"] for m in model_names}
    rset = rashomon_set(point_map, band_map, headline_metric, epsilon)

    abl = None
    conv = None
    if feature_blocks:
        abl = ablation(models[rset["best_model"]], X, y, groups, feature_blocks,
                       seeds=tuple(seeds[:3]), n_splits=n_splits,
                       preprocess=preprocess, group_key=group_key, k=k,
                       metric=headline_metric)
        # convergence among Rashomon-set members only
        conv_metric = "roc_auc"
        conv = convergence(models, X, y, feature_blocks, folds0,
                           rset["members"], preprocess=preprocess,
                           seed=RANDOM_SEED, k=k, metric=conv_metric)

    report = p_round({
        "headline_metric": headline_metric, "k": k, "n_splits": n_splits,
        "seeds": list(seeds), "n_rows": int(len(y)),
        "n_features": int(X.shape[1]) if hasattr(X, "shape") else None,
        "models": per_model, "tie_decisions": ties,
        "rashomon_set": rset, "ablation": abl, "convergence": conv,
        "split_lock": lock_report, "tie_band_note": TIE_BAND_NOTE,
    })

    # flat metrics table
    rows = []
    for name in model_names:
        r = {"model": name}
        r.update({f"point_{kk}": vv for kk, vv in per_model[name]["point"].items()})
        r["perm_" + headline_metric] = per_model[name]["permutation"].get(headline_metric)
        r["signal_gap"] = per_model[name]["signal_gap"]
        r.update({f"band_{kk}": vv for kk, vv in per_model[name]["rashomon"].items()
                  if kk in ("mean", "std", "band")})
        rows.append(r)
    metrics_df = pd.DataFrame(rows).sort_values("model").reset_index(drop=True)
    metrics_df = metrics_df.round(FLOAT_ROUND)
    metrics_df.to_csv(os.path.join(outdir, "metrics.csv"), index=False)
    report_json = json.dumps(report, indent=2, sort_keys=True)
    with open(os.path.join(outdir, "report.json"), "w") as fh:
        fh.write(report_json)
    # determinism hash over the canonical report.json (identical inputs -> same hash)
    import hashlib
    det_hash = hashlib.sha256(report_json.encode()).hexdigest()[:12]
    meta = dict(report_meta or {})
    meta.setdefault("determinism_hash", det_hash)
    if split_lock is not None:
        meta.setdefault("split_name", split_lock.get("source", "locked split"))
    with open(os.path.join(outdir, "verdict.md"), "w") as fh:
        fh.write(p_verdict_md(report, metrics_df))
    with open(os.path.join(outdir, "report.md"), "w") as fh:
        fh.write(p_report_md(report, metrics_df, report_meta=meta,
                             splits_table=splits_table))
    if make_figures:
        try:
            report["figures"] = emit_figures(report, outdir)
        except Exception as exc:  # figures are non-fatal; numbers already written
            report["figures_error"] = str(exc)
    return report


def p_df_to_markdown(df):
    """Minimal markdown table without the optional `tabulate` dependency."""
    cols = list(df.columns)
    out = ["| " + " | ".join(str(c) for c in cols) + " |",
           "| " + " | ".join("---" for _ in cols) + " |"]
    for _, row in df.iterrows():
        out.append("| " + " | ".join(str(row[c]) for c in cols) + " |")
    return "\n".join(out)


def p_report_md(report, metrics_df, report_meta=None, splits_table=None):
    """Full human-readable report. Ordered the way a data scientist interrogates
    a model comparison: (0) provenance, (1) can I trust it, (2) headline with
    uncertainty, (3) is the signal real, (4) generalization across splits,
    (5) what drives the models, (6) do families agree, (7) caveats + next steps.

    report        the run_comparison result dict.
    metrics_df    the flat metrics table.
    report_meta   optional provenance dict: any of run_timestamp, harness_version,
                  adapter_version, feature_spec_fingerprint, determinism_hash,
                  n_genes, positive_rate, dataset_name, split_name.
    splits_table  optional cross-split generalization table (DataFrame or list of
                  dicts with at least split, model, gene_roc_auc) from the
                  training harness's metrics_by_split (adapter Mode A). Enables
                  Section 4; omitted otherwise.
    """
    import numpy as np
    meta = report_meta or {}
    hm = report["headline_metric"]

    def _fmt(x, nd=4):
        try:
            if x is None or (isinstance(x, float) and np.isnan(x)):
                return "n/a"
            return f"{float(x):.{nd}f}"
        except (TypeError, ValueError):
            return str(x)

    L = ["# Model-comparison report", ""]

    # ---- 0. Provenance ------------------------------------------------------
    L += ["## 0. Run provenance", ""]
    prov = []
    if meta.get("run_timestamp"):
        prov.append(f"- Run: {meta['run_timestamp']}")
    if meta.get("dataset_name"):
        prov.append(f"- Dataset: {meta['dataset_name']}")
    prov.append(f"- Split evaluated: {meta.get('split_name', 'seed-0 canonical (unlocked)')}")
    prov.append(f"- Rows (target-indication pairs): {report['n_rows']}"
                + (f" · genes: {meta['n_genes']}" if meta.get("n_genes") else "")
                + (f" · positive rate: {_fmt(meta['positive_rate'], 3)}"
                   if meta.get("positive_rate") is not None else ""))
    prov.append(f"- Features: {report.get('n_features')} · "
                f"{report['n_splits']}-fold · {len(report['seeds'])} seeds "
                f"({', '.join(str(s) for s in report['seeds'])})")
    prov.append(f"- Headline metric: `{hm}` · precision@k with k={report['k']}")
    prov.append(f"- Models compared: {', '.join(sorted(report['models']))}")
    for key, label in [("harness_version", "Harness version"),
                       ("adapter_version", "Adapter version"),
                       ("feature_spec_fingerprint", "feature_spec fingerprint")]:
        if meta.get(key):
            prov.append(f"- {label}: `{meta[key]}`")
    L += prov + [""]

    # ---- 1. Trust / integrity ----------------------------------------------
    L += ["## 1. Can I trust this experiment?", "",
          "Integrity checks that gate every result below. Read these first: a "
          "number is only as good as the split that produced it.", ""]
    lock = report.get("split_lock")
    if lock and lock.get("enforced"):
        L.append(f"- **Split lock: ENFORCED.** Every model was evaluated on the "
                 f"byte-identical fold assignment. Fingerprint `{lock.get('fingerprint')}`"
                 + (f" (source: {lock['source']})." if lock.get("source") else "."))
    else:
        L.append("- **Split lock: not enforced** for this run — folds derived "
                 "locally from the canonical seed. Supply `split_lock`+`row_ids` "
                 "to bind this run to the training harness's exact split.")
    L.append("- **Leakage guard:** whole groups (homolog clusters / genes) are "
             "held out together, so no train fold contains a near-duplicate of a "
             "test row. Any in-fold preprocessing (scaling / PCA) is refit on the "
             "train fold only.")
    if meta.get("determinism_hash"):
        L.append(f"- **Determinism:** `report.json` is byte-reproducible across "
                 f"runs on identical inputs (sha `{meta['determinism_hash']}`).")
    else:
        L.append("- **Determinism:** fixed seeds and sorted/rounded outputs make "
                 "`report.json` byte-reproducible across runs on identical inputs.")
    rset = report.get("rashomon_set")
    if rset:
        voters = set(rset.get("members", []))
        allm = sorted(report["models"])
        nonvote = [m for m in allm if m not in voters]
        L.append(f"- **Adequacy gate:** {len(voters)} of {len(allm)} models cleared "
                 f"the ε-Rashomon adequacy bar (ε={_fmt(rset.get('epsilon'))}) and "
                 f"therefore *vote* on the interpretation stages: "
                 f"{', '.join(sorted(voters))}."
                 + (f" Shown but non-voting (below the adequacy band): "
                    f"{', '.join(nonvote)}." if nonvote else ""))
    L += [""]

    # ---- 2. Headline with uncertainty --------------------------------------
    L += ["## 2. Headline result (with uncertainty)", "",
          "Point metrics with the seed band. **The band, not the point, is the "
          "unit of comparison** — a gap smaller than the band is noise.", "",
          p_df_to_markdown(metrics_df), ""]
    ties = report.get("tie_decisions", [])
    tied = [t for t in ties if t.get("tied")]
    sep = [t for t in ties if not t.get("tied")]
    L += ["### Tie decisions (|Δ| vs larger Rashomon band)", ""]
    if tied:
        L.append("**Statistically tied — do NOT tell a story about the score gap:**")
        for t in tied:
            L.append(f"- {t['model_a']} vs {t['model_b']}: Δ={_fmt(t['diff'])} "
                     f"< band {_fmt(t['larger_band'])}")
    if sep:
        L.append("")
        L.append("**Separable (gap exceeds the band):**")
        for t in sep:
            L.append(f"- {t['model_a']} vs {t['model_b']}: Δ={_fmt(t['diff'])} "
                     f"> band {_fmt(t['larger_band'])}")
    L += ["", "![Performance with Rashomon bands](fig_metrics.png)", ""]

    # ---- 3. Is the signal real ---------------------------------------------
    L += ["## 3. Is the signal real? (permutation control)", "",
          "Real metric minus a group-level label-permuted null. A gap near zero "
          "means the model is fitting chance structure the pipeline permits.", ""]
    for name in sorted(report["models"]):
        d = report["models"][name]
        pnull = d.get("permutation", {}).get(hm)
        real = d.get("point", {}).get(hm)
        L.append(f"- **{name}**: real {_fmt(real)} − null {_fmt(pnull)} = "
                 f"gap **{_fmt(d.get('signal_gap'))}**")
    L += ["", "![Signal gap](fig_permutation.png)", ""]

    # ---- 4. Generalization across splits -----------------------------------
    if splits_table is not None:
        L += ["## 4. Generalization across split strategies", "",
              "Headline metric under each split strategy. The gap between "
              "`random` (optimistic, not leakage-safe by design) and "
              "`seqclust`/`gene` (honest, homolog-safe) is the true "
              "generalization signal — how far the score falls on genuinely "
              "unseen proteins.", ""]
        try:
            import pandas as pd
            sdf = (splits_table if hasattr(splits_table, "columns")
                   else pd.DataFrame(splits_table))
            metric_col = "gene_roc_auc" if "gene_roc_auc" in sdf.columns else (
                "gene_roc" if "gene_roc" in sdf.columns else None)
            if metric_col and {"split", "model"}.issubset(sdf.columns):
                piv = sdf.pivot_table(index="model", columns="split",
                                      values=metric_col, aggfunc="first")
                order = [c for c in ["random", "gene", "disease", "seqclust"]
                         if c in piv.columns] or list(piv.columns)
                piv = piv[order].round(FLOAT_ROUND).reset_index()
                L += [p_df_to_markdown(piv), ""]
                if "random" in order and "seqclust" in order:
                    drop = (piv.set_index("model")["random"]
                            - piv.set_index("model")["seqclust"]).dropna()
                    if len(drop):
                        L.append(f"Mean optimism (random − seqclust): "
                                 f"**{_fmt(drop.mean())}** "
                                 f"(range {_fmt(drop.min())}–{_fmt(drop.max())}).")
            else:
                L.append("_Cross-split table supplied but missing split/model/metric "
                         "columns — skipped._")
        except Exception as exc:
            L.append(f"_Cross-split table could not be rendered: {exc}_")
        L += [""]

    # ---- 5. What drives the models -----------------------------------------
    abl = report.get("ablation")
    if abl:
        L += ["## 5. What drives the models? (remove-and-retrain ablation)", "",
              "Each feature family is dropped and the model refit from scratch "
              "(ROAR discipline). A family whose Δ is within its own seed band "
              "adds nothing — that is redundancy, not importance.", ""]
        full = abl.get("__full__", {})
        L.append(f"Full-model {hm} = {_fmt(full.get('mean'))} "
                 f"(band {_fmt(full.get('band'))}).")
        fams = sorted((f for f in abl if f != "__full__"),
                      key=lambda f: -abl[f]["delta_vs_full"])
        for f in fams:
            r = abl[f]
            band = r.get("reduced_band", 0.0) or 0.0
            meaningful = "meaningful" if r["delta_vs_full"] > max(band, 0.005) else "within noise"
            L.append(f"- drop **{f}** ({r.get('n_dropped')} cols): {hm} → "
                     f"{_fmt(r.get('reduced_mean'))} "
                     f"(Δ=**{r['delta_vs_full']:+.4f}**, block band {_fmt(band)}) "
                     f"— *{meaningful}*")
        L += ["", "![Ablation by feature family](fig_ablation.png)", ""]

    # ---- 6. Do families agree ----------------------------------------------
    conv = report.get("convergence")
    if conv and conv.get("consensus_importance"):
        L += ["## 6. Do independent model families agree? (convergence)", "",
              "Each adequate model is projected onto grouped permutation "
              "importance over feature *families* (robust to Rashomon "
              "feature-swapping). Agreement across families with different "
              "inductive biases is the citable interpretability result.", ""]
        L.append(f"- Voting members: {', '.join(conv.get('members', []))}")
        L.append(f"- Mean pairwise Kendall τ over family rankings: "
                 f"**{_fmt(conv.get('mean_tau'), 3)}** "
                 f"· mean top-{conv.get('top_k')} family Jaccard: "
                 f"{_fmt(conv.get('mean_topk_jaccard'), 3)}")
        ci = sorted(conv["consensus_importance"].items(), key=lambda kv: -kv[1])
        L.append("- Consensus family importance (mean over members): "
                 + ", ".join(f"{f} {_fmt(v, 2)}" for f, v in ci))
        L += ["", "> Internal agreement (models agree with each other) becomes "
              "*external* validity only when the converged families match a "
              "domain prior (e.g. channels / GPCR / membrane druggability). "
              "State that check explicitly before claiming established signal.",
              "", "![Family × model importance](fig_convergence.png)", ""]

    # ---- 7. Caveats + next steps -------------------------------------------
    L += ["## 7. Caveats and next steps", ""]
    caveats = []
    # non-fitting models
    for name in sorted(report["models"]):
        pt = report["models"][name].get("point", {})
        if pt.get(hm) is None or (isinstance(pt.get(hm), float) and np.isnan(pt.get(hm))):
            caveats.append(f"Model **{name}** produced no {hm} on this split "
                           "(failed to fit or degenerate) — excluded from ranking.")
    if rset:
        nonvote = [m for m in sorted(report["models"]) if m not in set(rset.get("members", []))]
        if nonvote:
            caveats.append(f"Non-voting models ({', '.join(nonvote)}) are below the "
                           "adequacy band; their explanations are reported but do "
                           "not contribute to the convergence result.")
    if tied:
        caveats.append(f"{len(tied)} model pair(s) are within the Rashomon band and "
                       "cannot be ranked against each other on this metric.")
    if meta.get("score_not_calibrated", True):
        caveats.append("The model outputs a **raw score, not a calibrated "
                       "probability** — operating-point / tier decisions need a "
                       "separate calibration step (known limitation).")
    caveats += list(meta.get("extra_caveats", []) or [])
    if not caveats:
        caveats = ["No blocking caveats detected on this run."]
    for c in caveats:
        L.append(f"- {c}")

    L += ["", "---", "", "### Interpretation rule", "", report["tie_band_note"], ""]
    return "\n".join(L)


def p_verdict_md(report, metrics_df):
    """Human-readable verdict: metrics table, tie calls, ablation, and the
    one-line interpretation rule."""
    hm = report["headline_metric"]
    lines = ["# Model-comparison verdict", ""]
    lines.append(f"Headline metric: `{hm}` · {report['n_rows']} rows · "
                 f"{report.get('n_features')} features · {report['n_splits']}-fold · "
                 f"{len(report['seeds'])} seeds")
    lines += ["", "## Metrics", "", p_df_to_markdown(metrics_df), ""]
    lines += ["## Tie decisions (|Δ| vs larger Rashomon band)", ""]
    for t in report["tie_decisions"]:
        verdict = "TIED — compare explanations, not scores" if t["tied"] else "separable"
        band_s = "nan" if t["larger_band"] != t["larger_band"] else f"{t['larger_band']:.4f}"
        lines.append(f"- **{t['model_a']} vs {t['model_b']}**: "
                     f"Δ={t['diff']:.4f}, band={band_s} → {verdict}")
    rset = report.get("rashomon_set")
    if rset:
        lines += ["", "## ε-Rashomon set (adequate models that vote on convergence)", ""]
        lines.append(f"Best: **{rset['best_model']}** ({rset['best_value']:.4f}); "
                     f"ε={rset['epsilon']:.4f}; members: {', '.join(rset['members'])}")
    conv = report.get("convergence")
    if conv and conv.get("consensus_importance"):
        lines += ["", "## Cross-model convergence (grouped permutation importance)", ""]
        lines.append(f"Mean pairwise Kendall τ over families: **{conv['mean_tau']:.3f}** "
                     f"· mean top-{conv['top_k']} family Jaccard: {conv['mean_topk_jaccard']:.3f}")
        ci = sorted(conv["consensus_importance"].items(), key=lambda kv: -kv[1])
        lines.append("Consensus family importance (mean over members): "
                     + ", ".join(f"{f} {v:.2f}" for f, v in ci))
    if report.get("ablation"):
        lines += ["", "## Ablation (remove-and-retrain, best model)", ""]
        full = report["ablation"].get("__full__", {})
        lines.append(f"Full model {hm} = {full.get('mean')}")
        for name, r in report["ablation"].items():
            if name == "__full__":
                continue
            lines.append(f"- drop **{name}** ({r['n_dropped']} cols): "
                         f"{hm} → {r['reduced_mean']} (Δ={r['delta_vs_full']:+.4f}, "
                         f"block band {r['reduced_band']:.4f})")
    lines += ["", "## Rule", "", report["tie_band_note"], ""]
    return "\n".join(lines)
