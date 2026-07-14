#!/usr/bin/env python3
"""
CVS batch model-training harness  —  DGX Spark (GB10), aarch64, RAPIDS/CUDA.

Trains a set of model families for the CNS Clinical Viability Score under a SET of
train/validation split strategies, in one leakage-safe pass, and emits:
  * fitted models (per split, refit on all in-distribution data)  -> feed the interp harness
  * out-of-fold predictions + gene-level metrics per split         -> generalization table
  * a permutation control and human-readable model descriptors

The two design ideas the user asked for:
  (1) OUTPUT MODELS that plug straight into the interpretability harness. Every family is
      saved as a loadable object under a FROZEN feature spec (feature_spec.json), so the
      interp harness rebuilds the identical X and runs SHAP / ablation / Rashomon on the
      exact fitted estimators produced here.
  (2) ITERATE OVER SPLIT STRATEGIES. Chris-Deotte discipline: the local score is only
      trustworthy if the split mimics the real train/test relationship — so instead of
      guessing one split, we run the SAME models under several and report how each degrades:
        seqclust  frozen homolog-safe GroupKFold (headline; "unseen protein clusters")
        gene      GroupKFold on gene, homolog-cluster-safe (unseen genes)
        disease   GroupKFold on disease (unseen indications)
        random    StratifiedKFold (optimistic reference — NOT leakage-safe, by design)
      The gap between `random` and `seqclust`/`gene` is the honest generalization signal.

Everything trains fast on CUDA: XGBoost families use device=cuda + QuantileDMatrix; the
small additive/linear/rule families are CPU (no aarch64 GPU build) but are tiny.

Shared, fixed representation so families are comparable: 164 interpretable scalars +
in-fold PCA-32 of the ESMC-6B embedding (PCA refit INSIDE each train fold — no leakage).

------------------------------------------------------------------------------------------
ADD A MODEL  -> write  def _train_<name>(Xtr,ytr,Xte,ctx) -> (proba_te, fitted_or_None, repr_or_None)
                and add its name to MODELS.
ADD A SPLIT  -> write  def _split_<name>(df) -> [(train_mask, test_mask), ...]
                and add its name to SPLITS.  (group splits should call _cluster_safe)
------------------------------------------------------------------------------------------

Inputs staged into the workdir:
  gold.parquet       modeling matrix (label_consensus, fold_seqclust, efo_id, ensembl_gene_id,
                     gene_symbol, homol_cluster [optional], all interpretable + esmc6b columns)
  manifest.csv       column -> block map (block=='esmc6b' are the 2560 embedding dims)
  monotone_dirs.json {feature:+1/-1} data-derived monotone priors (optional; mono_gbm)

Outputs (OUT):
  oof_predictions.parquet   ids + oof_<split>_<model> columns (+ permuted-label OOF for headline)
  metrics_by_split.csv      one row per (split, model): gene/pair/within/p@k (+perm for headline)
  model_reprs.json          scorecard points / CART tree / RIPPER rules + status per split
  feature_spec.json         exactly how X is built (cols, PCA_K, scaling) — interp-harness contract
  models/<split>/<model>.*  fitted estimators refit on all in-distribution rows of that split
"""
import json
import os
import pickle
import time
import warnings

import numpy as np
import pandas as pd

# Scope noise suppression to the two known-benign, high-volume sources (sklearn convergence
# chatter on the L1/elastic-net fits, and RAPIDS/px future-deprecation notices) rather than a
# blanket filter — a global ignore would also swallow warnings that signal real problems.
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*ConvergenceWarning.*")
try:
    from sklearn.exceptions import ConvergenceWarning
    warnings.filterwarnings("ignore", category=ConvergenceWarning)
except Exception:
    pass

# ======================================================================================
# CONFIG  — the only block you normally edit
# ======================================================================================
OUT          = os.environ.get("OUT", "/work")   # entry point points this at a timestamped run dir
GOLD         = "gold.parquet"
MANIFEST     = "manifest.csv"
MONO         = "monotone_dirs.json"
LABEL_COL    = "label_consensus"
GENE_COL     = "ensembl_gene_id"
DISEASE_COL  = "efo_id"
SEQCLUST_COL = "fold_seqclust"          # frozen homolog-safe fold ids (0..4)
CLUSTER_COL  = "homol_cluster"          # for leakage-safe train exclusion (optional)
EMB_BLOCK    = "esmc6b"
INTERP_BLOCKS= ["intrinsic_84", "predicted_go", "structure", "homology"]
PCA_K        = 32
SCALE_BEFORE_PCA = True
XGB_ROUNDS   = 400
N_SPLITS_KF  = 5
SEED         = 0
DEVICE       = "cuda"                    # "cpu" fallback if no CUDA build
SPLITS       = ["seqclust", "gene", "disease", "random"]
HEADLINE     = "seqclust"                # split used for deploy refit + permutation control
MODELS       = ["xgb_baseline", "mono_gbm", "elasticnet_glm", "cart",
                "scorecard", "ripper", "ga2m", "nam", "knn"]
REFIT_FULL   = True                      # save each family refit on all rows (per split)
SAVE_MODELS  = True                      # persist fitted objects for the interp harness

# ======================================================================================
_t0 = time.time()
def log(m): print(f"[t+{time.time()-_t0:5.0f}s] {m}", flush=True)

# GPU acceleration (RAPIDS cuda-x-data-science playbook): cuml.accel transparently routes
# sklearn PCA / StandardScaler / LogisticRegression onto the GB10 — zero code change, with
# automatic CPU fallback for any unsupported estimator/param. This is the sanctioned
# tabular-acceleration path on this hardware. XGBoost families already run device=cuda.
ACCEL = False
try:
    from cuml.accel import install as _cuml_install
    _cuml_install(); ACCEL = True
    log("cuml.accel installed — PCA/StandardScaler/LogisticRegression routed to GPU")
except Exception as _e:
    log(f"cuml.accel unavailable ({type(_e).__name__}) — sklearn on CPU")

from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import GroupKFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, export_text

df = None
man = None
dirs = None
interp_cols = None
emb_cols = None
pc_names = None
FEAT_NAMES = None
y = None
gene = None
n = None
mono_vec = None
has_cluster = None
cluster = None
rng = None
uniq = None
gmap = None
lab_by_gene = None
y_perm = None


def load_data():
    """Load gold matrix + manifest and derive all module-level arrays. Idempotent;
    call once before run_split/build_fold. Kept out of import so the module is a clean,
    side-effect-free library the sweep job and tests can import."""
    global df, man, dirs, interp_cols, emb_cols, pc_names, FEAT_NAMES, y, gene, n, mono_vec, has_cluster, cluster, rng, uniq, gmap, lab_by_gene, y_perm
    if 'df' in globals() and df is not None:
        return  # already loaded
    df   = pd.read_parquet(GOLD)
    man  = pd.read_csv(MANIFEST)
    dirs = json.load(open(MONO)) if os.path.exists(MONO) else {}

    interp_cols = [c for c in man[man.block.isin(INTERP_BLOCKS)].column if c in df.columns]
    emb_cols    = [c for c in man[man.block == EMB_BLOCK].column if c in df.columns]
    pc_names    = [f"emb_pc{i}" for i in range(PCA_K)]
    FEAT_NAMES  = interp_cols + pc_names
    y     = df[LABEL_COL].astype(int).values
    gene  = df[GENE_COL].astype(str).values
    n     = len(df)
    mono_vec = [int(dirs.get(f, 0)) for f in FEAT_NAMES]
    has_cluster = CLUSTER_COL in df.columns
    cluster = df[CLUSTER_COL].astype(str).values if has_cluster else gene   # fall back to gene identity
    rng = np.random.default_rng(SEED)
    # gene-permuted labels for the honest null (all pairs of a gene share features)
    uniq = pd.unique(gene); gmap = dict(zip(uniq, rng.permutation(uniq), strict=False))
    lab_by_gene = pd.Series(y, index=gene).groupby(level=0).first()
    y_perm = np.array([lab_by_gene[gmap[g]] for g in gene], dtype=int)
    log(f"{df.shape}: {len(interp_cols)} interp + {PCA_K} PC; cluster_col={'present' if has_cluster else 'MISSING->gene'}; "
        f"splits={SPLITS}; models={MODELS}")



# ======================================================================================
# SPLIT REGISTRY  — each: (df) -> [(train_mask, test_mask), ...]
# _cluster_safe drops from TRAIN any row sharing a homolog cluster with the TEST block,
# so a near-identical paralog cannot leak across the split (except `random`, left raw).
# ======================================================================================
def _cluster_safe(folds):
    out = []
    for tr, te in folds:
        bad = set(cluster[te])
        out.append((tr & ~np.isin(cluster, list(bad)), te))
    return out
def _groupkfold(values):
    idx = np.arange(n); folds = []
    for tr, te in GroupKFold(N_SPLITS_KF).split(idx, groups=values):
        mtr = np.zeros(n, bool); mte = np.zeros(n, bool); mtr[tr] = True; mte[te] = True
        folds.append((mtr, mte))
    return folds
def _split_seqclust(df):
    g = df[SEQCLUST_COL].astype(int).values          # already homolog-safe by construction
    return [(g != k, g == k) for k in sorted(np.unique(g))]
def _split_gene(df):     return _cluster_safe(_groupkfold(gene))
def _split_disease(df):  return _cluster_safe(_groupkfold(df[DISEASE_COL].astype(str).values))
def _split_random(df):
    idx = np.arange(n); folds = []
    for tr, te in StratifiedKFold(N_SPLITS_KF, shuffle=True, random_state=SEED).split(idx, y):
        mtr = np.zeros(n, bool); mte = np.zeros(n, bool); mtr[tr] = True; mte[te] = True
        folds.append((mtr, mte))                     # deliberately NOT cluster-safe: optimistic ref
    return folds
SPLIT_REG = {"seqclust": _split_seqclust, "gene": _split_gene,
             "disease": _split_disease, "random": _split_random}

# ---- shared leakage-safe feature construction (in-fold PCA of the embedding) -----------
def build_fold(tr_mask, te_mask):
    E = df[emb_cols].values.astype(np.float32)
    cm = np.nanmean(E[tr_mask], axis=0); cm = np.where(np.isfinite(cm), cm, 0.0)
    E = np.where(np.isnan(E), cm, E)
    Etr, Ete = E[tr_mask], E[te_mask]
    if SCALE_BEFORE_PCA:
        ss = StandardScaler().fit(Etr); Etr, Ete = ss.transform(Etr), ss.transform(Ete)
    pca = PCA(PCA_K, random_state=SEED).fit(Etr)
    Xtr = np.hstack([df.loc[tr_mask, interp_cols].values, pca.transform(Etr)]).astype(np.float32)
    Xte = np.hstack([df.loc[te_mask, interp_cols].values, pca.transform(Ete)]).astype(np.float32)
    return np.nan_to_num(Xtr), np.nan_to_num(Xte)

def binarize_fit(Xi):
    edges = []
    for j in range(Xi.shape[1]):
        col = Xi[:, j]; u = np.unique(col[np.isfinite(col)])
        edges.append(("bin", u) if len(u) <= 2 else ("q", np.quantile(col[np.isfinite(col)], [.25, .5, .75])))
    return edges
def binarize_apply(Xi, edges):
    cols, names = [], []
    for j, (kind, e) in enumerate(edges):
        nm = interp_cols[j]
        if kind == "bin":
            cols.append((Xi[:, j] > 0.5).astype(np.int8)); names.append(f"{nm}>0.5")
        else:
            for q, thr in zip([25, 50, 75], e, strict=False):
                cols.append((Xi[:, j] > thr).astype(np.int8)); names.append(f"{nm}>q{q}")
    return np.vstack(cols).T, names

# ======================================================================================
# MODEL REGISTRY  — each: (Xtr,ytr,Xte,ctx) -> (proba_te, fitted_or_None, repr_or_None)
# ======================================================================================
def _xgb_params(mono=False):
    P = dict(objective="binary:logistic", eval_metric="auc", max_depth=6, colsample_bytree=0.7,
             subsample=0.8, min_child_weight=5, reg_lambda=2.0, learning_rate=0.03,
             tree_method="hist", device=DEVICE)
    if mono: P["monotone_constraints"] = "(" + ",".join(map(str, mono_vec)) + ")"
    return P
def _train_xgb_baseline(Xtr, ytr, Xte, ctx):
    import xgboost as xgb
    dtr = xgb.QuantileDMatrix(Xtr, label=ytr, feature_names=FEAT_NAMES)
    dte = xgb.QuantileDMatrix(Xte, ref=dtr, feature_names=FEAT_NAMES)
    b = xgb.train(_xgb_params(), dtr, num_boost_round=XGB_ROUNDS)
    return b.predict(dte), b, None
def _train_mono_gbm(Xtr, ytr, Xte, ctx):
    import xgboost as xgb
    dtr = xgb.QuantileDMatrix(Xtr, label=ytr, feature_names=FEAT_NAMES)
    dte = xgb.QuantileDMatrix(Xte, ref=dtr, feature_names=FEAT_NAMES)
    b = xgb.train(_xgb_params(mono=True), dtr, num_boost_round=XGB_ROUNDS)
    return b.predict(dte), b, None
def _train_elasticnet_glm(Xtr, ytr, Xte, ctx):
    ss = StandardScaler().fit(Xtr)
    m = LogisticRegression(penalty="elasticnet", l1_ratio=0.5, C=0.1, solver="saga",
                           max_iter=2000, class_weight="balanced").fit(ss.transform(Xtr), ytr)
    return m.predict_proba(ss.transform(Xte))[:, 1], (ss, m), None
def _train_cart(Xtr, ytr, Xte, ctx):
    ct = DecisionTreeClassifier(max_depth=4, min_samples_leaf=50, class_weight="balanced",
                                random_state=SEED, ccp_alpha=0.001).fit(ctx["Xi_tr"], ytr)
    rep = export_text(ct, feature_names=list(interp_cols), max_depth=4)[:4000] if ctx["first"] else None
    return ct.predict_proba(ctx["Xi_te"])[:, 1], ct, rep
def _train_scorecard(Xtr, ytr, Xte, ctx):
    Btr, Bte, bnames = ctx["Btr"], ctx["Bte"], ctx["bnames"]
    g = LogisticRegression(penalty="l1", C=0.05, solver="liblinear", class_weight="balanced",
                           max_iter=2000).fit(Btr, ytr)
    coef = g.coef_[0]
    nz = np.abs(coef[coef != 0])
    scale = 2.0 / (nz.mean() + 1e-9) if nz.size else 1.0     # guard all-zero coef (degenerate data)
    pts = np.round(coef * scale).astype(int)
    raw = Bte @ pts + g.intercept_[0] * scale
    proba = 1 / (1 + np.exp(-(raw - raw.mean()) / (raw.std() + 1e-9)))
    rep = [(nm, int(p)) for nm, p in sorted(zip(bnames, pts, strict=False), key=lambda x: -abs(x[1])) if p != 0][:25] if ctx["first"] else None
    return proba, (g, pts), rep
def _train_ripper(Xtr, ytr, Xte, ctx):
    import wittgenstein as lw
    Btr, Bte, bnames = ctx["Btr"], ctx["Bte"], ctx["bnames"]
    rip = lw.RIPPER(random_state=SEED, max_rules=12)
    rip.fit(pd.DataFrame(Btr, columns=bnames), ytr)
    pr = np.asarray(rip.predict_proba(pd.DataFrame(Bte, columns=bnames)))
    return (pr[:, 1] if pr.ndim == 2 else pr), rip, (str(rip.ruleset_)[:3000] if ctx["first"] else None)
def _train_ga2m(Xtr, ytr, Xte, ctx):
    from interpret.glassbox import ExplainableBoostingClassifier
    ga = ExplainableBoostingClassifier(random_state=SEED, interactions=10, n_jobs=-1)
    ga.fit(pd.DataFrame(Xtr, columns=FEAT_NAMES), ytr)
    return ga.predict_proba(pd.DataFrame(Xte, columns=FEAT_NAMES))[:, 1], ga, None
def _train_nam(Xtr, ytr, Xte, ctx):
    import torch
    import torch.nn as nn_t
    torch.manual_seed(SEED)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ss = StandardScaler().fit(Xtr)
    Xt = torch.tensor(ss.transform(Xtr), dtype=torch.float32, device=dev)
    Xv = torch.tensor(ss.transform(Xte), dtype=torch.float32, device=dev)
    yt = torch.tensor(ytr, dtype=torch.float32, device=dev); F = Xt.shape[1]
    class FeatNet(nn_t.Module):
        def __init__(s): super().__init__(); s.n = nn_t.Sequential(
            nn_t.Linear(1, 32), nn_t.ReLU(), nn_t.Linear(32, 16), nn_t.ReLU(), nn_t.Linear(16, 1))
        def forward(s, x): return s.n(x)
    class NAM(nn_t.Module):
        def __init__(s): super().__init__(); s.fs = nn_t.ModuleList([FeatNet() for _ in range(F)]); s.b = nn_t.Parameter(torch.zeros(1))
        def forward(s, x): return sum(s.fs[i](x[:, i:i+1]) for i in range(F)).squeeze(1) + s.b
    m = NAM().to(dev); opt = torch.optim.Adam(m.parameters(), lr=1e-3, weight_decay=1e-5)
    pw = torch.tensor([(ytr == 0).sum() / max((ytr == 1).sum(), 1)], dtype=torch.float32, device=dev)
    lossf = nn_t.BCEWithLogitsLoss(pos_weight=pw)
    for _ in range(60):
        opt.zero_grad(); loss = lossf(m(Xt), yt); loss.backward(); opt.step()
    with torch.no_grad():
        return torch.sigmoid(m(Xv)).cpu().numpy(), (ss, m.cpu()), None
def _train_knn(Xtr, ytr, Xte, ctx):
    # SAME 196-feature matrix as every other family so the benchmark row is comparable.
    # Standardize inside so Euclidean distance is fair across mixed-scale features
    # (binary flags vs counts 0-24 vs z-scores). cuML on GPU if present, else sklearn.
    ss = StandardScaler().fit(Xtr)
    Xt, Xv = ss.transform(Xtr), ss.transform(Xte)
    try:
        from cuml.neighbors import KNeighborsClassifier as GKNN
        m = GKNN(n_neighbors=50); m.fit(Xt.astype(np.float32), ytr.astype(np.float32))
        pr = m.predict_proba(Xv.astype(np.float32))
        pr = pr.to_numpy() if hasattr(pr, "to_numpy") else np.asarray(pr)
        return (pr[:, 1] if pr.ndim == 2 else pr), None, None
    except Exception:
        from sklearn.neighbors import KNeighborsClassifier as SKNN
        m = SKNN(n_neighbors=50).fit(Xt, ytr)
        return m.predict_proba(Xv)[:, 1], None, None

REGISTRY = {"xgb_baseline": _train_xgb_baseline, "mono_gbm": _train_mono_gbm,
            "elasticnet_glm": _train_elasticnet_glm, "cart": _train_cart,
            "scorecard": _train_scorecard, "ripper": _train_ripper,
            "ga2m": _train_ga2m, "nam": _train_nam, "knn": _train_knn}

def save_fitted(obj, path_stub):
    """Persist a fitted family for the interp harness. XGBoost -> .ubj; torch -> .pt; else pickle."""
    try:
        import xgboost as xgb
        if isinstance(obj, xgb.Booster):
            obj.save_model(path_stub + ".ubj"); return "ubj"
    except Exception:
        pass
    try:
        import torch
        import torch.nn as nn_t
        if isinstance(obj, tuple) and any(isinstance(o, nn_t.Module) for o in obj):
            torch.save({"blob": obj}, path_stub + ".pt"); return "pt"
    except Exception:
        pass
    with open(path_stub + ".pkl", "wb") as fh:
        pickle.dump(obj, fh); return "pkl"

# ======================================================================================
# per-split OOF loop (+ optional permutation on the headline split)
# ======================================================================================
def run_split(split_name, labels):
    folds = SPLIT_REG[split_name](df)
    oof = {m: np.full(n, np.nan) for m in MODELS}
    status = {m: "ok" for m in MODELS}; reprs = {}
    for fi, (tr, te) in enumerate(folds):
        Xtr, Xte = build_fold(tr, te); ytr = labels[tr]
        Xi_tr = np.nan_to_num(df.loc[tr, interp_cols].values)
        Xi_te = np.nan_to_num(df.loc[te, interp_cols].values)
        edges = binarize_fit(Xi_tr)
        Btr, bnames = binarize_apply(Xi_tr, edges); Bte, _ = binarize_apply(Xi_te, edges)
        ctx = dict(first=(fi == 0), Xi_tr=Xi_tr, Xi_te=Xi_te, Btr=Btr, Bte=Bte, bnames=bnames)
        for m in MODELS:
            if status[m] != "ok": continue
            try:
                proba, _fitted, rep = REGISTRY[m](Xtr, ytr, Xte, ctx)
                oof[m][te] = proba
                if rep is not None: reprs[m] = rep
            except Exception as e:
                status[m] = f"fail:{type(e).__name__}:{e}"
        log(f"[{split_name}] fold {fi} ({int(te.sum())} test, {int(tr.sum())} train): "
            + ", ".join(f"{m}={'ok' if status[m]=='ok' else 'X'}" for m in MODELS))
    return oof, status, reprs

# ======================================================================================
# metrics
# ======================================================================================
def gene_roll(p, lab=None):
    lab = y if lab is None else lab      # bind at call time (y is populated by load_data)
    d = pd.DataFrame({"g": gene, "y": lab, "p": p}).dropna().groupby("g").mean()
    return (d.y > 0.5).astype(int).values, d.p.values
def within_disease(p):
    aucs = []
    for _, s in pd.DataFrame({"d": df[DISEASE_COL].values, "y": y, "p": p}).dropna().groupby("d"):
        if s.y.nunique() == 2 and len(s) >= 10: aucs.append(roc_auc_score(s.y, s.p))
    return float(np.mean(aucs)) if aucs else np.nan
def p_at_k(p, k=20):
    d = pd.DataFrame({"g": gene, "y": y, "p": p}).dropna().groupby("g").mean().sort_values("p", ascending=False)
    return float((d.y.head(k) > 0.5).mean())


def _atomic_write(path, writer):
    """Write via a temp file + os.replace so a crash/interrupt never leaves a partial or corrupt
    output — the previous checkpoint stays intact until the new one is fully on disk."""
    tmp = f"{path}.tmp"
    writer(tmp)
    os.replace(tmp, path)


def main():
    rows = []; all_oof = {}; all_reprs = {}; all_status = {}

    # feature_spec is a pure function of config (the interp-harness contract) — write it BEFORE
    # training so it exists even if the run is interrupted partway.
    _atomic_write(f"{OUT}/feature_spec.json", lambda p: json.dump(
        {"interp_cols": interp_cols, "pc_names": pc_names, "feat_names": FEAT_NAMES,
         "emb_block": EMB_BLOCK, "pca_k": PCA_K, "scale_before_pca": SCALE_BEFORE_PCA,
         "label_col": LABEL_COL, "gene_col": GENE_COL, "disease_col": DISEASE_COL,
         "headline_split": HEADLINE, "splits": SPLITS, "models": MODELS,
         "xgb_rounds": XGB_ROUNDS, "seed": SEED}, open(p, "w"), indent=2))

    def flush_eval():
        """CRASH-SAFE CHECKPOINT. Write metrics + OOF + reprs for the splits finished SO FAR.
        Called after every split so an interrupt never discards a completed split's results, and
        once more at the end for the final state (incl. permutation columns). Atomic writes."""
        done = [s for s in SPLITS if s in all_oof]
        ids = df[[GENE_COL, DISEASE_COL, "gene_symbol", LABEL_COL, SEQCLUST_COL]].copy()
        ids["gidx"] = np.arange(n)
        for s in done:
            for m in MODELS:
                ids[f"oof_{s}_{m}"] = all_oof[s][m]
        for m in all_oof.get("_perm", {}):
            ids[f"permoof_{m}"] = all_oof["_perm"][m]
        _atomic_write(f"{OUT}/oof_predictions.parquet", ids.to_parquet)
        _atomic_write(f"{OUT}/metrics_by_split.csv", lambda p: pd.DataFrame(rows).to_csv(p, index=False))
        _atomic_write(f"{OUT}/model_reprs.json", lambda p: json.dump(
            {"status_by_split": all_status, "reprs_headline": all_reprs.get(HEADLINE, {})}, open(p, "w"), indent=2))

    for sp in SPLITS:
        oof, status, reprs = run_split(sp, y)
        all_oof[sp] = oof; all_reprs[sp] = reprs; all_status[sp] = status
        perm_oof = run_split(sp, y_perm)[0] if sp == HEADLINE else {}
        for m in MODELS:
            if status[m] != "ok" or np.isnan(oof[m]).all():
                rows.append({"split": sp, "model": m, "status": status[m]}); continue
            gy, gp = gene_roll(oof[m]); ok = ~np.isnan(oof[m])
            r = {"split": sp, "model": m, "status": "ok",
                 "gene_roc": round(roc_auc_score(gy, gp), 4), "gene_pr": round(average_precision_score(gy, gp), 4),
                 "pair_roc": round(roc_auc_score(y[ok], oof[m][ok]), 4),
                 "within_disease_roc": round(within_disease(oof[m]), 4),
                 "p_at_20": round(p_at_k(oof[m]), 3), "n_scored": int(ok.sum())}
            if sp == HEADLINE and m in perm_oof and not np.isnan(perm_oof[m]).all():
                pgy, pgp = gene_roll(perm_oof[m], y_perm)
                r["perm_gene_roc"] = round(roc_auc_score(pgy, pgp), 4)
                all_oof.setdefault("_perm", {})[m] = perm_oof[m]
            rows.append(r)
            log(f"{sp}/{m:15s} gene_roc={r['gene_roc']} p@20={r['p_at_20']} perm={r.get('perm_gene_roc','-')}")
        flush_eval()                                    # checkpoint: this split's results are now on disk
        log(f"[{sp}] eval checkpoint written ({len(rows)} metric rows)")

    flush_eval()                                        # final flush (adds permutation columns/rows)

    # refit each family on ALL in-distribution rows per split and SAVE for the interp harness
    if REFIT_FULL and SAVE_MODELS:
        saved = {}
        for sp in SPLITS:
            # in-distribution = all rows (random/seqclust) — for group splits we still refit on all,
            # since the deploy/interp model is the "seen everything" estimator
            os.makedirs(f"{OUT}/models/{sp}", exist_ok=True)
            tr = np.ones(n, bool)
            Xfull, _ = build_fold(tr, tr)
            Xi = np.nan_to_num(df[interp_cols].values)
            edges = binarize_fit(Xi); Bfull, bnames = binarize_apply(Xi, edges)
            ctx = dict(first=False, Xi_tr=Xi, Xi_te=Xi, Btr=Bfull, Bte=Bfull, bnames=bnames)
            for m in MODELS:
                if all_status[sp][m] != "ok": continue
                try:
                    _p, fitted, _r = REGISTRY[m](Xfull, y, Xfull, ctx)
                    if fitted is not None:
                        saved[f"{sp}/{m}"] = save_fitted(fitted, f"{OUT}/models/{sp}/{m}")
                except Exception as e:
                    saved[f"{sp}/{m}"] = f"skip:{e}"
            log(f"[{sp}] refit+saved: " + ", ".join(k.split('/')[1] for k in saved if k.startswith(sp+'/')))
            # checkpoint the model index after each split so a crash mid-refit keeps the models
            # already written on disk discoverable (the .pkl files themselves are written per-fit).
            _atomic_write(f"{OUT}/models/index.json", lambda p: json.dump(saved, open(p, "w"), indent=2))

    log(f"DONE  splits={SPLITS}  status={all_status}")
    return all_status


if __name__ == "__main__":
    # Provenance + fail-loud GPU guard (see cvs_run_logging). run_meta.json is written to OUT
    # alongside the analytical outputs; it is the block a methods section / git repo cites, and
    # is deliberately EXCLUDED from the report's determinism hash (it carries clocks + versions).
    import cvs_run_logging as RL
    _lg = RL.RunLogger(OUT, tag="cvs_train_harness")
    _pf = RL.gpu_preflight(logger=_lg, require=(DEVICE == "cuda"))
    load_data()
    try:
        _status = main()
        _outs = [f"{OUT}/{f}" for f in ("metrics_by_split.csv", "oof_predictions.parquet",
                 "model_reprs.json", "feature_spec.json")]
        RL.write_run_meta(OUT, extra={"job": "cvs_train_harness", "seed": SEED, "device": DEVICE,
                          "splits": SPLITS, "models": MODELS, "xgb_rounds": XGB_ROUNDS,
                          "pca_k": PCA_K, "status_by_split": _status}, preflight=_pf, outputs=_outs)
        _lg.close("ok")
    except Exception:
        _lg.close("error"); raise
