"""Adapter: CVS training harness (cvs_train_harness.py) -> model-comparison harness (kernel.py).

Bridges the two harnesses the two Claudes built. The training harness produces
per-split fitted models + OOF predictions + a feature_spec.json contract; this
comparison harness needs factory(seed)->estimator objects it can RETRAIN per
fold/seed. This adapter closes that gap without re-deriving anything:

  * build_matrix()      rebuilds X = raw interp cols + raw embedding cols (NOT
                        the pre-PCA'd 32 dims), keyed to feature_spec, plus the
                        feature-FAMILY map from the manifest.
  * RegistryEstimator   wraps each training-harness model family as an
                        sklearn-style fit/predict_proba estimator that does the
                        in-fold PCA of the embedding INSIDE fit (refit on the
                        train rows only) - so retraining across folds/seeds
                        stays leakage-safe and faithful to build_fold().
  * make_factories()    {name: factory(seed)} ready for run_comparison(models=).
  * build_split_lock_from_seqclust()  turns the frozen fold_seqclust column into
                        a fingerprinted split lock both harnesses can enforce.
  * ingest_oof()        Mode A: read the training harness's own oof_predictions
                        + metrics_by_split and produce the cross-model
                        comparison table with no retraining.

WHY raw embedding columns, not the 32 PCs: the training harness fits the PCA
INSIDE each train fold (build_fold) to avoid leakage. If the adapter handed the
comparison harness a globally-PCA'd matrix, the PCA would have seen every row -
leaky. So X carries the raw 2560-dim embedding and each estimator refits the
PCA on its own training fold, exactly mirroring build_fold.

FEATURE-SPACE NOTE: the training harness's families do NOT all consume one
matrix. FEAT-space models (xgb, mono_gbm, elasticnet, ga2m, nam) use
interp + in-fold PCA; CART uses raw interp only (ignores the embedding);
scorecard + RIPPER use binarized interp. Each wrapper reconstructs its own
space from the columns it receives. The embedding block is ALWAYS the trailing
len(emb_cols) columns and is dropped all-or-nothing, so a wrapper detects the
embedding by width (len(interp) < len(emb) always holds for the CVS matrix,
asserted in build_matrix) - which is what lets remove-and-retrain ablation of
the embedding family work through the generic harness path.
"""
import json

# Faithful mirror of cvs_train_harness.py REGISTRY hyperparameters. Kept as a
# literal table so drift against the training harness is a one-line diff.
XGB_PARAMS = {
    "objective": "binary:logistic", "eval_metric": "auc", "max_depth": 6,
    "colsample_bytree": 0.7, "subsample": 0.8, "min_child_weight": 5,
    "reg_lambda": 2.0, "learning_rate": 0.03, "tree_method": "hist",
}
FEAT_SPACE_MODELS = ("xgb_baseline", "mono_gbm", "elasticnet_glm", "ga2m", "nam", "knn")
INTERP_ONLY_MODELS = ("cart",)
BINARIZED_MODELS = ("scorecard", "ripper")
# MUST match cvs_train_harness.py INTERP_BLOCKS exactly — the ablation family
# map is built from these, so any block here that the training harness did not
# train on would create a phantom (empty or out-of-frozen-set) ablation family.
INTERP_BLOCKS = ("intrinsic_84", "predicted_go", "structure", "homology")
EMB_BLOCK = "esmc6b"


def load_feature_spec(path):
    with open(path) as fh:
        return json.load(fh)


def build_matrix(gold_path, manifest_path, spec):
    """Rebuild the modelling frame from the training-harness contract.

    Returns dict with:
      X            DataFrame: interp columns (in manifest order) then raw emb columns
      y            int label array
      groups       homolog-cluster (or gene fallback) grouping for leakage-safe CV
      group_key    per-row gene id for gene-level aggregation
      feature_blocks {family: [columns]} from the manifest (embedding family =
                     the raw emb columns; the estimators PCA them internally)
      row_ids      stable biological key gene|disease for the split lock
      n_interp, n_emb
    """
    import pandas as pd
    df = pd.read_parquet(gold_path)
    man = pd.read_csv(manifest_path)
    label_col = spec["label_col"]; gene_col = spec["gene_col"]; disease_col = spec["disease_col"]
    interp_cols = [c for c in spec["interp_cols"] if c in df.columns]
    emb_cols = [c for c in man[man.block == spec.get("emb_block", EMB_BLOCK)].column
                if c in df.columns]
    if not emb_cols:
        raise ValueError("no embedding columns found via manifest block "
                         + str(spec.get("emb_block", EMB_BLOCK)))
    if not len(interp_cols) < len(emb_cols):
        raise ValueError("adapter width rule requires n_interp < n_emb; got "
                         f"{len(interp_cols)} interp vs {len(emb_cols)} emb")
    X = pd.concat([df[interp_cols].reset_index(drop=True),
                   df[emb_cols].reset_index(drop=True)], axis=1)
    y = df[label_col].astype(int).values
    gene = df[gene_col].astype(str).values
    cluster_col = "homol_cluster"
    groups = (df[cluster_col].astype(str).values if cluster_col in df.columns else gene)
    # feature families from manifest, restricted to columns actually in X
    fam = {}
    inX = set(X.columns)
    for blk in INTERP_BLOCKS:
        cols = [c for c in man[man.block == blk].column if c in inX]
        if cols:
            fam[blk] = cols
    fam[spec.get("emb_block", EMB_BLOCK)] = emb_cols
    row_ids = [f"{g}|{d}" for g, d in zip(gene, df[disease_col].astype(str).values, strict=False)]
    return {"X": X, "y": y, "groups": groups, "group_key": gene,
            "feature_blocks": fam, "row_ids": row_ids,
            "n_interp": len(interp_cols), "n_emb": len(emb_cols)}


class RegistryEstimator:
    """sklearn-style wrapper around one training-harness model family. Rebuilds
    the family's own feature space from the columns it receives and does the
    in-fold embedding PCA inside fit(). Exposes fit / predict_proba so the
    comparison harness can retrain it per fold and seed."""

    def __init__(self, model_name, n_interp_total, n_emb, seed=0, pca_k=32,
                 scale_before_pca=True, xgb_rounds=400, mono_vec=None):
        self.model_name = model_name
        self.n_interp_total = int(n_interp_total)
        self.n_emb = int(n_emb)
        self.seed = int(seed)
        self.pca_k = int(pca_k)
        self.scale_before_pca = bool(scale_before_pca)
        self.xgb_rounds = int(xgb_rounds)
        self.mono_vec = mono_vec

    # -- feature-space construction -------------------------------------------
    def _split_cols(self, X):
        import numpy as np
        X = np.asarray(X, dtype=float)
        emb_present = X.shape[1] > self.n_interp_total
        if emb_present:
            return X[:, :-self.n_emb], X[:, -self.n_emb:], True
        return X, None, False

    def _emb_to_pcs_fit(self, E):
        import numpy as np
        from sklearn.decomposition import PCA
        from sklearn.preprocessing import StandardScaler
        self.emb_mean_ = np.nanmean(E, axis=0)
        self.emb_mean_ = np.where(np.isfinite(self.emb_mean_), self.emb_mean_, 0.0)
        E = np.where(np.isnan(E), self.emb_mean_, E)
        if self.scale_before_pca:
            self.ss_emb_ = StandardScaler().fit(E); E = self.ss_emb_.transform(E)
        self.pca_ = PCA(self.pca_k, random_state=self.seed).fit(E)
        return self.pca_.transform(E)

    def _emb_to_pcs_apply(self, E):
        import numpy as np
        E = np.where(np.isnan(E), self.emb_mean_, E)
        if self.scale_before_pca:
            E = self.ss_emb_.transform(E)
        return self.pca_.transform(E)

    def _feat_fit(self, X):
        import numpy as np
        Xi, E, emb = self._split_cols(X)
        self._emb_present = emb
        if emb:
            P = self._emb_to_pcs_fit(E)
            return np.hstack([np.nan_to_num(Xi), P]), np.nan_to_num(Xi)
        return np.nan_to_num(Xi), np.nan_to_num(Xi)

    def _feat_apply(self, X):
        import numpy as np
        Xi, E, emb = self._split_cols(X)
        if emb and getattr(self, "_emb_present", False):
            P = self._emb_to_pcs_apply(E)
            return np.hstack([np.nan_to_num(Xi), P]), np.nan_to_num(Xi)
        return np.nan_to_num(Xi), np.nan_to_num(Xi)

    # -- binarization for scorecard / ripper ----------------------------------
    def _binarize_fit(self, Xi):
        import numpy as np
        edges = []
        for j in range(Xi.shape[1]):
            col = Xi[:, j]; u = np.unique(col[np.isfinite(col)])
            edges.append(("bin", None) if len(u) <= 2
                         else ("q", np.quantile(col[np.isfinite(col)], [.25, .5, .75])))
        self._edges = edges

    def _binarize_apply(self, Xi):
        import numpy as np
        cols = []
        for j, (kind, e) in enumerate(self._edges):
            if kind == "bin":
                cols.append((Xi[:, j] > 0.5).astype(np.int8))
            else:
                for thr in e:
                    cols.append((Xi[:, j] > thr).astype(np.int8))
        return np.vstack(cols).T

    # -- fit / predict ---------------------------------------------------------
    def fit(self, X, y):
        import numpy as np
        y = np.asarray(y).astype(int)
        feat, Xi = self._feat_fit(X)
        name = self.model_name
        if name in ("xgb_baseline", "mono_gbm"):
            import xgboost as xgb
            params = dict(XGB_PARAMS, device="cpu")
            if name == "mono_gbm" and self.mono_vec is not None and len(self.mono_vec) == feat.shape[1]:
                params["monotone_constraints"] = "(" + ",".join(map(str, self.mono_vec)) + ")"
            dtr = xgb.QuantileDMatrix(feat, label=y)
            self._booster = xgb.train(params, dtr, num_boost_round=self.xgb_rounds)
        elif name == "elasticnet_glm":
            from sklearn.linear_model import LogisticRegression
            from sklearn.preprocessing import StandardScaler
            self._ss = StandardScaler().fit(feat)
            self._m = LogisticRegression(penalty="elasticnet", l1_ratio=0.5, C=0.1,
                                         solver="saga", max_iter=2000, random_state=self.seed,
                                         class_weight="balanced").fit(self._ss.transform(feat), y)
        elif name == "cart":
            from sklearn.tree import DecisionTreeClassifier
            self._m = DecisionTreeClassifier(max_depth=4, min_samples_leaf=50,
                                             class_weight="balanced", random_state=self.seed,
                                             ccp_alpha=0.001).fit(Xi, y)
        elif name == "scorecard":
            from sklearn.linear_model import LogisticRegression
            self._binarize_fit(Xi); B = self._binarize_apply(Xi)
            g = LogisticRegression(penalty="l1", C=0.05, solver="liblinear",
                                   class_weight="balanced", max_iter=2000).fit(B, y)
            coef = g.coef_[0]; scale = 2.0 / (np.abs(coef[coef != 0]).mean() + 1e-9)
            self._pts = np.round(coef * scale).astype(int)
            self._intercept = g.intercept_[0] * scale
            raw = B @ self._pts + self._intercept
            self._raw_mean, self._raw_std = raw.mean(), raw.std() + 1e-9
        elif name == "ripper":
            import pandas as pd
            import wittgenstein as lw
            self._binarize_fit(Xi); B = self._binarize_apply(Xi)
            self._bn = [f"b{i}" for i in range(B.shape[1])]
            self._rip = lw.RIPPER(random_state=self.seed, max_rules=12)
            self._rip.fit(pd.DataFrame(B, columns=self._bn), y)
        elif name == "ga2m":
            from interpret.glassbox import ExplainableBoostingClassifier
            self._m = ExplainableBoostingClassifier(random_state=self.seed,
                                                    interactions=10, n_jobs=-1).fit(feat, y)
        elif name == "nam":
            self._fit_nam(feat, y)
        elif name == "knn":
            # Mirror the training harness's KNN family: standardize the
            # interp+in-fold-PCA feature space, then k-NN (cuML on GPU when
            # available, sklearn fallback). Instance-based — "fit" stores the
            # standardized train rows + labels; refit per fold/seed is honest
            # because the neighbour set is rebuilt from the train fold only.
            import numpy as np
            from sklearn.preprocessing import StandardScaler
            self._ss = StandardScaler().fit(feat)
            Ftr = self._ss.transform(feat)
            self._knn_k = 50
            try:
                from cuml.neighbors import KNeighborsClassifier as cuKNN
                self._m = cuKNN(n_neighbors=self._knn_k)
                self._m.fit(Ftr.astype(np.float32), y.astype(np.float32))
                self._knn_backend = "cuml"
            except Exception:
                from sklearn.neighbors import KNeighborsClassifier as skKNN
                self._m = skKNN(n_neighbors=self._knn_k, weights="distance")
                self._m.fit(Ftr, y)
                self._knn_backend = "sklearn"
        else:
            raise ValueError(f"unknown model {name}")
        return self

    def _fit_nam(self, feat, y):
        import torch
        import torch.nn as nn_t
        from sklearn.preprocessing import StandardScaler
        torch.manual_seed(self.seed)
        self._ss = StandardScaler().fit(feat)
        Xt = torch.tensor(self._ss.transform(feat), dtype=torch.float32)
        yt = torch.tensor(y, dtype=torch.float32); F = Xt.shape[1]

        class FeatNet(nn_t.Module):
            def __init__(s):
                super().__init__()
                s.n = nn_t.Sequential(nn_t.Linear(1, 32), nn_t.ReLU(),
                                      nn_t.Linear(32, 16), nn_t.ReLU(), nn_t.Linear(16, 1))

            def forward(s, x):
                return s.n(x)

        class NAM(nn_t.Module):
            def __init__(s):
                super().__init__()
                s.fs = nn_t.ModuleList([FeatNet() for _ in range(F)])
                s.b = nn_t.Parameter(torch.zeros(1))

            def forward(s, x):
                return sum(s.fs[i](x[:, i:i + 1]) for i in range(F)).squeeze(1) + s.b

        self._nam = NAM()
        opt = torch.optim.Adam(self._nam.parameters(), lr=1e-3, weight_decay=1e-5)
        pw = torch.tensor([(y == 0).sum() / max((y == 1).sum(), 1)], dtype=torch.float32)
        lossf = nn_t.BCEWithLogitsLoss(pos_weight=pw)
        for _ in range(60):
            opt.zero_grad(); loss = lossf(self._nam(Xt), yt); loss.backward(); opt.step()

    def predict_proba(self, X):
        import numpy as np
        feat, Xi = self._feat_apply(X)
        name = self.model_name
        if name in ("xgb_baseline", "mono_gbm"):
            import xgboost as xgb
            # DMatrix (not QuantileDMatrix) for inference: QuantileDMatrix is a
            # TRAINING structure (quantizes to build histogram cut points);
            # the plain DMatrix is the stateless, device-agnostic inference
            # equivalent and is byte-identical to the ref=dtr path on CPU.
            p = self._booster.predict(xgb.DMatrix(feat))
        elif name == "elasticnet_glm":
            p = self._m.predict_proba(self._ss.transform(feat))[:, 1]
        elif name == "cart":
            p = self._m.predict_proba(Xi)[:, 1]
        elif name == "scorecard":
            B = self._binarize_apply(Xi)
            raw = B @ self._pts + self._intercept
            p = 1.0 / (1.0 + np.exp(-(raw - self._raw_mean) / self._raw_std))
        elif name == "ripper":
            import pandas as pd
            B = self._binarize_apply(Xi)
            pr = np.asarray(self._rip.predict_proba(pd.DataFrame(B, columns=self._bn)))
            p = pr[:, 1] if pr.ndim == 2 else pr
        elif name == "ga2m":
            p = self._m.predict_proba(feat)[:, 1]
        elif name == "nam":
            import torch
            with torch.no_grad():
                p = torch.sigmoid(self._nam(torch.tensor(
                    self._ss.transform(feat), dtype=torch.float32))).numpy()
        elif name == "knn":
            import numpy as np
            F = self._ss.transform(feat)
            if self._knn_backend == "cuml":
                pr = self._m.predict_proba(F.astype(np.float32))
                pr = np.asarray(pr.to_numpy() if hasattr(pr, "to_numpy") else pr)
            else:
                pr = self._m.predict_proba(F)
            p = pr[:, 1] if pr.ndim == 2 and pr.shape[1] > 1 else np.asarray(pr).ravel()
        else:
            raise ValueError(f"unknown model {name}")
        p = np.clip(np.asarray(p, dtype=float), 1e-7, 1 - 1e-7)
        return np.column_stack([1.0 - p, p])


def make_factories(spec, n_interp, n_emb, models=None, mono_vec=None,
                   xgb_rounds=None):
    """Return {name: factory(seed)->RegistryEstimator} for run_comparison. Only
    the families named in `models` (default spec['models']) are included."""
    names = list(models) if models is not None else list(spec.get("models", []))
    rounds = int(xgb_rounds if xgb_rounds is not None else spec.get("xgb_rounds", 400))
    pca_k = int(spec.get("pca_k", 32)); scale = bool(spec.get("scale_before_pca", True))

    def _mk(name):
        def factory(seed):
            return RegistryEstimator(name, n_interp_total=n_interp, n_emb=n_emb,
                                     seed=seed, pca_k=pca_k, scale_before_pca=scale,
                                     xgb_rounds=rounds, mono_vec=mono_vec)
        return factory
    return {name: _mk(name) for name in names}


def build_split_lock_from_seqclust(gold_path, spec, harness, n_splits=None):
    """Turn the training harness's frozen fold_seqclust column into a
    fingerprinted split lock both harnesses can enforce. `harness` is the loaded
    kernel.py module (for make_split_lock/p_split_fingerprint)."""
    import pandas as pd
    df = pd.read_parquet(gold_path)
    gene = df[spec["gene_col"]].astype(str).values
    dis = df[spec["disease_col"]].astype(str).values
    row_ids = [f"{g}|{d}" for g, d in zip(gene, dis, strict=False)]
    seqfold = df["fold_seqclust"].astype(int).values
    ns = int(n_splits if n_splits is not None else len(set(seqfold.tolist())))
    # groups = the seqclust fold id itself, so the lock reproduces fold_seqclust
    lock = harness.make_split_lock(row_ids, groups=seqfold, n_splits=ns, seed=0)
    # overwrite the derived folds with the ACTUAL frozen assignment (authoritative)
    lock["row_fold"] = {r: int(f) for r, f in zip(row_ids, seqfold, strict=False)}
    lock["fingerprint"] = harness.p_split_fingerprint(row_ids, seqfold, ns, 0)
    lock["source"] = "fold_seqclust (frozen, homolog-safe)"
    return lock, row_ids


def ingest_oof(oof_path, metrics_csv_path, harness, split=None,
               gene_col="ensembl_gene_id", label_col="label_consensus", k=20):
    """Mode A: build the cross-model comparison table from the training
    harness's OWN outputs, no retraining. Uses oof columns oof_<split>_<model>
    and the permuted-label columns permoof_<model> for the signal gap."""
    import numpy as np
    import pandas as pd
    oof = pd.read_parquet(oof_path)
    metrics = pd.read_csv(metrics_csv_path)
    splits = ([split] if split else sorted({c.split("_")[1]
              for c in oof.columns if c.startswith("oof_")}))
    gene = oof[gene_col].astype(str).values
    y = oof[label_col].astype(int).values

    def gene_metrics(pred, lab):
        # Canonical gene roll-up: MAJORITY VOTE on the label (mean > 0.5), mean of
        # scores — matches cvs_train_harness.gene_roll exactly. (Earlier `max` on the
        # label over-counted gene positives, inflating base rate and gene-PR-AUC.)
        d = pd.DataFrame({"g": gene, "y": lab, "s": pred}).dropna().groupby("g").mean()
        gy = (d["y"] > 0.5).astype(int).values
        return harness.p_metric_block(gy, d["s"].values, k)

    rows = []
    for sp in splits:
        for col in [c for c in oof.columns if c.startswith(f"oof_{sp}_")]:
            model = col[len(f"oof_{sp}_"):]
            pred = oof[col].values
            if np.isnan(pred).all():
                continue
            gm = gene_metrics(pred, y)
            r = {"split": sp, "model": model,
                 "gene_roc_auc": gm.get("roc_auc"), "gene_pr_auc": gm.get("pr_auc"),
                 f"gene_p_at_{k}": gm.get(f"p_at_{k}")}
            permcol = f"permoof_{model}"
            if sp == spec_headline(oof) and permcol in oof.columns and not oof[permcol].isna().all():
                pm = gene_metrics(oof[permcol].values, y)
                r["perm_gene_roc_auc"] = pm.get("roc_auc")
                r["signal_gap"] = (gm.get("roc_auc") or np.nan) - (pm.get("roc_auc") or np.nan)
            rows.append(r)
    return pd.DataFrame(rows)


def spec_headline(oof):
    """Headline split name if permoof columns exist (they are only produced for
    the headline split); else None."""
    return "seqclust" if any(c.startswith("permoof_") for c in oof.columns) else None
