#!/usr/bin/env python3
"""
reproduce_headline.py -- reproduce the headline CNS-CVS result from the gold dataset.

Trains the deployed XGBoost classifier under the frozen, homolog-safe cross-validation
split (`fold_seqclust`) and reports the gene-level ROC-AUC. Expected: ~0.856.

The single most important anti-leakage step is reproduced faithfully: the 2,560-dim
protein-language-model embedding is reduced to 32 principal components fit INSIDE each
training fold (StandardScaler + PCA on train only, applied to test), so no held-out
information enters the reduction. The 164 human-readable features are used as-is.

Runs on CPU in a few minutes. Requires: pandas, numpy, scikit-learn, xgboost.

Usage:
    python reproduce_headline.py --data data/gold/cns_cvs_gold_modeling_dataset.parquet \
                                 --manifest data/gold/cns_cvs_gold_modeling_dataset_manifest.csv
"""
import argparse

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.decomposition import PCA
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

LABEL_COL, GENE_COL, SEQCLUST_COL = "label_consensus", "ensembl_gene_id", "fold_seqclust"
EMB_BLOCK = "esmc6b"
INTERP_BLOCKS = ["intrinsic_84", "predicted_go", "structure", "homology"]
PCA_K, XGB_ROUNDS, SEED = 32, 400, 0

XGB_PARAMS = dict(objective="binary:logistic", eval_metric="auc", max_depth=6,
                  colsample_bytree=0.7, subsample=0.8, min_child_weight=5,
                  reg_lambda=2.0, learning_rate=0.03, tree_method="hist")

def build_fold(df, emb_cols, interp_cols, tr, te):
    """In-fold embedding reduction: mean-impute + scale + PCA-32 fit on TRAIN only."""
    E = df[emb_cols].values.astype(np.float32)
    cm = np.nanmean(E[tr], axis=0); cm = np.where(np.isfinite(cm), cm, 0.0)
    E = np.where(np.isnan(E), cm, E)
    ss = StandardScaler().fit(E[tr])
    Etr, Ete = ss.transform(E[tr]), ss.transform(E[te])
    pca = PCA(PCA_K, random_state=SEED).fit(Etr)
    Ptr, Pte = pca.transform(Etr), pca.transform(Ete)
    I = np.nan_to_num(df[interp_cols].values.astype(np.float32))
    Xtr = np.hstack([Ptr, I[tr]]); Xte = np.hstack([Pte, I[te]])
    return Xtr, Xte

def gene_roll(gene, y, p):
    """Aggregate pair predictions to the gene level by mean, then score."""
    d = pd.DataFrame({"g": gene, "y": y, "p": p}).dropna().groupby("g").mean()
    return (d.y > 0.5).astype(int).values, d.p.values

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/gold/cns_cvs_gold_modeling_dataset.parquet")
    ap.add_argument("--manifest", default="data/gold/cns_cvs_gold_modeling_dataset_manifest.csv")
    args = ap.parse_args()

    df = pd.read_parquet(args.data)
    man = pd.read_csv(args.manifest)
    emb_cols = man[man.block == EMB_BLOCK].column.tolist()
    interp_cols = man[man.block.isin(INTERP_BLOCKS)].column.tolist()
    y = df[LABEL_COL].values.astype(float)
    gene = df[GENE_COL].values
    folds = df[SEQCLUST_COL].astype(int).values

    print(f"loaded {df.shape[0]} pairs x {df.shape[1]} cols "
          f"({len(emb_cols)} embedding dims -> PCA-{PCA_K}, {len(interp_cols)} readable features)")

    oof = np.full(len(df), np.nan)
    for k in sorted(np.unique(folds)):
        tr, te = folds != k, folds == k
        Xtr, Xte = build_fold(df, emb_cols, interp_cols, tr, te)
        dtr = xgb.QuantileDMatrix(Xtr, label=y[tr])
        dte = xgb.QuantileDMatrix(Xte, ref=dtr)
        bst = xgb.train(XGB_PARAMS, dtr, num_boost_round=XGB_ROUNDS)
        oof[te] = bst.predict(dte)
        print(f"  fold {k}: n_test={te.sum()}")

    gy, gp = gene_roll(gene, y, oof)
    roc = roc_auc_score(gy, gp)
    pr = average_precision_score(y, oof)
    print(f"\nHomolog-safe (fold_seqclust) gene-level ROC-AUC : {roc:.4f}   (reference 0.856)")
    print(f"Pair-level PR-AUC                              : {pr:.4f}   (reference 0.576)")

if __name__ == "__main__":
    main()
