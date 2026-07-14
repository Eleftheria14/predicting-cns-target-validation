#!/usr/bin/env python3
"""8-family glass-box sweep for CNS CVS — full accuracy-vs-transparency bracket.
Shares the frozen fold_seqclust GroupKFold + in-fold PCA-32 of ESMC-6B with the
baseline benchmark, so every family is directly comparable and leakage-safe
(NVIDIA tabular playbook: match CV to test structure; diverse baselines in one pass).
Families are compared as STANDALONE glass-boxes — deliberately NOT ensembled,
because stacking would destroy the transparency we are measuring.

Family map (2 canonical libs fail to build on aarch64 -> documented proxies):
  1 Monotonic GBM       XGBoost monotone_constraints (data-supported dirs)   NATIVE-GPU
  2 Sparse decision tree GOSDT fails -> shallow CART + ccp pruning (PROXY)
  3 Integer scorecard    FasterRisk fails -> rounded elastic-net scorecard (PROXY)
  4 Elastic-net GLM      sklearn LogisticRegression(elasticnet)              NATIVE
  5 Rule set             wittgenstein RIPPER on binarized readable feats     NATIVE
  6 GAM + interactions   EBM(interactions=10) = GA2M                         NATIVE
  7 Neural additive      NAM (CPU torch, small data)                         NATIVE
  8 Learned prototype    reuse benchmarked ESMC-6B KNN (nearest-neighbour)   (from prior job)
Emits OOF predictions + model descriptors; gene-level metrics computed locally.
"""
import json
import time
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
t0 = time.time()
OUT = "/work"
df  = pd.read_parquet("gold.parquet")
man = pd.read_csv("manifest.csv")
dirs_map = json.load(open("monotone_dirs.json"))

INTERP_BLOCKS = ["intrinsic_84","predicted_go","structure","homology"]
interp_cols = man[man.block.isin(INTERP_BLOCKS)].column.tolist()
emb_cols    = man[man.block=="esmc6b"].column.tolist()
y    = df["label_consensus"].astype(int).values
fold = df["fold_seqclust"].astype(int).values
efo  = df["efo_id"].values
n    = len(df)
PCA_K, NROUND, FOLDS = 32, 400, sorted(np.unique(fold))
pc_names   = [f"emb_pc{i}" for i in range(PCA_K)]
feat_names = interp_cols + pc_names
mono_vec   = [int(dirs_map.get(f,0)) for f in feat_names]
print(f"[t+{time.time()-t0:.0f}s] {df.shape}; {len(interp_cols)} interp + {PCA_K} PC; "
      f"mono +{sum(v==1 for v in mono_vec)}/-{sum(v==-1 for v in mono_vec)}", flush=True)

import xgboost as xgb
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier


def emb_impute(tr, te):
    E = df[emb_cols].values.astype(np.float32)
    cm = np.nanmean(E[tr],axis=0); cm = np.where(np.isfinite(cm),cm,0.0)
    Ef = np.where(np.isnan(E),cm,E)
    return Ef[tr], Ef[te]

# binarizer for rule/scorecard families: quantile bins on interp feats (train-fit)
def binarize_fit(Xtr_interp):
    edges=[]
    for j in range(Xtr_interp.shape[1]):
        col=Xtr_interp[:,j]; u=np.unique(col[np.isfinite(col)])
        if len(u)<=2: edges.append(("bin",u)); # already ~binary
        else: edges.append(("q", np.quantile(col[np.isfinite(col)],[.25,.5,.75])))
    return edges
def binarize_apply(Xi, edges):
    cols=[]; names=[]
    for j,(kind,e) in enumerate(edges):
        nm=interp_cols[j]
        if kind=="bin":
            cols.append((Xi[:,j]>0.5).astype(np.int8)); names.append(f"{nm}>0.5")
        else:
            for q,thr in zip([25,50,75],e, strict=False):
                cols.append((Xi[:,j]>thr).astype(np.int8)); names.append(f"{nm}>q{q}")
    return np.vstack(cols).T, names

oof = {m:np.zeros(n) for m in ["mono_gbm","sparse_tree","scorecard","glm","ripper","ga2m","nam"]}
status = {m:"ok" for m in oof}
scorecard_repr = None; tree_repr=None; ripper_repr=None

def fold_matrices(k):
    tr, te = fold!=k, fold==k
    Etr_f,Ete_f = emb_impute(tr,te)
    pca=PCA(PCA_K,random_state=0).fit(Etr_f)
    Xtr=np.hstack([df.loc[tr,interp_cols].values, pca.transform(Etr_f)]).astype(np.float32)
    Xte=np.hstack([df.loc[te,interp_cols].values, pca.transform(Ete_f)]).astype(np.float32)
    return tr,te,Xtr,Xte

for k in FOLDS:
    tr,te,Xtr,Xte = fold_matrices(k)
    Xtr=np.nan_to_num(Xtr); Xte=np.nan_to_num(Xte)   # interp feats carry NaN; PCs are clean
    ytr=y[tr]
    # ---- 1. Monotonic GBM ----
    try:
        P=dict(objective="binary:logistic",eval_metric="auc",max_depth=6,colsample_bytree=0.7,
               subsample=0.8,min_child_weight=5,reg_lambda=2.0,learning_rate=0.03,
               tree_method="hist",device="cuda",monotone_constraints="("+",".join(map(str,mono_vec))+")")
        dtr=xgb.QuantileDMatrix(Xtr,label=ytr,feature_names=feat_names)
        dte=xgb.QuantileDMatrix(Xte,ref=dtr,feature_names=feat_names)
        oof["mono_gbm"][te]=xgb.train(P,dtr,num_boost_round=NROUND).predict(dte)
    except Exception as e: status["mono_gbm"]=f"fail:{e}"
    # ---- 4. Elastic-net GLM (standardized) ----
    try:
        sc=StandardScaler().fit(Xtr)
        glm=LogisticRegression(penalty="elasticnet",l1_ratio=0.5,C=0.1,solver="saga",
                               max_iter=2000,class_weight="balanced").fit(sc.transform(Xtr),ytr)
        oof["glm"][te]=glm.predict_proba(sc.transform(Xte))[:,1]
    except Exception as e: status["glm"]=f"fail:{e}"
    # ---- 2. Sparse decision tree (GOSDT proxy: pruned CART, readable feats only) ----
    try:
        Xi_tr=df.loc[tr,interp_cols].values; Xi_te=df.loc[te,interp_cols].values
        ct=DecisionTreeClassifier(max_depth=4,min_samples_leaf=50,class_weight="balanced",
                                  random_state=0,ccp_alpha=0.001).fit(np.nan_to_num(Xi_tr),ytr)
        oof["sparse_tree"][te]=ct.predict_proba(np.nan_to_num(Xi_te))[:,1]
        if k==FOLDS[0]:
            from sklearn.tree import export_text
            tree_repr=export_text(ct,feature_names=list(interp_cols),max_depth=4)[:4000]
    except Exception as e: status["sparse_tree"]=f"fail:{e}"
    # ---- 3 & 5: binarized-feature families ----
    Xi_tr=np.nan_to_num(df.loc[tr,interp_cols].values); Xi_te=np.nan_to_num(df.loc[te,interp_cols].values)
    edges=binarize_fit(Xi_tr)
    Btr,bnames=binarize_apply(Xi_tr,edges); Bte,_=binarize_apply(Xi_te,edges)
    # 3. Integer scorecard = rounded elastic-net logistic on binary feats
    try:
        scg=LogisticRegression(penalty="l1",C=0.05,solver="liblinear",class_weight="balanced",
                               max_iter=2000).fit(Btr,ytr)
        coef=scg.coef_[0]; scale=2.0/ (np.abs(coef[coef!=0]).mean()+1e-9)
        pts=np.round(coef*scale).astype(int)   # integer points
        raw=Bte@pts + scg.intercept_[0]*scale
        oof["scorecard"][te]=1/(1+np.exp(-(raw-raw.mean())/ (raw.std()+1e-9)))
        if k==FOLDS[0]:
            top=sorted(zip(bnames,pts, strict=False),key=lambda x:-abs(x[1]))
            scorecard_repr=[(nn,int(pp)) for nn,pp in top if pp!=0][:25]
    except Exception as e: status["scorecard"]=f"fail:{e}"
    # 5. RIPPER rule set
    if status["ripper"]=="ok":
        try:
            import wittgenstein as lw
            rip=lw.RIPPER(random_state=0,max_rules=12)
            rip.fit(pd.DataFrame(Btr,columns=bnames), ytr)
            proba=rip.predict_proba(pd.DataFrame(Bte,columns=bnames))
            oof["ripper"][te]=np.asarray(proba)[:,1] if np.ndim(proba)==2 else np.asarray(proba)
            if k==FOLDS[0]: ripper_repr=str(rip.ruleset_)[:3000]
        except Exception as e: status["ripper"]=f"fail:{type(e).__name__}:{e}"
    # ---- 6. GA2M (EBM with interactions) ----
    if status["ga2m"]=="ok":
        try:
            from interpret.glassbox import ExplainableBoostingClassifier
            ga=ExplainableBoostingClassifier(random_state=0,interactions=10,n_jobs=-1)
            ga.fit(pd.DataFrame(Xtr,columns=feat_names),ytr)
            oof["ga2m"][te]=ga.predict_proba(pd.DataFrame(Xte,columns=feat_names))[:,1]
        except Exception as e: status["ga2m"]=f"fail:{type(e).__name__}:{e}"
    # ---- 7. NAM (neural additive, CPU) ----
    if status["nam"]=="ok":
        try:
            import torch
            import torch.nn as nn_t
            torch.manual_seed(0)
            sc2=StandardScaler().fit(Xtr); Xt=torch.tensor(sc2.transform(Xtr),dtype=torch.float32)
            Xv=torch.tensor(sc2.transform(Xte),dtype=torch.float32); yt=torch.tensor(ytr,dtype=torch.float32)
            F=Xt.shape[1]
            class FeatNet(nn_t.Module):
                def __init__(s): super().__init__(); s.n=nn_t.Sequential(nn_t.Linear(1,32),nn_t.ReLU(),nn_t.Linear(32,16),nn_t.ReLU(),nn_t.Linear(16,1))
                def forward(s,x): return s.n(x)
            class NAM(nn_t.Module):
                def __init__(s): super().__init__(); s.fs=nn_t.ModuleList([FeatNet() for _ in range(F)]); s.b=nn_t.Parameter(torch.zeros(1))
                def forward(s,x): return sum(s.fs[i](x[:,i:i+1]) for i in range(F)).squeeze(1)+s.b
            m=NAM(); opt=torch.optim.Adam(m.parameters(),lr=1e-3,weight_decay=1e-5)
            pw=torch.tensor([(ytr==0).sum()/max((ytr==1).sum(),1)],dtype=torch.float32)
            lossf=nn_t.BCEWithLogitsLoss(pos_weight=pw)
            for ep in range(60):
                opt.zero_grad(); out=m(Xt); loss=lossf(out,yt); loss.backward(); opt.step()
            with torch.no_grad(): oof["nam"][te]=torch.sigmoid(m(Xv)).numpy()
        except Exception as e: status["nam"]=f"fail:{type(e).__name__}:{e}"
    print(f"[t+{time.time()-t0:.0f}s] fold {k}: "+", ".join(f"{m}={'ok' if status[m]=='ok' else 'X'}" for m in oof), flush=True)

# ---- write ----
ids=df[["ensembl_gene_id","efo_id","gene_symbol","label_consensus","fold_seqclust"]].copy()
ids["gidx"]=np.arange(n)
for m in oof: ids[f"oof_{m}"]=oof[m]
ids.to_parquet(f"{OUT}/glassbox_oof.parquet")
json.dump({"status":status,"scorecard":scorecard_repr,"tree":tree_repr,"ripper":ripper_repr,
           "n":int(n),"mono_nonzero":int(sum(v!=0 for v in mono_vec))},
          open(f"{OUT}/glassbox_models.json","w"),indent=2)
print(f"[t+{time.time()-t0:.0f}s] DONE status={status}",flush=True)
