#!/usr/bin/env python3
"""
cvs_sweep_job.py — hyperparameter SENSITIVITY sweeps for the CVS model families.

Deotte-style (nvidia-xgboost skill §4): build a trustworthy in-split CV, then sweep the handful of
knobs that matter and record OOF score vs knob. Runs the SAME 4 splits as the training harness
(seqclust/gene/disease/random) so each sweep curve has one line per generalization axis. Feature
matrix and split/leakage logic are imported from cvs_train_harness.py (single source of truth).

Emits sweep_results.csv (tidy: model, knob, value, split, gene_pr, gene_roc, n_scored) for
render_report.py to turn into Figures 2..N. Deterministic: SEED fixed, CPU-pinned, sorted iteration.

Run inside the harness container:  python /work/cvs_sweep_job.py
"""
import json
import os
import sys
import time

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

OUT = os.environ.get("OUT", "/work")
# Optional family filter: SWEEP_ONLY="ga2m,nam,knn" scopes BOTH the grid and random search to
# those families (e.g. to top up families a prior run's wall-clock ceiling cut off). When set,
# results are APPENDED to any existing sweep_results.csv / search_results.csv rather than
# overwriting, so a partial prior run is completed, not discarded. Empty = all families.
SWEEP_ONLY = [f for f in os.environ.get("SWEEP_ONLY", "").split(",") if f.strip()]
_t0 = time.time()
def log(m): print(f"[t+{time.time()-_t0:5.0f}s] {m}", flush=True)

# ---- reuse the harness as a LIBRARY: build_fold, the split registry, binarizers, in-fold PCA
#      and the exact feature spec are the single source of truth. The harness is import-clean
#      (no side effects at import); load_data() populates its module-level arrays. This replaces
#      the old string-split-on-a-magic-comment hack, which coupled the two files by a literal
#      line and broke silently whenever that line moved. -------------------------------------
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cvs_run_logging as RL
import cvs_train_harness as H

# Provenance + fail-loud GPU guard, shared with the training harness. The sweep's GPU tier
# (xgb/mono via device=cuda, knn via cuML) is the whole point of running here, so abort rather
# than silently sweep on CPU. run_meta.json is written next to sweep_results/search_results.
_lg = RL.RunLogger(OUT, tag="cvs_sweep_job")
_pf = RL.gpu_preflight(logger=_lg, require=(H.DEVICE == "cuda"))
H.load_data()
SEED = H.SEED
log(f"harness loaded: {len(H.MODELS)} models, splits={H.SPLITS}, PCA_K={H.PCA_K}")

# ---- per-family knob grids (the load-bearing knobs; kept small & fixed for determinism) ----
# GPU tier (xgb/mono via device=cuda; knn via cuML on-device) gets FULL grids;
# CPU-bound families (sklearn / interpret / wittgenstein) get trimmed 3-value grids
# so the whole sweep fits in one wall-clock window. Tier choice recorded per user.
GRIDS = {
 # --- GPU tier: full resolution ---
 "xgb_baseline":   [("max_depth",[3,4,6,8,10,12]), ("colsample_bytree",[0.3,0.5,0.7,0.9])],
 "mono_gbm":       [("max_depth",[3,4,6,8,10,12]), ("colsample_bytree",[0.3,0.5,0.7,0.9])],
 "knn":            [("k",[5,10,20,30,50,75,100])],
 # --- CPU tier: trimmed to 3 values per knob ---
 "elasticnet_glm": [("C",[0.01,0.1,1.0]), ("l1_ratio",[0.0,0.5,1.0])],
 "cart":           [("max_depth",[3,4,8]), ("ccp_alpha",[0.0,0.001,0.01])],
 "scorecard":      [("C",[0.03,0.1,1.0])],                 # L1 strength -> # nonzero points
 "ripper":         [("max_rules",[4,12,24])],               # rule-list length
 "ga2m":           [("max_bins",[64,128,256]), ("interactions",[0,8,16])],
 "nam":            [("hidden",[8,16,32]), ("weight_decay",[1e-5,1e-4,1e-3])],
}
# tier tag stamped into each result row for the report/caption
GPU_TIER = {"xgb_baseline","mono_gbm","knn"}

# ---- parametrized trainers: return OOF proba on the test fold given a knob override ----
def train_one(model, knob, val, Xtr, ytr, Xte, ctx):
    import xgboost as xgb
    from sklearn.preprocessing import StandardScaler
    if model in ("xgb_baseline","mono_gbm"):
        # device follows the harness (cuda on the GB10) — the GBT families are the GPU tier.
        p = H._xgb_params(mono=(model=="mono_gbm")); p["device"]=H.DEVICE; p[knob]=val
        dtr=xgb.QuantileDMatrix(Xtr,label=ytr); dte=xgb.QuantileDMatrix(Xte,ref=dtr)
        spw=(ytr==0).sum()/max((ytr==1).sum(),1); p["scale_pos_weight"]=spw
        bst=xgb.train(p,dtr,num_boost_round=H.XGB_ROUNDS); return bst.predict(dte)
    if model=="elasticnet_glm":
        from sklearn.linear_model import LogisticRegression
        kw=dict(penalty="elasticnet",l1_ratio=0.5,C=0.1,solver="saga",max_iter=2000,class_weight="balanced",random_state=SEED)
        kw[knob]=val; ss=StandardScaler().fit(Xtr)
        g=LogisticRegression(**kw).fit(ss.transform(Xtr),ytr); return g.predict_proba(ss.transform(Xte))[:,1]
    if model=="cart":
        from sklearn.tree import DecisionTreeClassifier
        kw=dict(max_depth=6,ccp_alpha=0.001,class_weight="balanced",random_state=SEED); kw[knob]=val
        t=DecisionTreeClassifier(**kw).fit(ctx["Xi_tr"],ytr); return t.predict_proba(ctx["Xi_te"])[:,1]
    if model=="scorecard":
        from sklearn.linear_model import LogisticRegression
        Btr,Bte,bn = ctx["Btr"],ctx["Bte"],ctx["bnames"]
        g=LogisticRegression(penalty="l1",C=val,solver="liblinear",class_weight="balanced",max_iter=2000).fit(Btr,ytr)
        coef=g.coef_[0]; nz=np.abs(coef[coef!=0]); scale=2.0/(nz.mean()+1e-9) if nz.size else 1.0
        pts=np.round(coef*scale).astype(int); raw=Bte@pts + g.intercept_[0]*scale
        return 1/(1+np.exp(-(raw-raw.mean())/(raw.std()+1e-9)))
    if model=="ripper":
        import wittgenstein as lw
        Btr,Bte,bn = ctx["Btr"],ctx["Bte"],ctx["bnames"]
        r=lw.RIPPER(random_state=SEED,max_rules=val)
        r.fit(pd.DataFrame(Btr,columns=bn),ytr)
        pr=r.predict_proba(pd.DataFrame(Bte,columns=bn))
        return pr[:,1] if getattr(pr,"ndim",1)==2 else np.asarray(pr,float)
    if model=="ga2m":
        from interpret.glassbox import ExplainableBoostingClassifier as EBC
        kw=dict(random_state=SEED); kw[knob]=val
        e=EBC(**kw).fit(Xtr,ytr); return e.predict_proba(Xte)[:,1]
    if model=="nam":
        return _train_nam_sweep(Xtr,ytr,Xte,knob,val)
    if model=="knn":
        from sklearn.neighbors import KNeighborsClassifier as SKNN
        ss=StandardScaler().fit(Xtr)
        try:
            from cuml.neighbors import KNeighborsClassifier as GKNN
            m=GKNN(n_neighbors=val); m.fit(ss.transform(Xtr).astype(np.float32),ytr.astype(np.float32))
            pr=m.predict_proba(ss.transform(Xte).astype(np.float32)); pr=pr.to_numpy() if hasattr(pr,"to_numpy") else np.asarray(pr)
            return pr[:,1] if pr.ndim==2 else pr
        except Exception:
            m=SKNN(n_neighbors=val).fit(ss.transform(Xtr),ytr); return m.predict_proba(ss.transform(Xte))[:,1]
    raise ValueError(model)

def _train_nam_sweep(Xtr,ytr,Xte,knob,val):
    import torch
    import torch.nn as nn
    torch.manual_seed(SEED)
    from sklearn.preprocessing import StandardScaler
    ss=StandardScaler().fit(Xtr); Xt=torch.tensor(ss.transform(Xtr),dtype=torch.float32); Xv=torch.tensor(ss.transform(Xte),dtype=torch.float32)
    yt=torch.tensor(ytr,dtype=torch.float32); F=Xt.shape[1]
    hid = val if knob=="hidden" else 16; wd = val if knob=="weight_decay" else 1e-5
    class FN(nn.Module):
        def __init__(s): super().__init__(); s.n=nn.Sequential(nn.Linear(1,hid),nn.ReLU(),nn.Linear(hid,1))
        def forward(s,x): return s.n(x)
    class NAM(nn.Module):
        def __init__(s): super().__init__(); s.fs=nn.ModuleList([FN() for _ in range(F)]); s.b=nn.Parameter(torch.zeros(1))
        def forward(s,x): return sum(s.fs[i](x[:,i:i+1]) for i in range(F)).squeeze(1)+s.b
    m=NAM(); opt=torch.optim.Adam(m.parameters(),lr=1e-3,weight_decay=wd)
    pw=torch.tensor([(ytr==0).sum()/max((ytr==1).sum(),1)],dtype=torch.float32)
    lf=nn.BCEWithLogitsLoss(pos_weight=pw)
    for _ in range(60): opt.zero_grad(); lf(m(Xt),yt).backward(); opt.step()
    with torch.no_grad(): return torch.sigmoid(m(Xv)).numpy()

# ---- gene-level metrics (same convention as the harness) ----
def gene_metrics(p, te_mask):
    d=pd.DataFrame({"g":H.gene[te_mask],"y":H.y[te_mask],"p":p})
    gg=d.groupby("g").agg(y=("y","mean"),p=("p","mean")).reset_index()
    gg=gg[(gg.y==0)|(gg.y==1)]
    if gg.y.nunique()<2: return (np.nan,np.nan,len(gg))
    return (average_precision_score(gg.y,gg.p), roc_auc_score(gg.y,gg.p), len(gg))

# families this run touches (all, unless SWEEP_ONLY scopes a top-up)
FAMS = [m for m in H.MODELS if (not SWEEP_ONLY or m in SWEEP_ONLY)]
if SWEEP_ONLY:
    log(f"SWEEP_ONLY set — scoping to {FAMS}; results APPENDED to any existing CSVs")

def _atomic_csv(rows, path):
    """Write via temp + os.replace so an interrupt never corrupts the CSV; the last checkpoint
    survives intact. When SWEEP_ONLY scopes a top-up, MERGE with the existing file: keep rows for
    families NOT in this run, replace rows for families that ARE. Otherwise overwrite."""
    df_new = pd.DataFrame(rows)
    if SWEEP_ONLY and os.path.exists(path):
        prev = pd.read_csv(path)
        prev = prev[~prev["model"].isin(FAMS)]                 # drop families we're re-running
        df_new = pd.concat([prev, df_new], ignore_index=True)
    tmp=f"{path}.tmp"; df_new.to_csv(tmp,index=False); os.replace(tmp,path)

# SWEEP_STAGE lets a top-up run only the phase it needs: "grid", "search", or "both" (default).
SWEEP_STAGE = os.environ.get("SWEEP_STAGE", "both")

# ---- sweep loop: model x knob x value x split, in-split CV OOF ----
records=[]
for model in (FAMS if SWEEP_STAGE in ("grid","both") else []):
    if model not in GRIDS: continue
    for knob,vals in GRIDS[model]:
        for split in H.SPLITS:
            folds=H.SPLIT_REG[split](H.df)
            for val in vals:
                oof=np.full(H.n,np.nan)
                ok=True
                for tr,te in folds:
                    try:
                        Xtr,Xte=H.build_fold(tr,te)
                        ytr=H.y[tr]
                        Xi_tr=np.nan_to_num(H.df.loc[tr,H.interp_cols].values)
                        Xi_te=np.nan_to_num(H.df.loc[te,H.interp_cols].values)
                        ctx={"Xi_tr":Xi_tr,"Xi_te":Xi_te}
                        if model in ("scorecard","ripper"):
                            edges=H.binarize_fit(Xi_tr)
                            Btr,bn=H.binarize_apply(Xi_tr,edges); Bte,_=H.binarize_apply(Xi_te,edges)
                            ctx.update(Btr=Btr,Bte=Bte,bnames=bn)
                        oof[np.where(te)[0]]=train_one(model,knob,val,Xtr,ytr,Xte,ctx)
                    except Exception as e:
                        ok=False; log(f"{model}/{knob}={val}/{split} FAIL: {type(e).__name__}: {str(e)[:80]}"); break
                if not ok: continue
                te_all=~np.isnan(oof)
                pr,roc,ns=gene_metrics(oof[te_all], te_all)
                records.append(dict(model=model,knob=knob,value=val,split=split,gene_pr=pr,gene_roc=roc,n_scored=ns,
                                    tier=("gpu" if model in GPU_TIER else "cpu")))
            log(f"{model}/{knob} [{split}] done ({len(vals)} values)")
    _atomic_csv(records, f"{OUT}/sweep_results.csv")     # checkpoint after each model family
    log(f"[{model}] sweep checkpoint ({len(records)} rows)")

if SWEEP_STAGE in ("grid","both"):
    _atomic_csv(records, f"{OUT}/sweep_results.csv")
    log(f"wrote sweep_results.csv: {len(records)} rows")
else:
    log("SWEEP_STAGE=search — grid phase skipped, sweep_results.csv left as-is")

# ======================================================================================
# DEOTTE-STYLE RANDOM SEARCH over the joint knob space — ALL families.
# Deotte's principle ("make the loop fast, then run many experiments"; nvidia-xgboost §4)
# is not XGBoost-specific. Every family fits fast on this 196-feat/6843-row matrix, so each
# gets a joint random search over its own space, with a per-family budget so the slow ones
# (EBM/NAM) don't dominate wall time. GBT families run device=cuda; the rest are CPU but still
# sub-second-to-few-seconds per fit. Headline split only, seeded/deterministic.
# ======================================================================================
import xgboost as xgb
from sklearn.preprocessing import StandardScaler

HEADLINE = "seqclust"
rng = np.random.default_rng(SEED)

# per-family search budget (n configs). Full budgets on every family — the CPU-tier families
# are parallelized across the Grace cores (see below), which makes a full search affordable in
# roughly the same wall-clock the trimmed serial search used to take. RIPPER stays modest: it is
# pure-Python (GIL-bound) so it does not thread-parallelize and each fit is comparatively slow.
BUDGET = {"xgb_baseline":256,"mono_gbm":256,"knn":120,                # GPU tier
          "elasticnet_glm":160,"cart":160,"scorecard":128,"ga2m":96,"nam":96,"ripper":48}  # CPU tier (full)

# CPU-tier families are fit config-by-config in a THREAD pool: their inner fits are C-backed
# (sklearn saga/liblinear/tree, EBM C++) or release the GIL (torch NAM), so threads give near-
# linear speedup WITHOUT a process pool — which would be unsafe here (this module runs work at
# import, so loky workers would re-execute the whole sweep). GPU-tier families stay SERIAL: they
# share one GPU context and parallel host threads would only contend. Measured ~6x on the GLM.
import os as _os

from joblib import Parallel, delayed
from threadpoolctl import threadpool_limits

N_THREADS = max(1, (_os.cpu_count() or 8) - 2)   # leave a couple cores for the driver / GPU host
GPU_FAMS  = {"xgb_baseline","mono_gbm","knn"}
# Families that must run SERIAL: GPU tier (shared GPU context) + NAM (uses the GLOBAL torch RNG
# via torch.manual_seed, so concurrent threads would clobber each other's generator — breaking
# determinism). All other CPU families seed through a thread-local estimator random_state.
SERIAL_FAMS = GPU_FAMS | {"nam"}

# per-family joint-space samplers -> a config dict (deterministic via the shared rng)
def samp_gbt():
    return dict(max_depth=int(rng.integers(3,13)), colsample_bytree=round(float(rng.uniform(0.3,0.95)),3),
                subsample=round(float(rng.uniform(0.5,1.0)),3), min_child_weight=int(rng.integers(1,16)),
                reg_lambda=round(float(10**rng.uniform(-1,1.3)),3), learning_rate=round(float(10**rng.uniform(-2,-0.7)),4))
def samp_glm():
    return dict(C=round(float(10**rng.uniform(-2.5,0.5)),4), l1_ratio=round(float(rng.uniform(0.0,1.0)),3))
def samp_cart():
    return dict(max_depth=int(rng.integers(2,13)), ccp_alpha=round(float(10**rng.uniform(-4,-1.7)),5),
                min_samples_leaf=int(rng.integers(10,120)))
def samp_scorecard(): return dict(C=round(float(10**rng.uniform(-2.5,0.3)),4))
def samp_knn():       return dict(k=int(rng.integers(3,120)))
def samp_ripper():    return dict(max_rules=int(rng.integers(3,28)), k=int(rng.integers(1,4)))
def samp_ga2m():      return dict(max_bins=int(rng.choice([64,128,256,512])), interactions=int(rng.integers(0,20)),
                                  learning_rate=round(float(10**rng.uniform(-2.5,-0.7)),4))
def samp_nam():       return dict(hidden=int(rng.choice([8,16,32,64])), weight_decay=round(float(10**rng.uniform(-6,-2.5)),7))
SAMPLERS = {"xgb_baseline":samp_gbt,"mono_gbm":samp_gbt,"elasticnet_glm":samp_glm,"cart":samp_cart,
            "scorecard":samp_scorecard,"knn":samp_knn,"ripper":samp_ripper,"ga2m":samp_ga2m,"nam":samp_nam}

# fit one config across the prebuilt folds -> OOF proba (reuses train_one's per-family logic,
# but here a config is a DICT of several knobs, so we fit directly)
def fit_config(fam, cfg, prebuilt, prebuilt_bin):
    oof=np.full(H.n,np.nan)
    for k,(Xtr,ytr,Xte,te,Xi_tr,Xi_te) in enumerate(prebuilt):
        try:
            if fam in ("xgb_baseline","mono_gbm"):
                p=H._xgb_params(mono=(fam=="mono_gbm")); p.update(cfg)
                p["scale_pos_weight"]=(ytr==0).sum()/max((ytr==1).sum(),1)
                dtr=xgb.QuantileDMatrix(Xtr,label=ytr); dte=xgb.QuantileDMatrix(Xte,ref=dtr)
                pr=xgb.train(p,dtr,num_boost_round=H.XGB_ROUNDS).predict(dte)
            elif fam=="elasticnet_glm":
                from sklearn.linear_model import LogisticRegression
                ss=StandardScaler().fit(Xtr)
                g=LogisticRegression(penalty="elasticnet",solver="saga",max_iter=2000,class_weight="balanced",
                                     random_state=SEED,**cfg).fit(ss.transform(Xtr),ytr)
                pr=g.predict_proba(ss.transform(Xte))[:,1]
            elif fam=="cart":
                from sklearn.tree import DecisionTreeClassifier
                t=DecisionTreeClassifier(class_weight="balanced",random_state=SEED,**cfg).fit(Xi_tr,ytr)
                pr=t.predict_proba(Xi_te)[:,1]
            elif fam=="scorecard":
                from sklearn.linear_model import LogisticRegression
                Btr,Bte,bn=prebuilt_bin[k]
                g=LogisticRegression(penalty="l1",solver="liblinear",class_weight="balanced",max_iter=2000,C=cfg["C"]).fit(Btr,ytr)
                coef=g.coef_[0]; nz=np.abs(coef[coef!=0]); scale=2.0/(nz.mean()+1e-9) if nz.size else 1.0
                pts=np.round(coef*scale).astype(int); raw=Bte@pts+g.intercept_[0]*scale
                pr=1/(1+np.exp(-(raw-raw.mean())/(raw.std()+1e-9)))
            elif fam=="knn":
                from sklearn.neighbors import KNeighborsClassifier as SKNN
                ss=StandardScaler().fit(Xtr)
                try:
                    from cuml.neighbors import KNeighborsClassifier as GKNN
                    mdl=GKNN(n_neighbors=cfg["k"]); mdl.fit(ss.transform(Xtr).astype(np.float32),ytr.astype(np.float32))
                    pp=mdl.predict_proba(ss.transform(Xte).astype(np.float32)); pp=pp.to_numpy() if hasattr(pp,"to_numpy") else np.asarray(pp)
                    pr=pp[:,1] if pp.ndim==2 else pp
                except Exception:
                    pr=SKNN(n_neighbors=cfg["k"]).fit(ss.transform(Xtr),ytr).predict_proba(ss.transform(Xte))[:,1]
            elif fam=="ripper":
                import wittgenstein as lw
                Btr,Bte,bn=prebuilt_bin[k]
                r=lw.RIPPER(random_state=SEED,max_rules=cfg["max_rules"],k=cfg["k"])
                r.fit(pd.DataFrame(Btr,columns=bn),ytr)
                pp=r.predict_proba(pd.DataFrame(Bte,columns=bn)); pr=pp[:,1] if getattr(pp,"ndim",1)==2 else np.asarray(pp,float)
            elif fam=="ga2m":
                from interpret.glassbox import ExplainableBoostingClassifier as EBC
                e=EBC(random_state=SEED,**cfg).fit(Xtr,ytr); pr=e.predict_proba(Xte)[:,1]
            elif fam=="nam":
                pr=_train_nam_sweep(Xtr,ytr,Xte,"hidden",cfg["hidden"]) if cfg.get("weight_decay") is None else _nam_cfg(Xtr,ytr,Xte,cfg)
            else:
                return None
            oof[np.where(te)[0]]=pr
        except Exception as e:
            log(f"{fam} cfg fail: {type(e).__name__}: {str(e)[:70]}"); return None
    return oof

def _nam_cfg(Xtr,ytr,Xte,cfg):
    import torch
    import torch.nn as nn
    torch.manual_seed(SEED); ss=StandardScaler().fit(Xtr)
    Xt=torch.tensor(ss.transform(Xtr),dtype=torch.float32); Xv=torch.tensor(ss.transform(Xte),dtype=torch.float32)
    yt=torch.tensor(ytr,dtype=torch.float32); F=Xt.shape[1]; hid=cfg["hidden"]; wd=cfg["weight_decay"]
    class FN(nn.Module):
        def __init__(s): super().__init__(); s.n=nn.Sequential(nn.Linear(1,hid),nn.ReLU(),nn.Linear(hid,1))
        def forward(s,x): return s.n(x)
    class NAM(nn.Module):
        def __init__(s): super().__init__(); s.fs=nn.ModuleList([FN() for _ in range(F)]); s.b=nn.Parameter(torch.zeros(1))
        def forward(s,x): return sum(s.fs[i](x[:,i:i+1]) for i in range(F)).squeeze(1)+s.b
    mdl=NAM(); opt=torch.optim.Adam(mdl.parameters(),lr=1e-3,weight_decay=wd)
    pw=torch.tensor([(ytr==0).sum()/max((ytr==1).sum(),1)],dtype=torch.float32); lf=nn.BCEWithLogitsLoss(pos_weight=pw)
    for _ in range(60): opt.zero_grad(); lf(mdl(Xt),yt).backward(); opt.step()
    with torch.no_grad(): return torch.sigmoid(mdl(Xv)).numpy()

# prebuild folds once (feature matrices are config-independent — the expensive part is shared)
folds_hl=H.SPLIT_REG[HEADLINE](H.df)
prebuilt=[]; prebuilt_bin=[]
for tr,te in folds_hl:
    Xtr,Xte=H.build_fold(tr,te)
    Xi_tr=np.nan_to_num(H.df.loc[tr,H.interp_cols].values); Xi_te=np.nan_to_num(H.df.loc[te,H.interp_cols].values)
    prebuilt.append((Xtr,H.y[tr],Xte,te,Xi_tr,Xi_te))
    edges=H.binarize_fit(Xi_tr); Btr,bn=H.binarize_apply(Xi_tr,edges); Bte,_=H.binarize_apply(Xi_te,edges)
    prebuilt_bin.append((Btr,Bte,bn))
log(f"random search: per-family budgets {BUDGET} on {HEADLINE} (folds prebuilt, device={H.DEVICE})")

def _eval_config(fam, i, cfg):
    """Fit one config across the prebuilt folds and score it — one unit of parallel work.
    threadpool_limits(1) pins each fit to a single BLAS/OpenMP thread so N parallel fits use N
    cores cleanly instead of N*cores oversubscribing. Returns a result row or None."""
    with threadpool_limits(1):
        oof = fit_config(fam, cfg, prebuilt, prebuilt_bin)
    if oof is None:
        return None
    te_all = ~np.isnan(oof); pr, roc, ns = gene_metrics(oof[te_all], te_all)
    return dict(model=fam, config_id=i, split=HEADLINE, gene_pr=pr, gene_roc=roc, config=json.dumps(cfg))

search_rows=[]
for fam in (FAMS if SWEEP_STAGE in ("search","both") else []):
    nb=BUDGET.get(fam,50); samp=SAMPLERS[fam]
    cfgs=[samp() for _ in range(nb)]                    # sample all configs first (shared seeded rng)
    if fam in SERIAL_FAMS:                              # GPU tier + NAM: serial
        res=[_eval_config(fam,i,c) for i,c in enumerate(cfgs)]
    else:                                               # other CPU families: thread-parallel
        res=Parallel(n_jobs=N_THREADS, prefer="threads")(
            delayed(_eval_config)(fam,i,c) for i,c in enumerate(cfgs))
    search_rows.extend(r for r in res if r is not None)
    _atomic_csv(search_rows, f"{OUT}/search_results.csv")           # checkpoint after each family
    log(f"{fam} random search done ({sum(r is not None for r in res)}/{nb} configs, "
        f"{'serial' if fam in SERIAL_FAMS else f'{N_THREADS}-thread'})")
    log(f"{fam} random search done ({nb} configs)")

if SWEEP_STAGE in ("search","both"):
    _atomic_csv(search_rows, f"{OUT}/search_results.csv")
    log(f"wrote search_results.csv: {len(search_rows)} rows (device={H.DEVICE})")
else:
    log("SWEEP_STAGE=grid — search phase skipped, search_results.csv left as-is")

RL.write_run_meta(OUT, extra={"job": "cvs_sweep_job", "seed": SEED, "device": H.DEVICE,
                  "headline": HEADLINE, "grids": {k: [kn for kn, _ in v] for k, v in GRIDS.items()},
                  "budget": BUDGET, "gpu_tier": sorted(GPU_TIER)},
                  preflight=_pf, outputs=[f"{OUT}/sweep_results.csv", f"{OUT}/search_results.csv"])
_lg.close("ok")
