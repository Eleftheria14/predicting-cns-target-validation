#!/usr/bin/env python
"""Driver: run the full model-comparison harness protocol against the CVS
publication training run, via cvs_interp_adapter.

Retrains every family from factories across folds+seeds on the frozen,
homolog-safe seqclust split lock (the harness contract refuses pre-fitted
models). Produces the Rashomon band, remove-and-retrain ablation over feature
families, and cross-family convergence — the stages Mode A cannot give.

Inputs staged alongside this script (same dir):
  cvs_interp_adapter.py  harness_kernel.py  feature_spec.json
  gold.parquet  manifest.csv  monotone_dirs.json  metrics_by_split.csv
Env:
  OUT       output dir (default ./interp_out)
  SEEDS     number of Rashomon seeds (default 10)
  MODELS    comma-separated family subset (default: all in feature_spec)
"""
import importlib.util
import json
import os
import sys
import time

OUT = os.environ.get("OUT", "interp_out"); os.makedirs(OUT, exist_ok=True)
N_SEEDS = int(os.environ.get("SEEDS", "10"))
HERE = os.path.dirname(os.path.abspath(__file__))
_t0 = time.time()
def log(m): print(f"[t+{time.time()-_t0:6.0f}s] {m}", flush=True)

def _load(name, path):
    s = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(s); sys.modules[name] = m; s.loader.exec_module(m); return m

AD = _load("cvs_interp_adapter", os.path.join(HERE, "cvs_interp_adapter.py"))
H  = _load("harness_kernel",     os.path.join(HERE, "harness_kernel.py"))
log("adapter + harness kernel loaded")

spec = AD.load_feature_spec(os.path.join(HERE, "feature_spec.json"))
mono = None
mono_path = os.path.join(HERE, "monotone_dirs.json")
if os.path.exists(mono_path):
    with open(mono_path) as fh: mono = json.load(fh)

models_env = [m for m in os.environ.get("MODELS", "").split(",") if m.strip()] or None

# rebuild the modelling frame (raw interp + raw embedding; estimators PCA in-fold)
built = AD.build_matrix(os.path.join(HERE, "gold.parquet"),
                        os.path.join(HERE, "manifest.csv"), spec)
X, y = built["X"], built["y"]
log(f"matrix: {X.shape}  n_interp={built['n_interp']} n_emb={built['n_emb']}  "
    f"families={list(built['feature_blocks'])}")

# mono vector aligned to interp+PCA feature order, if the harness provided one
mono_vec = None
if isinstance(mono, dict) and "vector" in mono:
    mono_vec = mono["vector"]
elif isinstance(mono, list):
    mono_vec = mono

factories = AD.make_factories(spec, built["n_interp"], built["n_emb"],
                              models=models_env, mono_vec=mono_vec)
log(f"factories: {list(factories)}")

# frozen seqclust split lock — both harnesses enforce byte-identical folds
lock, row_ids = AD.build_split_lock_from_seqclust(
    os.path.join(HERE, "gold.parquet"), spec, H)
log(f"split lock: {lock.get('source')}  fp={str(lock.get('fingerprint'))[:16]}")

# optimism-ladder table (random->seqclust) for report Section 4
import pandas as pd

splits_table = pd.read_csv(os.path.join(HERE, "metrics_by_split.csv"))

report = H.run_comparison(
    X, y,
    models=factories,
    groups=built["groups"],
    feature_blocks=built["feature_blocks"],
    group_key=built["group_key"],
    seeds=range(N_SEEDS),
    k=20,
    headline_metric="gene_roc_auc",
    split_lock=lock,
    row_ids=row_ids,
    splits_table=splits_table,
    report_meta={
        "dataset_name": "CNS CVS gold (publication run a2314ad3)",
        "n_genes": int(pd.Series(built["group_key"]).nunique()),
        "positive_rate": float(y.mean()),
        "harness_version": "kernel.py 4d892df4",
        "adapter_version": "cvs_interp_adapter f59ea204",
        "score_not_calibrated": True,
        "extra_caveats": [
            "Harness retrains from factories; its numbers are NOT bit-identical "
            "to the training run's fitted-model metrics (same direction, "
            "resampled values). Do not quote both as one measurement.",
            "KNN is instance-based: the factory rebuilds the neighbour set from "
            "each training fold (cuML GPU k-NN, sklearn fallback), k=50, on the "
            "standardized interp+in-fold-PCA space — no persisted model reused.",
        ],
    },
    outdir=OUT,
)
log(f"run_comparison done -> {OUT}")
print("HEADLINE_TIE_DECISIONS:", json.dumps(report.get("tie_decisions", {}))[:800])
