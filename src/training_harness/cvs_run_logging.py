#!/usr/bin/env python3
"""
cvs_run_logging.py — shared run provenance + logging for the CVS training/sweep jobs.

Single source of truth for how a run records *what it was* so the git repo and a methods
section can cite it. Imported by cvs_train_harness.py and cvs_sweep_job.py. No heavy deps
(stdlib + whatever the RAPIDS image already ships); safe to import before the CUDA stack.

Provides:
  RunLogger(run_dir, tag)         — timestamped, tee'd logging to <run_dir>/run.log + stdout
  gpu_preflight(require=True)     — FAIL-LOUD check that cuML KNN and XGBoost-CUDA really use
                                     the GPU; raises SystemExit(2) if not (never silent-CPU)
  environment_manifest()          — dict of image/GPU/driver/CUDA/package versions
  write_run_meta(run_dir, extra)  — emit run_meta.json (provenance block; hash-excluded from
                                     determinism check), merging environment + caller extras
  file_checksums(paths)           — sha256 of each output for the provenance record

Design notes:
  * run.log is line-buffered and flushed per record so a killed job still leaves a readable tail.
  * run_meta.json is the ONE place clocks/host-ids/versions live — the report body stays
    deterministic by never reading it into the hashed outputs.
  * gpu_preflight is the guard against a failure mode seen this project: on the alpha image a
    broken cuML import silently fell back to CPU while the job still exited 0 (e.g. job 76d5cc36
    finished exit 0 entirely on CPU). Preflight makes that abort instead of shipping CPU results.
"""
import datetime
import hashlib
import json
import os
import platform
import subprocess
import sys
import time


def _run(cmd):
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=30).stdout.strip()
    except Exception:
        return ""


class RunLogger:
    """Timestamped logger that tees to <run_dir>/run.log and stdout, with elapsed-time stamps."""
    def __init__(self, run_dir, tag="cvs"):
        self.run_dir = run_dir
        self.tag = tag
        os.makedirs(run_dir, exist_ok=True)
        self.path = os.path.join(run_dir, "run.log")
        self._t0 = time.time()
        self._fh = open(self.path, "a", buffering=1)  # line-buffered
        self.log(f"=== {tag} run start {datetime.datetime.utcnow().isoformat()}Z (pid {os.getpid()}) ===")

    def log(self, msg):
        line = f"[t+{time.time() - self._t0:6.0f}s] {msg}"
        print(line, flush=True)
        self._fh.write(line + "\n"); self._fh.flush()

    def close(self, status="ok"):
        self.log(f"=== run end status={status} elapsed={time.time() - self._t0:.0f}s ===")
        self._fh.close()


def _pkg_versions():
    out = {}
    for p in ("numpy", "scipy", "pandas", "scikit_learn", "xgboost", "cuml", "cudf",
              "cupy", "torch", "interpret", "wittgenstein"):
        mod = p.replace("scikit_learn", "sklearn")
        try:
            out[p] = __import__(mod).__version__
        except Exception:
            out[p] = None
    return out


def environment_manifest():
    """Everything a reader needs to know WHICH machine/stack produced the run."""
    gpu_name = _run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"])
    driver = _run(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"])
    cuda = ""
    smi = _run(["nvidia-smi"])
    for tok in smi.split():
        if tok.replace(".", "").isdigit() and "." in tok and smi.find("CUDA Version") >= 0:
            pass  # parsed below more robustly
    if "CUDA Version:" in smi:
        cuda = smi.split("CUDA Version:")[1].split()[0]
    return {
        "docker_image": os.environ.get("CVS_IMAGE", "unknown"),
        "gpu_name": gpu_name,
        "driver_version": driver,
        "cuda_version": cuda,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": _pkg_versions(),
    }


def gpu_preflight(logger=None, require=True):
    """
    FAIL-LOUD GPU check. Verifies (a) cuML KNN and (b) XGBoost device=cuda actually run on GPU.
    Returns a dict of results. If require=True and either can't use the GPU, exits non-zero so a
    'GPU run' can never silently produce CPU results. This is checklist gate A3/B4.
    """
    def _say(m):
        (logger.log if logger else print)(m)
    res = {"cuml_knn": False, "xgb_cuda": False}
    # (a) cuML KNN on GPU
    try:
        import numpy as np
        from cuml.neighbors import KNeighborsClassifier as GKNN
        X = np.random.rand(100, 8).astype("float32"); y = (np.random.rand(100) > 0.6).astype("float32")
        GKNN(n_neighbors=5).fit(X, y).predict(X[:10])
        res["cuml_knn"] = True; _say("preflight: cuML KNN on GPU OK")
    except Exception as e:
        _say(f"preflight: cuML KNN FAILED — {type(e).__name__}: {str(e)[:160]}")
    # (b) XGBoost device=cuda
    try:
        import numpy as np
        import xgboost as xgb
        X = np.random.rand(200, 10).astype("float32"); y = (np.random.rand(200) > 0.7).astype(int)
        d = xgb.QuantileDMatrix(X, label=y)
        xgb.train({"device": "cuda", "tree_method": "hist", "max_depth": 3}, d, num_boost_round=5)
        res["xgb_cuda"] = True; _say("preflight: XGBoost device=cuda OK")
    except Exception as e:
        _say(f"preflight: XGBoost CUDA FAILED — {type(e).__name__}: {str(e)[:160]}")
    if require and not (res["cuml_knn"] and res["xgb_cuda"]):
        _say("preflight: GPU REQUIREMENT NOT MET — aborting rather than running degraded on CPU.")
        sys.exit(2)
    return res


def file_checksums(paths):
    out = {}
    for p in paths:
        if os.path.exists(p):
            h = hashlib.sha256()
            with open(p, "rb") as f:
                for chunk in iter(lambda: f.read(1 << 20), b""):
                    h.update(chunk)
            out[os.path.basename(p)] = {"sha256": h.hexdigest(), "bytes": os.path.getsize(p)}
    return out


def write_run_meta(run_dir, extra=None, preflight=None, outputs=None):
    """
    Emit run_meta.json — the provenance block. HASH-EXCLUDED from the report determinism
    check (it carries clocks/versions on purpose). Merges environment + preflight + caller extras.
    """
    meta = {
        "written_utc": datetime.datetime.utcnow().isoformat() + "Z",
        "environment": environment_manifest(),
    }
    if preflight is not None:
        meta["gpu_preflight"] = preflight
    if extra:
        meta.update(extra)
    if outputs:
        meta["outputs"] = file_checksums(outputs)
    path = os.path.join(run_dir, "run_meta.json")
    json.dump(meta, open(path, "w"), indent=2, sort_keys=True)
    return path
