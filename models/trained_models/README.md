# Trained model estimators

Serialized fitted models from the two Spark (NVIDIA GB10) runs behind the paper. Each
tarball holds eight model families per split (`random`, `gene`, `disease`, `seqclust`);
k-nearest-neighbour is instance-based and rebuilt from the training rows at scoring time,
so it has no saved file. Metrics and out-of-fold predictions in `results/` still cover all
nine families.

| Tarball | Source | What it is |
| --- | --- | --- |
| `models_training_harness.tar.gz` | `src/cvs_train_harness.py` | the final training run; each family refit on the full in-distribution data per split. These are the deployable estimators, and feed the Mode-A metric / permutation / out-of-fold stages. |
| `models_interpretation_harness.tar.gz` | `src/run_interp_harness.py` | the interpretation / comparison run; the fold-and-seed refits behind the Rashomon tie band, remove-and-retrain ablation, and cross-family convergence. |

The two runs are not bit-identical: the interpretation harness resamples across folds and
seeds, so its numbers move within the noise band rather than matching the training run's
fitted-model metrics exactly. Quote a number from one run or the other, never one value from
each as a single measurement.

## Layout inside each tarball

```
models/
  index.json                 which family -> which serialization format
  <split>/                   random | gene | disease | seqclust
    xgb_baseline.ubj         XGBoost reference        (Booster.load_model)
    mono_gbm.ubj             monotone GBM             (Booster.load_model)
    elasticnet_glm.pkl       elastic-net logistic     (pickle / joblib)
    cart.pkl                 decision tree            (pickle)
    scorecard.pkl            integer scorecard        (pickle)
    ripper.pkl               RIPPER rule list         (pickle)
    ga2m.pkl                 GA2M / EBM               (pickle)
    nam.pt                   neural additive model    (torch state_dict)
```

`nam.pkl` is present but empty: the NAM's model class is defined locally inside the training
function and cannot be pickled, so the weights are saved as a torch `state_dict` in `nam.pt`
and the architecture is rebuilt from `src/cvs_interp_adapter.py` before loading.

## Loading examples

```python
import tarfile, xgboost as xgb, pickle, io

with tarfile.open("models_training_harness.tar.gz") as t:
    # XGBoost reference on the homolog-safe split
    xgb_bytes = t.extractfile("models/seqclust/xgb_baseline.ubj").read()
    booster = xgb.Booster(); booster.load_model(bytearray(xgb_bytes))

    # a glass-box family (elastic-net GLM)
    glm = pickle.load(t.extractfile("models/seqclust/elasticnet_glm.pkl"))
```

The XGBoost reference is also shipped un-tarred as `models/cvs_gold_model.ubj`
(100%-refit on all rows), the single deployable artifact described in `model/model_card.md`.
The interpretable native representations (scorecard points, tree splits, rule text, GA2M
shape functions) are in `results/interpretation_harness/glassbox_models.json` and `results/interpretation_harness/ebm_shapes.json`.
