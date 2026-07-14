# Gold Modeling Dataset: Data Card

`cns_cvs_gold_modeling_dataset.parquet`, the frozen, production-ready training dataset for
the CNS Clinical Viability Score. One row per **(target gene, indication) pair**.

- **Shape:** 6,843 rows × 2,735 columns
- **Coverage:** 1,786 genes × 258 CNS indications (ATC class N; EFO/MONDO nervous-system & mental-disorder branches)
- **Columns:** 3 identifiers + 5 label columns + 3 frozen fold columns + **2,724 features**
- **Companion:** `cns_cvs_gold_modeling_dataset_manifest.csv`, one row per column giving its
  `block` and `test_time_source`.

## Identifier columns (3)
| Column | Meaning |
|---|---|
| `ensembl_gene_id` | Ensembl gene ID of the target |
| `efo_id` | EFO/MONDO ID of the indication |
| `gene_symbol` | HGNC symbol (convenience) |

## Label columns (5)
| Column | Values | Use |
|---|---|---|
| **`label_consensus`** | 1,570 pos / 5,273 neg | **CANONICAL training target.** 1 = clinically validated, 0 = efficacy-failure or broad negative |
| `label_consensus_v2` | 1,570 / 5,267 (6 NaN dropped) | cleaned twin, the version actually trained on |
| `label_efficacy_gold` | 565 pos / 96 neg / 6,182 NaN | tier-1 efficacy-adjudicated subset only (confidence 0.95) |
| `label_tier` | 1–4 | evidence quality: 1 = gold efficacy … 4 = broad negative |
| `label_confidence` | 0.25–0.95 | per-pair confidence weight |

Label construction (11 mining channels, precedence-weighted consensus, observed vs mined
assumptions) is documented in `../provenance/PROVENANCE.md`.

## Frozen fold columns (3): evaluate with these, do not re-split
| Column | Split | When to use |
|---|---|---|
| **`fold_seqclust`** | genes clustered by DIAMOND ≥50% identity (union-find), whole clusters held out, homolog-safe, 0% cross-fold homology | **PRIMARY**, the honest deployment number for a novel protein |
| `fold_gene` | random gene assignment (seed 0), gene-disjoint but NOT homolog-safe | diagnostic, ~0.05 optimistic |
| `fold_disease` | random disease assignment (seed 0) | combine with `fold_gene` for both-axes-disjoint (test = gene_fold==k AND disease_fold==k) |

## Feature blocks (2,724 features, 5 blocks: all cold-start / novel-target safe)
| Block | n | Source |
|---|---|---|
| `esmc6b` | 2,560 | ESM-Cambrian 6B embeddings (layer 64, mean-pooled), reduce to PCA-32 **in-fold** |
| `intrinsic_84` | 84 | UniProt sequence, InterPro domains, gnomAD constraint, GTEx/HPA brain expression |
| `predicted_go` | 61 | GO terms predicted from sequence-derived InterPro domains |
| `structure` | 15 | AlphaFold-predicted structure descriptors |
| `homology` | 4 | DIAMOND homology to known targets (label-free) |

**Zero leakage-tier features**, labels, post-hoc outcome-derived, temporal-confounded,
realized-investment/survivorship, and popularity features are all excluded (audited against
`../provenance/tables/feature_provenance_contract.csv`). Per-column source/computation in
`../provenance/tables/feature_computation_provenance.csv`; group-level include/exclude rationale in
`../provenance/tables/feature_group_decisions.csv`.

## Load & evaluate
```python
import pandas as pd
df  = pd.read_parquet("cns_cvs_gold_modeling_dataset.parquet")
man = pd.read_csv("cns_cvs_gold_modeling_dataset_manifest.csv")

feature_cols = man[man.block.isin(
    ["esmc6b","intrinsic_84","predicted_go","structure","homology"])].column.tolist()
X, y = df[feature_cols], df["label_consensus"]

for k in range(5):                       # homolog-safe primary split
    train = df.fold_seqclust != k
    test  = df.fold_seqclust == k
    # fit on train (PCA-32 the esmc6b block IN-FOLD), score on test
```

## Provenance of the reported result
Model trained on this dataset (XGBoost, GPU): **0.856 ROC-AUC** under `fold_seqclust`
(honest deployment number), 0.909 under `fold_gene`, 0.860 both-axes-disjoint; permutation
control 0.457; precision@20 = 1.0. See `../models/model_card.md`.
