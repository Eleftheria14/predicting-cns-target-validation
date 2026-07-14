# Provenance & Data Sources, CNS Clinical Viability Score

This is the single authoritative provenance document for the study. It records **where
every feature and label came from**, **how each was computed**, **which feature groups were
included or excluded and why**, and **the complete inventory of external databases, tools,
and comparable work**. It supports the manuscript's Methods and Supporting Information.

**Machine-readable companions (in this folder):**

| File | One row per | Columns |
|---|---|---|
| `tables/feature_computation_provenance.csv` | model feature (2,724) | column, block, source_database, extraction_tool, computation |
| `tables/feature_group_decisions.csv` | feature group | feature_group, decision, leaky, rationale |
| `tables/feature_provenance_contract.csv` | column in the raw 360-feature matrix | leakage tier (the authoritative banned-tier list) |
| `tables/label_provenance.csv` | label channel | n, pos/neg, evidence_type, mean_confidence, what_was_observed, what_was_assumed |
| `tables/label_column_dictionary.csv` | label column | definition |
| `tables/literature_comparison.csv` | comparable study | metrics, split, evidence type |
| `tables/cns_outcome_interpretation_layer.parquet` | (target, indication) pair | per-pair outcome interpretation / audit fields |

> **Reproducibility note.** Column-to-source assignments were reconstructed from column names
> and the block manifest, cross-checked against the pipeline's block structure. They are
> accurate at the block/source level and at the per-column level for the 84 intrinsic
> features (each individually verified). Exact tool *versions* (gnomAD v2 vs v4,
> InterProScan build, GTEx v8) should be confirmed against the upstream pull logs before
> publication, see Section 7, Version caveats.

---

# Part I, Features

## 1. Feature data sources

| Block | n | Source database | Extraction | Computation |
|---|---|---|---|---|
| ESMC-6B embeddings | 2,560 | UniProt reviewed sequence | ESM-Cambrian 6B forward pass (GPU) | Mean-pooled residue embedding, transformer **layer 64** (best of a layer sweep) |
| Genome-intrinsic | 84 | UniProt / InterPro / gnomAD / GTEx / HPA | see per-column table | AA composition, biophysics, domain/family membership, genetic constraint, brain expression |
| Predicted-GO | 61 | InterPro 5 + InterPro2GO map | InterProScan -> GO mapping | GO-term presence predicted from **sequence-derived** domains (NOT curated GO) |
| Structure | 15 | AlphaFold DB v6 | Biotite/Biopython on predicted monomer | pLDDT stats, radius of gyration, secondary-structure content, contact-order |
| Homology | 4 | This target set (all-vs-all) | DIAMOND blastp (e <= 1e-3) | Top-hit identity, hit counts, label-free |

Within the 84 intrinsic features: 32 from the UniProt sequence (Biopython ProtParam
biophysics + AA-composition fractions), 32 from InterPro (family flags, domain counts),
10 from gnomAD constraint, 4 from GTEx v8, 3 from HPA, 3 from UniProt topology annotation.
Full per-column breakdown in `tables/feature_computation_provenance.csv`.

## 2. How the ESMC-6B embeddings were computed
1. Reviewed protein sequences pulled from UniProt for all target genes.
2. Forward pass through **ESM-Cambrian 6B** (ESM3-family, 6-billion parameters) on the
   DGX Spark GPU (token budget 8,000; sequences chunked as needed).
3. Per-residue hidden states extracted at **layer 64** (selected by a layer sweep over
   {40,48,52,56,60,64,72}; 64 gave the best held-out lift, +0.059).
4. **Mean-pooled** over residues -> one 2,560-dim vector per protein.
5. At modeling time, reduced to **32 principal components fit IN-FOLD** (PCA fit on the
   training fold only, applied to test) to prevent embedding leakage.

The gold dataset ships only the ESMC-6B block (the block the model uses). The full layer
sweep and the raw embeddings superset (5,432 columns: ESM-2 650M + ESMC-600M + ESMC-6B +
GO/structure/homology) are not stored in the repository: they are not needed to reproduce the
model or the EDA, and the pipeline in `src/` documents how to regenerate them from the
sequences.

## 3. Feature groups: included / excluded and why

See `tables/feature_group_decisions.csv` for the machine-readable table. Summary:

### Included (all cold-start / novel-target safe, 0 leakage)
- **ESMC-6B (2,560)**, **genome-intrinsic (84)**, **predicted-GO (61)**, **structure (15)**,
  **homology (4)**, 2,724 features total. Every one is computable from a protein's
  sequence (± its gene ID) at target-discovery time, before any clinical investment.
- **Causal / genetic evidence** (gen_*, ClinGen, GWAS), the causal covariate arm.

### Explored then excluded
- **7 gene-property groupby aggregates**, label-free and within-disease-discriminating in
  isolation, but redundant with ESMC-6B+GO+structure (±0.002 on the full stack). Kept in v3 lineage, dropped from canonical.

### Excluded, bias / misalignment (not leaky, but wrong for the objective)
- **13 disease-keyed groupby aggregates**, a disease-knowledge prior (+0.058 cross-disease,
  +0.005 within-disease), not a target discriminator.
- **Popularity / group-size counts**, disease crowding (-0.54 with success) and gene
  track-record (survivorship, ~0 for novel targets). Excluded per the anti-waste objective.
- **Patent activity**, used as a target COVARIATE, never as a feature in the gold set nor as
  a label. A patent signals a target was identified, not that it succeeded or failed; patent
  intensity is a popularity signal (the confound the study excludes) and is target-only and
  time-smeared.

### Excluded, banned leakage tiers (from `tables/feature_provenance_contract.csv`)
- **Tier 0, labels (31):** the outcome itself.
- **Tier 1, post-hoc outcome-derived (35):** computed from what happened *after* the
  decision (max_phase_reached, ctgov_n_failed_trials, tgt_other_failures, moa_confirmed).
- **Tier 2, temporal-confounded (6):** accrue over time as a target is studied (first_year,
  total_evidence, recent_evidence).
- **Tier 3, realized-investment / survivorship (16):** exist because someone already
  invested (n_ligands, is_ligandable, sm_tractable, chem_n_compounds).

**Verification:** the gold modeling dataset contains **0** of the 88 banned tier-0/1/2/3
features (audited against the contract), **0** disease-keyed groupby features, and **0**
popularity/count features.

### Available but used cautiously
- **Association evidence (assoc_*)**, Open Targets association scores; missing-not-at-random
  (present mainly for already-studied targets). Down-weighted; adds ±0.0003 on the fair set.

## 4. How features were combined / transformed

Most features are used as-is, but several were **derived by combining or transforming**
upstream columns, these need explicit description in Methods:

- **ESMC-6B -> PCA-32 (in-fold).** The 2,560-dim embedding is reduced to 32 principal
  components, with the PCA **fit on the training fold only** and applied to the test fold,
  inside every CV iteration. This is the single most important anti-leakage step on the
  feature side.
- **Groupby aggregates (explored).** Candidate features were built by grouping pairs on a key
  (disease, protein-family, membrane-class, length-bin, domain-count-bin) and aggregating an
  intrinsic property (mean/std/min/max/median of GRAVY, gnomAD constraint, brain expression).
  A GPU forward-search (cuDF groupby, Deotte-style: generate many, keep what improves CV)
  selected 21. Decomposition split them into **disease-keyed** (cross-disease prior, excluded)
  and **gene-property-keyed** (within-disease discriminators). The 7 gene-property survivors
  were folded into v3 then dropped from canonical as redundant. All still reproducible from
  `../data/raw_lineage/`. See `../figures/groupby_feature_search.png` and
  `../figures/within_disease_decomposition.png`.
- **Causal vs association arms.** Genetic-causality features (gnomAD constraint, ClinGen
  curation strength, GWAS min-p) form the *causal arm*; Open Targets scores form the
  *association arm*. These were tested as separate covariate blocks (the causal-vs-association
  experiment, `../figures/causal_vs_association.png`), controlling for brain penetrance.
- **Predicted-GO.** Not a raw pull, InterPro domains (from sequence) are mapped to GO terms
  via the InterPro2GO table, then encoded as presence features. Combination of two upstream
  resources into one block.
- **Group-key derivation.** membrane_class / length_bin / domain_count_bin / protein_family
  were derived by binning raw columns (e.g. length: S<=250, M 250-500, L 500-1000, XL>1000);
  derivation rules validated at 97.9-100% agreement against the source pass.
- **Near-duplicate passes.** `seq_len` approx `protein_length` and `seq_mass` approx
  `protein_mass` come from two annotation passes; note or deduplicate in Methods.

---

# Part II, Labels (the outcome)

## 5. Label mining channels

Every (target, indication) label was assembled from **11 independent mining channels**, each
recorded as a `src_*` flag in `../data/raw_lineage/cns_labels_master.parquet`. The final
`label_consensus` is a precedence-weighted consolidation across channels.

| Channel (`src_*`) | Pairs | What it is |
|---|---|---|
| `src_gold` | 661 | **Gold tier**, efficacy-adjudicated approved-vs-efficacy-failure verdicts (tier 1, confidence 0.95) |
| `src_advancement` | 3,350 | Clinical phase-advancement signal (phase reached vs stalled) |
| `src_pipeline` | 2,297 | Company pipeline appearance / disappearance (shareholder & pipeline docs) |
| `src_ctgov` | 1,824 | ClinicalTrials.gov trial outcomes / status |
| `src_llm` | 431 | LLM-adjudicated stop-reason classification of ambiguous trials (down-weighted proxy) |
| `src_ctgov_moa` | 171 | ctgov cross-checked against ChEMBL mechanism-of-action (MoA-confirmed target attribution) |
| `src_ot` | 105 | Open Targets stop-reason failure channel |
| `src_sec` | 74 | SEC filing disclosures (disappeared programs from 10-K/20-F) |
| `src_press` | 23 | Company press-release discontinuation announcements |
| `src_conf_abstract` | 18 | Conference-abstract mining |
| `src_sec_company` | 12 | SEC per-company deep sweep |

**The central caveat:** the outcome label is *not* uniformly a hard clinical fact. Only a
minority of pairs have an adjudicated efficacy verdict; the majority are **inferred from mined
signals**, a program that disappeared from a pipeline, a trial terminated for futility, a
disclosure dropped from an SEC filing.

## 6. Evidence type per channel (observed vs inferred)

Channels split into **observed** (hard or self-reported facts) and **inferred** (an
assumption from mined text/behaviour). Confidence weights encode that distinction.

| Channel | n | pos/neg | Evidence type | Mean conf | The assumption made |
|---|---|---|---|---|---|
| `src_gold` | 661 | 565/96 | **OBSERVED, hard** | 0.95 | none, FDA approval / published efficacy verdict (ground truth) |
| `src_press` | 23 | 3/20 | **OBSERVED, announcement** | 0.87 | company named the discontinuation itself |
| `src_ctgov_moa` | 171 | 63/108 | **OBSERVED, MoA-confirmed** | 0.80 | ctgov outcome + target link verified via ChEMBL mechanism |
| `src_ot` | 105 | 19/86 | **OBSERVED, curated** | 0.75 | Open Targets curated stop-reason categories |
| `src_ctgov` | 1,824 | 78/1,746 | OBSERVED status + INFERRED reason | 0.52 | trial terminated/withdrawn -> failure; why-stopped parsed (efficacy_futility 1,400, safety 424) |
| `src_advancement` | 3,350 | 1,465/1,885 | **INFERRED, maturity rule** | 0.75 | Phase 1-2 with first_year <= 2015 & no progress -> "matured stall" = failure |
| `src_pipeline` | 2,297 | 104/2,193 | **INFERRED, disappearance** | 0.34 | present then absent >= 5 yr (median 8 yr) -> assumed discontinued |
| `src_sec` | 74 | 6/68 | **INFERRED, disclosure drop** | 0.30 | program dropped from later SEC filings -> assumed discontinued |
| `src_sec_company` | 12 | 3/9 | **INFERRED, per-company sweep** | 0.30 | same, company-scoped |
| `src_llm` | 431 | 14/417 | **INFERRED, LLM adjudication** | 0.41 | LLM classified ambiguous why-stopped free text; **down-weighted (0.40)** for target-attribution promiscuity |
| `src_conf_abstract` | 18 | 2/16 | **INFERRED, abstract mining** | 0.51 | negative result in a conference abstract -> failure signal |

**Assumption thresholds worth citing in Methods:**
- *Pipeline disappearance:* silence >= **5 years** (median 8, max 21) before a program is
  called dead, `pipeline_years_silent` records the exact gap per pair.
- *Phase-maturity stall:* Phase 1-2 pairs first seen **<= 2015** with no advancement are
  treated as matured failures (a stalled early-phase program that never progressed).
- *Patent activity was used as a FEATURE (covariate), never as a label*, a patent signals a
  target was identified, not that it succeeded or failed.

## 7. Label tiers (confidence stratification)

| Tier | n | Composition | label_confidence |
|---|---|---|---|
| 1, gold | 661 | efficacy-adjudicated (565 pos / 96 neg) | 0.95 |
| 2, phase-advancement | 1,794 | maturity-rule negatives | ~0.75 |
| 3, ctgov/pipeline/proxy | 2,550 | mixed observed+inferred (1,005 pos / 1,545 neg) | mixed |
| 4, broad negatives | 1,838 | investment-decision framing | lower |

Overall `label_confidence`: mean 0.61, median 0.70, range 0.25-0.95 (recorded per pair).
**A reviewer-facing sensitivity analysis should re-fit on tier-1+2 only** and report whether
conclusions hold, the model card's headline uses the full `label_consensus`.

## 8. How the channels were combined into `label_consensus`

Multiple channels can touch the same (target, indication) pair. Combination rule:

1. **Precedence** (not majority vote): `gold > adjudicated-efficacy (ctgov / MoA / OT /
   press) > phase-advancement > proxy (llm / pipeline / sec)`. The highest-precedence channel
   present sets the label and its tier.
2. **Multi-source support** is recorded in `n_sources` (1-5 channels per pair): 5,238 pairs
   rest on a single channel, 1,210 on two, 133 on three, 31 on four, 5 on five. More sources
   -> higher effective confidence.
3. **Conflicts** are flagged, not hidden: `conflict = 1` for **401 pairs** where channels
   disagreed on the outcome. Precedence resolves the final label; the flag lets you exclude
   or inspect them.
4. `label_consensus_v2` is a cleaned twin (6 NaN pairs dropped: 1,570 pos / 5,267 neg) -
   the version actually trained on.
5. Per-pair audit trail: `label_source` (winning channel), `label_tier`, `label_confidence`,
   `n_sources`, `conflict`, plus every raw `src_*` flag and `*_confidence` and the reason
   fields (`ctgov_reason`, `pipeline_years_silent`, `sec_programs`, `ctgov_nct_ids`) are all
   preserved in `../data/raw_lineage/cns_labels_master.parquet`, so any pair's label can be
   traced to its evidence.

Final: `label_consensus` = **1,570 positive / 5,273 negative**; the gold efficacy subset
(`label_efficacy_gold`) = 565 positive / 96 negative.

**Recommended Methods framing.** Describe the label as a **precedence-weighted consensus over
11 evidence channels spanning hard outcomes (regulatory approval, curated efficacy) and mined
proxies (trial termination reasons, pipeline disappearance, SEC-disclosure drops,
LLM-adjudicated stop reasons)**, with a per-pair confidence (0.25-0.95) and a tier (1 gold ...
4 broad-negative). State the disappearance/maturity thresholds explicitly. Report the
tier-1+2-only sensitivity analysis alongside the full-label result so the dependence on mined
assumptions is transparent.

---

# Part III, Complete source inventory

For the manuscript's Supporting Information / Data Availability. **Confirm exact version tags
and access dates against the pull logs before submission**, flagged with (confirm) where the
version was not recorded at build time.

## S1. Feature data sources

| # | Resource | Used for | Access point | Version |
|---|---|---|---|---|
| 1 | **UniProt** (reviewed/SwissProt) | Protein sequences; topology; protein names, accessions | https://www.uniprot.org · REST API | confirm release |
| 2 | **InterPro / InterProScan** | Protein family & domain annotation -> intrinsic features and predicted-GO | https://www.ebi.ac.uk/interpro · InterProScan | confirm build |
| 3 | **InterPro2GO** mapping | Domain -> GO-term prediction (cold-start-safe GO) | https://www.ebi.ac.uk/GOA/InterPro2GO | confirm date |
| 4 | **gnomAD** (genetic constraint) | pLI, LOEUF, missense/LoF z-scores, oe ratios | https://gnomad.broadinstitute.org | v2 vs v4, confirm |
| 5 | **GTEx** | Brain-tissue expression (TPM, tissue-specificity) | https://gtexportal.org | v8 |
| 6 | **Human Protein Atlas (HPA)** | Brain-region normalized expression (nTPM) | https://www.proteinatlas.org | confirm release |
| 7 | **AlphaFold Protein Structure DB** | Predicted monomer structures -> 15 structure descriptors | https://alphafold.ebi.ac.uk | v6 (AFDB) |
| 8 | **ESM-Cambrian 6B** (protein LM) | 2,560-dim sequence embeddings (layer 64, mean-pooled) | ESM3-family model weights | record checkpoint hash |
| 9 | **DIAMOND** (blastp) | All-vs-all homology -> homology features + sequence-clustered folds | https://github.com/bbuchfink/diamond | confirm version |
| 10 | **Biopython** (SeqUtils.ProtParam) | Sequence biophysics (GRAVY, pI, instability, MW, SS fractions) | https://biopython.org | confirm version |
| 11 | **Open Targets** | Association-evidence arm (assoc_* features) | https://platform.opentargets.org | confirm release |
| 12 | **ClinGen** | Gene-disease causal curation strength (causal arm) | https://clinicalgenome.org | confirm date |
| 13 | **GWAS Catalog** | Genetic-association p-values (causal arm) | https://www.ebi.ac.uk/gwas | confirm release |
| 14 | **ChEMBL** | Mechanism-of-action target attribution (label cross-check); ligandability | https://www.ebi.ac.uk/chembl | confirm version |
| 15 | **IMPC** | Mouse knockout / nervous-phenotype (causal audit) | https://www.mousephenotype.org | confirm release |

Cheminformatics (RDKit for physchem/MPO/fingerprints) was computed for the compound side
where compounds were available; RDKit version to confirm.

## S2. Label / outcome data sources (11 mining channels)

Full per-channel detail (observed vs assumed, confidence, thresholds) in Sections 5-8 and
`tables/label_provenance.csv`. Sources:

| # | Channel | Resource | Access point |
|---|---|---|---|
| 1 | `src_gold` | FDA approvals / published efficacy verdicts | https://www.accessdata.fda.gov/scripts/cder/daf (Drugs@FDA) |
| 2 | `src_ctgov`, `src_ctgov_moa` | ClinicalTrials.gov (trial status, why-stopped) | https://clinicaltrials.gov · API v2 |
| 3 | `src_advancement` | Clinical phase reached (ctgov-derived) | ClinicalTrials.gov |
| 4 | `src_pipeline` | Company pipeline / shareholder pipeline docs | company IR pages; press/pipeline archives |
| 5 | `src_ot` | Open Targets stop-reason categories | https://platform.opentargets.org |
| 6 | `src_press` | Company press-release discontinuations | company newsrooms; PR wires |
| 7 | `src_sec`, `src_sec_company` | SEC filings (10-K / 20-F pipeline disclosures) | https://www.sec.gov/edgar (EDGAR) |
| 8 | `src_llm` | LLM adjudication of ambiguous ctgov why-stopped text | derived (in-project) |
| 9 | `src_conf_abstract` | Conference-abstract mining | conference proceedings |

**Medicinal-chemistry dead-drug literature venues** (author-affiliation filtered for pharma):
*J. Med. Chem.*, *Bioorg. Med. Chem.*, *J. Med. Chem. Lett.* Literature access via
CrossRef/Unpaywall/PubMed. **Company reference list:** ~1,042 firms compiled from FDA
applicants + SEC filers.

Patent activity (ChEMBL patent links, Espacenet/Google Patents attempts) was used as a
**feature (covariate), never as a label**.

## S3. Comparable published work (verified DOIs)

Full table with metrics/splits in `tables/literature_comparison.csv`.

| Study | Venue | DOI | Reported |
|---|---|---|---|
| Ferrero et al. 2017 | J Transl Med | `10.1186/s12967-017-1285-6` | ROC-AUC 0.76 (random split, association) |
| Han et al. 2022 | BMC Bioinformatics | `10.1186/s12859-022-04753-4` | AUROC 0.808 (assoc) -> 0.914 (+computed); PR-AUC 0.73 |
| Lo et al. 2019 | Harvard Data Sci Rev | `10.1162/99608f92.5c5f0525` | ROC-AUC 0.78-0.81 (walk-forward, trial-level) |
| OTRec 2025 | preprint | `10.64898/2025.12.21.695803` | ROC-AUC 0.559 / 0.863 / 0.950 |
| Minikel et al. 2024 | Nature | `10.1038/s41586-024-07316-0` | genetic support -> 2.6x approval |

## S4. Software & compute
- **XGBoost** (GPU, `device=cuda`, GPU-resident QuantileDMatrix), reference model.
- **RAPIDS** (cuDF, cuML), GPU feature search; container `rapidsai/base:26.08a-cuda13-py3.12`.
- **scikit-learn** (PCA, StandardScaler, metrics), **pandas**, **numpy**.
- Glass-box families: **interpret** (GA2M/EBM), **wittgenstein** (RIPPER), **torch** (NAM).
- **Compute:** NVIDIA DGX Spark (GB10 GPU) for embeddings and model training; local CPU for tabular/label work.
- Record exact library versions (`pip freeze`) for the Methods/SI before submission.

## S5. Scope definitions
- **Disease scope:** ATC class N; EFO/MONDO nervous-system and mental-disorder branches
  (neurodegeneration, neuroimmune/MS, psychiatry, epilepsy, pain, related).
- **Unit of analysis:** (target gene, indication) pair, 6,843 pairs, 1,786 genes, 258 diseases.

---

# Part IV, Version caveats (confirm before publication)
- **gnomAD:** constraint columns present; confirm v2 vs v4 against the pull log.
- **InterProScan / InterPro2GO:** confirm the release build.
- **GTEx v8, HPA, AlphaFold DB v6:** versions as noted; confirm release dates.
- **ESMC-6B:** confirm exact model checkpoint / weights hash (recorded in the embedding job log).
- Two `seq_len`/`protein_length` and `seq_mass`/`protein_mass` pairs are near-duplicates from
  two annotation passes, deduplicate or note in Methods.
- Record exact library versions (`pip freeze`) for all software in S4.
