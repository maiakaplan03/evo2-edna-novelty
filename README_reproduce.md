# DL Project — Hand-in & Reproducibility Bundle

**Author:** Maia Kaplan
**Contents of this Drive folder:**

1. `project_paper_Kaplan_Maia.pdf` — the written report.
2. `DL_project_handin_Kaplan_Maia.ipynb` — the hand-in notebook (saved **with all outputs**, so you can read every result and figure without running anything).
3. `reproducibility_files/` — a self-contained copy of everything needed to **re-execute** the notebook from scratch.

---

## Just want to read the results?

Open `DL_project_handin_Kaplan_Maia.ipynb` (top level). All cell outputs, tables, and figures are saved in the file — nothing needs to run.

## Want to re-execute from scratch?

Run the **copy inside the bundle**, not the top-level one:

```
reproducibility_files/notebooks/DL_project_handin_Kaplan_Maia.ipynb
```

This matters because the notebook resolves all data paths relative to its folder
(`ROOT = notebook_dir/..`). The top-level copy is for viewing only; the one inside
`reproducibility_files/notebooks/` sits in the correct layout so every relative path
resolves. Open it from that location (or set the Jupyter working dir there) and
"Run All".

### Environment

Python 3.10+ with:

```
pip install numpy pandas matplotlib scikit-learn biopython torch
```

- `biopython` is **required** — the notebook rebuilds the cross-validation splits
  via `make_repeated_splits()`, which reads the reference FASTA with `Bio.SeqIO`.
- `torch` is needed only to **retrain** the MLP/CNN. If torch is absent, those cells
  automatically fall back to the saved result CSVs in
  `analysis_outputs/master_paper/` and the figures still reproduce.

### What's in `reproducibility_files/` (layout matters — keep it intact)

```
reproducibility_files/
  notebooks/
    DL_project_handin_Kaplan_Maia.ipynb      <- run THIS one
  reference_code/                            <- 4 local modules the notebook imports
    master_paper_lib.py
    curated_paths.py                         <- single source of truth for all paths
    build_curated_splits.py
    p1b_curated_v5_analysis.py
  analysis_outputs/
    master_paper/                            <- precomputed result CSVs + run_meta.json
    curated_v5/
      multiclass/   confusion_A/B_mlp_balanced.csv
      depth_ablation/ mlp_depth_ablation_summary.csv
  data_generation/
    2_build_contamination_cohorts/phase1_output_curated_v5/
      phase1_manifest.csv.gz                 <- read manifest
      phase1_all_curated_v5.fasta.gz         <- combined references + reads
      splits/group_kfold_v2.csv.gz
    1_simulate_novelty/references/
      arthropod_curated_v5.fasta             <- reference set used to build CV clusters
  lt_evo2_evo2_results_curated/curated_v5_p1b/
      per_base.jsonl                         <- 167 MB Evo2 per-base NLL profiles
```

> Note: in this repository the three large inputs are stored **gzipped**
> (`per_base.jsonl.gz` 37 MB, `phase1_all_curated_v5.fasta.gz` 2.1 MB,
> `phase1_manifest.csv.gz` 0.5 MB) so the repo stays clonable. Do **not** unpack them
> by hand — `curated_paths._resolve()` picks the `.gz` automatically and the loaders
> decompress on read (~1.5 s). An uncompressed copy, if present, is preferred silently,
> so a local working tree with the raw files behaves identically.
