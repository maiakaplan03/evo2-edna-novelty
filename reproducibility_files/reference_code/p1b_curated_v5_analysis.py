#!/usr/bin/env python3
"""
P1b curated-v5 analysis driver (CODEX protocol §POSTRUN).

ONE command to refresh the entire P1 step-8 analysis on the curated v5 rebuild,
with the three mandated upgrades baked in:

  U1  group_kfold_v2 (cluster-atomic) + drop boundary=True rows.
  U2  drop the species-floor pos∩negative exact-sequence label collisions from eval.
  U3  evaluate EVERY taxonomic rank vs the shared C1-C5 negative pool -> calibration curve.

Models per (rank-vs-contamination) task, group-aware 5-fold CV over the v2 folds:
  - mean_nll  (scalar)         - total surprise
  - std_nll   (scalar)         - surprise variability
  - logistic  (full 311 profile, StandardScaler fit on TRAIN folds only)
  - tiny MLP  (optional, --with-mlp; needs torch; same val/early-stop recipe as step 8)

Outputs (all under analysis_outputs/curated_v5/):
  qc_table.csv                          per-rank n / prevalence / fold balance / NLL length
  eval_excluded_label_collisions.csv    the U2 dropped seq_ids (auditable)
  per_rank_curve.csv                    AUROC & AP@prevalence, mean±SD/fold, per rank per model
  fig_per_rank_calibration_curve.png    the HEADLINE figure (AUROC vs rank, one line per model)
  model_comparison_species_genus.csv    scalar→logistic→(MLP) at the two hard ranks
  fig_mean_nll_overlap.png              mean-NLL histograms (positives vs C1-C5) per rank
  run_summary.json                      counts, dropped rows, prevalence, timestamp

Runnable the moment per_base.jsonl exists:
  python reference_code/p1b_curated_v5_analysis.py            # scalar + logistic curve
  python reference_code/p1b_curated_v5_analysis.py --with-mlp # adds the tiny MLP (slower)

Everything except the NLL-dependent modeling is testable before scoring with --self-test,
which fabricates a small synthetic per_base so the whole pipeline can be dry-run.
"""
from __future__ import annotations

import argparse
import gzip
import json
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import curated_paths as CP  # noqa: E402


# ----------------------------------------------------------------- loaders
def _open_text(path: Path):
    """open() that transparently handles gzipped inputs (see curated_paths._resolve)."""
    path = Path(path)
    if path.suffix == ".gz":
        return gzip.open(path, "rt")
    return open(path)


def load_fasta(path: Path) -> dict[str, str]:
    seqs, name, buf = {}, None, []
    with _open_text(path) as fh:
        for line in fh:
            line = line.rstrip("\n")
            if line.startswith(">"):
                if name is not None:
                    seqs[name] = "".join(buf).upper()
                name = line[1:].split()[0]
                buf = []
            else:
                buf.append(line.strip())
    if name is not None:
        seqs[name] = "".join(buf).upper()
    return seqs


def load_per_base(path: Path) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    with _open_text(path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            sid = rec.get("id") or rec.get("seq_id")
            nll = rec.get("per_base_nll") or rec.get("nll")
            out[sid] = np.asarray(nll, dtype=np.float32)
    return out


# ------------------------------------------------------ U2: label collisions
def label_collision_ids(fasta: dict[str, str], cohort_of: dict[str, str]) -> tuple[list[str], int]:
    """seq_ids whose exact sequence is shared by a PRIMARY-positive and a negative cohort.

    Returns (all dropped seq_ids, number of distinct colliding sequence-groups). The group
    count reconciles with the audit's NOTE line; the id list is every read of those sequences
    (a single ambiguous sequence has many reads, so len(ids) > n_groups)."""
    primary = {CP.primary_cohort(r) for r in CP.ALL_POS_RANKS}
    neg = set(CP.NEG_COHORTS)
    by_seq: dict[str, list[str]] = defaultdict(list)
    for sid, s in fasta.items():
        by_seq[s].append(sid)
    drop: list[str] = []
    n_groups = 0
    for ids in by_seq.values():
        if len(ids) < 2:
            continue
        cohorts = {cohort_of.get(i, "?") for i in ids}
        if (cohorts & primary) and (cohorts & neg):
            drop.extend(ids)  # drop the whole colliding group from eval
            n_groups += 1
    return sorted(set(drop)), n_groups


# ------------------------------------------------------------- build frame
def build_frame(per_base: dict[str, np.ndarray], require_nll=True) -> tuple[pd.DataFrame, dict]:
    man = pd.read_csv(CP.MANIFEST, usecols=["seq_id", "cohort", "parent_a"], low_memory=False)
    spl = pd.read_csv(CP.SPLITS)  # seq_id, cohort, group_id, fold, boundary
    if "boundary" not in spl.columns:
        raise SystemExit("SPLITS is not group_kfold_v2 (no `boundary` column). Fix curated_paths.SPLITS.")
    df = man.merge(spl[["seq_id", "fold", "boundary", "group_id"]], on="seq_id", how="inner")

    n_before = len(df)
    # keep only the task cohorts: all illumina positive ranks + the negative pool
    pos_cohorts = {CP.primary_cohort(r) for r in CP.ALL_POS_RANKS}
    keep = pos_cohorts | set(CP.NEG_COHORTS)
    df = df[df["cohort"].isin(keep)].copy()
    df = df[~df["cohort"].isin(CP.REFERENCE_EXCLUDED)]
    df = df[~df["cohort"].str.startswith("C6_pyvolve_clean")]  # blindspot #4

    # U1: drop boundary rows
    df["boundary"] = df["boundary"].astype(str).str.lower().isin(["true", "1"])
    n_boundary = int(df["boundary"].sum())
    df = df[~df["boundary"]].copy()

    # attach NLL
    if require_nll:
        df["nll"] = df["seq_id"].map(per_base)
        missing = int(df["nll"].isna().sum())
        if missing:
            raise SystemExit(f"{missing} task rows have no per_base entry — is the Evo2 run complete? "
                             f"(expected join 1:1). Aborting.")
        bad_len = df["nll"].map(lambda a: a is None or len(a) != CP.NLL_LEN).sum()
        if bad_len:
            raise SystemExit(f"{bad_len} NLL profiles are not length {CP.NLL_LEN}.")
        df["mean_nll"] = df["nll"].map(lambda a: float(np.mean(a)))
        df["std_nll"] = df["nll"].map(lambda a: float(np.std(a)))

    df["rank"] = df["cohort"].str.replace("C6_pyvolve_", "", regex=False)
    df["is_pos"] = df["cohort"].isin(pos_cohorts)
    meta = {"n_before": n_before, "n_boundary_dropped": n_boundary}
    return df.reset_index(drop=True), meta


# ---------------------------------------------------------------- metrics
def _auroc_ap(y, s):
    from sklearn.metrics import average_precision_score, roc_auc_score
    return float(roc_auc_score(y, s)), float(average_precision_score(y, s))


def eval_task(df: pd.DataFrame, rank: str, with_mlp: bool, seeds) -> list[dict]:
    """One rank-vs-NEG binary task, group-aware 5-fold CV (folds already cluster-atomic)."""
    pos = df[df["cohort"] == CP.primary_cohort(rank)]
    neg = df[~df["is_pos"]]
    task = pd.concat([pos, neg], ignore_index=True)
    task["y"] = task["cohort"].eq(CP.primary_cohort(rank)).astype(int)
    folds = sorted(task["fold"].unique())
    rows = []

    X = None
    if not task.empty and "nll" in task.columns and task["nll"].notna().all():
        X = np.vstack(task["nll"].to_numpy()).astype(np.float32)
    yv = task["y"].to_numpy()
    fv = task["fold"].to_numpy()

    for fold in folds:
        test = fv == fold
        train = ~test
        if yv[test].sum() == 0 or yv[train].sum() == 0:
            continue
        base = {"rank": rank, "fold": int(fold), "test_n": int(test.sum()),
                "test_pos": int(yv[test].sum()), "prevalence": float(yv[test].mean())}

        # scalar baselines (orientation handled by AUROC/AP directly)
        for name, col in [("mean_nll", "mean_nll"), ("std_nll", "std_nll")]:
            au, ap = _auroc_ap(yv[test], task.loc[test, col].to_numpy())
            rows.append({**base, "model": name, "auroc": au, "ap": ap})

        # logistic on full profile (scaler fit on train only)
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
        lr = Pipeline([("sc", StandardScaler()),
                       ("lr", LogisticRegression(max_iter=4000, class_weight="balanced",
                                                 random_state=42))])
        lr.fit(X[train], yv[train])
        au, ap = _auroc_ap(yv[test], lr.predict_proba(X[test])[:, 1])
        rows.append({**base, "model": "logistic_profile", "auroc": au, "ap": ap})

        if with_mlp:
            au, ap = _mlp_fold(X, yv, fv, fold, seeds)
            rows.append({**base, "model": "tiny_mlp", "auroc": au, "ap": ap})
    return rows


def _mlp_fold(X, y, folds, test_fold, seeds):
    """Tiny MLP averaged over seeds; val = next fold, early-stop on val AP. Needs torch."""
    import torch
    from torch import nn
    from sklearn.metrics import average_precision_score, roc_auc_score

    uniq = sorted(np.unique(folds))
    val_fold = uniq[(uniq.index(test_fold) + 1) % len(uniq)]
    test = folds == test_fold
    val = folds == val_fold
    train = ~(test | val)
    mu, sd = X[train].mean(0, keepdims=True), X[train].std(0, keepdims=True) + 1e-6
    Xtr, Xva, Xte = (X[train] - mu) / sd, (X[val] - mu) / sd, (X[test] - mu) / sd
    dev = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    aus, aps = [], []
    for seed in seeds:
        torch.manual_seed(seed); np.random.seed(seed)
        net = nn.Sequential(nn.Linear(X.shape[1], 64), nn.ReLU(), nn.Dropout(0.10), nn.Linear(64, 1)).to(dev)
        opt = torch.optim.Adam(net.parameters(), lr=1e-3)
        lossf = nn.BCEWithLogitsLoss()
        xtr = torch.tensor(Xtr, device=dev); ytr = torch.tensor(y[train], dtype=torch.float32, device=dev)
        xva = torch.tensor(Xva, device=dev)
        best_ap, best_state, bad = -1, None, 0
        for _ in range(80):
            net.train(); opt.zero_grad()
            loss = lossf(net(xtr).squeeze(-1), ytr); loss.backward(); opt.step()
            net.eval()
            with torch.no_grad():
                vp = torch.sigmoid(net(xva).squeeze(-1)).cpu().numpy()
            ap = average_precision_score(y[val], vp) if y[val].sum() else 0.0
            if ap > best_ap:
                best_ap, best_state, bad = ap, {k: v.clone() for k, v in net.state_dict().items()}, 0
            else:
                bad += 1
                if bad >= 10:
                    break
        if best_state:
            net.load_state_dict(best_state)
        net.eval()
        with torch.no_grad():
            tp = torch.sigmoid(net(torch.tensor(Xte, device=dev)).squeeze(-1)).cpu().numpy()
        aus.append(roc_auc_score(y[test], tp)); aps.append(average_precision_score(y[test], tp))
    return float(np.mean(aus)), float(np.mean(aps))


# ---------------------------------------------------------------- outputs
def summarize_curve(fold_rows: list[dict]) -> pd.DataFrame:
    d = pd.DataFrame(fold_rows)
    g = (d.groupby(["rank", "model"])
           .agg(auroc_mean=("auroc", "mean"), auroc_sd=("auroc", "std"),
                ap_mean=("ap", "mean"), ap_sd=("ap", "std"),
                prevalence=("prevalence", "mean"), n_folds=("fold", "nunique"))
           .reset_index())
    g["rank"] = pd.Categorical(g["rank"], categories=CP.RANK_ORDER, ordered=True)
    return g.sort_values(["model", "rank"]).reset_index(drop=True)


def plot_curve(curve: pd.DataFrame, out_png: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ranks = [r for r in CP.RANK_ORDER if r in set(curve["rank"].astype(str))]
    fig, ax = plt.subplots(figsize=(8, 5))
    for model in sorted(curve["model"].unique()):
        sub = curve[curve["model"] == model].set_index(curve[curve["model"] == model]["rank"].astype(str))
        y = [sub.loc[r, "auroc_mean"] if r in sub.index else np.nan for r in ranks]
        e = [sub.loc[r, "auroc_sd"] if r in sub.index else np.nan for r in ranks]
        ax.errorbar(ranks, y, yerr=e, marker="o", capsize=3, label=model)
    ax.axhline(0.5, ls="--", c="grey", lw=1, label="chance")
    ax.set_xlabel("Taxonomic rank of simulated novelty (hard → ceiling)")
    ax.set_ylabel("AUROC (mean ± SD over 5 folds)")
    ax.set_title("P1b curated v5: novelty-vs-contamination detection boundary")
    ax.set_ylim(0.45, 1.02); ax.legend(fontsize=8); fig.tight_layout()
    fig.savefig(out_png, dpi=200); plt.close(fig)


def plot_mean_nll_overlap(df: pd.DataFrame, out_png: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ranks = [r for r in CP.RANK_ORDER if r in set(df["rank"])]
    neg = df[~df["is_pos"]]["mean_nll"].to_numpy()
    n = len(ranks)
    fig, axes = plt.subplots(1, n, figsize=(3 * n, 3), sharey=True)
    if n == 1:
        axes = [axes]
    for ax, r in zip(axes, ranks):
        pos = df[(df["is_pos"]) & (df["rank"] == r)]["mean_nll"].to_numpy()
        ax.hist(neg, bins=50, density=True, alpha=0.5, label="C1-C5")
        ax.hist(pos, bins=25, density=True, alpha=0.7, label=r)
        ax.set_title(r, fontsize=9); ax.set_xlabel("mean NLL")
    axes[0].set_ylabel("density"); axes[-1].legend(fontsize=7)
    fig.tight_layout(); fig.savefig(out_png, dpi=200); plt.close(fig)


def qc_table(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    neg_n = int((~df["is_pos"]).sum())
    for r in CP.RANK_ORDER:
        if r not in set(df["rank"]):
            continue
        npos = int(((df["is_pos"]) & (df["rank"] == r)).sum())
        rows.append({"rank": r, "n_pos": npos, "n_neg_pool": neg_n,
                     "prevalence_vs_pool": round(npos / (npos + neg_n), 4) if npos else 0.0})
    return pd.DataFrame(rows)


# ------------------------------------------------------------------- main
def run(with_mlp: bool, seeds, self_test: bool):
    CP.OUT_ROOT.mkdir(parents=True, exist_ok=True)
    man = pd.read_csv(CP.MANIFEST, usecols=["seq_id", "cohort"], low_memory=False)
    cohort_of = dict(zip(man["seq_id"], man["cohort"]))
    fasta = load_fasta(CP.FASTA)

    # U2 — label collisions (independent of per_base; computable before scoring)
    collisions, n_collision_groups = label_collision_ids(fasta, cohort_of)
    pd.DataFrame({"seq_id": collisions,
                  "cohort": [cohort_of.get(i, "?") for i in collisions]}
                 ).to_csv(CP.OUT_ROOT / "eval_excluded_label_collisions.csv", index=False)

    if self_test:
        rng = np.random.default_rng(0)
        per_base = {sid: rng.normal(2.0, 0.5, CP.NLL_LEN).astype(np.float32)
                    for sid in cohort_of}
        # inject faint signal so logistic/MLP have something to find
        for sid, c in cohort_of.items():
            if c.startswith("C6_pyvolve_") and "clean" not in c and "REF" not in c:
                per_base[sid] += 0.15
    else:
        if not CP.PER_BASE.exists():
            raise SystemExit(f"per_base not found: {CP.PER_BASE}\nRun Evo2 scoring (step 5) first, "
                             f"or pass --self-test to dry-run the pipeline.")
        per_base = load_per_base(CP.PER_BASE)

    df, meta = build_frame(per_base, require_nll=True)
    df = df[~df["seq_id"].isin(set(collisions))].copy()  # U2 drop

    qc = qc_table(df)
    qc.to_csv(CP.OUT_ROOT / "qc_table.csv", index=False)

    fold_rows = []
    for r in CP.ALL_POS_RANKS:
        if CP.primary_cohort(r) not in set(df["cohort"]):
            continue
        fold_rows += eval_task(df, r, with_mlp=with_mlp, seeds=seeds)
    curve = summarize_curve(fold_rows)
    curve.to_csv(CP.OUT_ROOT / "per_rank_curve.csv", index=False)
    pd.DataFrame(fold_rows).to_csv(CP.OUT_ROOT / "per_rank_folds.csv", index=False)

    plot_curve(curve, CP.OUT_ROOT / "fig_per_rank_calibration_curve.png")
    plot_mean_nll_overlap(df, CP.OUT_ROOT / "fig_mean_nll_overlap.png")

    near = curve[curve["rank"].astype(str).isin(["Species", "Genus"])]
    near.to_csv(CP.OUT_ROOT / "model_comparison_species_genus.csv", index=False)

    summary = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "self_test": self_test, "with_mlp": with_mlp,
        "rows_after_filters": int(len(df)),
        "boundary_dropped": meta["n_boundary_dropped"],
        "label_collision_groups": n_collision_groups,
        "label_collision_rows_dropped": len(collisions),
        "negative_pool_n": int((~df["is_pos"]).sum()),
        "positives_per_rank": {r: int(((df["is_pos"]) & (df["rank"] == r)).sum())
                               for r in CP.RANK_ORDER if r in set(df["rank"])},
        "models": ["mean_nll", "std_nll", "logistic_profile"] + (["tiny_mlp"] if with_mlp else []),
    }
    (CP.OUT_ROOT / "run_summary.json").write_text(json.dumps(summary, indent=2))

    print("\n=== curated v5 P1b analysis complete ===")
    print(f"  rows (after boundary+collision+clean filters): {len(df)}")
    print(f"  boundary dropped: {meta['n_boundary_dropped']} | "
          f"label-collisions: {n_collision_groups} sequences ({len(collisions)} reads dropped)")
    print(f"  outputs -> {CP.OUT_ROOT}")
    print("\nAUROC by rank (logistic_profile):")
    lp = curve[curve["model"] == "logistic_profile"][["rank", "auroc_mean", "auroc_sd", "ap_mean"]]
    print(lp.to_string(index=False))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--with-mlp", action="store_true", help="also train the tiny MLP (needs torch)")
    ap.add_argument("--seeds", default="42,123,7")
    ap.add_argument("--self-test", action="store_true",
                    help="fabricate a synthetic per_base to dry-run before scoring")
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    run(with_mlp=args.with_mlp, seeds=seeds, self_test=args.self_test)


if __name__ == "__main__":
    main()
