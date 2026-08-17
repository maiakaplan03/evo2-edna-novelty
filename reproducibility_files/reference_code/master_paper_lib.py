#!/usr/bin/env python3
"""
Master-paper engine — rigorous, reproducible P1b/P1a results for the report.

WHAT IS NEW vs p1b_curated_v5_analysis.py
-----------------------------------------
1. REPEATED group k-fold.  Instead of one fixed 5-fold partition (n=5 error bars),
   we regenerate the *cluster-atomic* v2 split several times with different seeds
   (REPEAT_SEEDS). Each repeat is a full, leak-free 5-fold CV. The headline error
   bar is then computed over all (repeat x fold) replicates, not 5 correlated folds.
   - Faithfulness: build() at seed=42 reproduces the original group_kfold_v2.csv
     exactly (fold 100%, boundary 100%, 7719 boundary). Verified 2026-06-20.
   - Atomicity is guaranteed *by construction* every repeat (build() assigns each
     sequence-cluster to one fold and drops C2/C5 boundary reads).

2. EVERY replicate row is recorded (never pre-averaged). The MLP is run per
   (repeat, fold, seed) so seed/init variance is visible, not hidden.

3. ERROR BARS = mean +/- SD across replicates (course-consistent: notes.md S1,
   "k-fold cross-validation gives error bars"). No confidence intervals, no
   significance tests (out of course scope, by design).

WHO RUNS WHAT
-------------
- Deterministic models (mean_nll, std_nll, logistic_profile): pure sklearn/numpy,
  run anywhere incl. the Cowork sandbox.
- Tiny MLP: needs torch -> run locally on the M4 (evo2-dl). Same recipe as step 8.

All outputs land in analysis_outputs/master_paper/ (accessible to the tutor).
"""
from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import curated_paths as CP            # noqa: E402
import build_curated_splits as BS     # noqa: E402
from p1b_curated_v5_analysis import (  # noqa: E402  (reuse the validated loaders)
    load_fasta, load_per_base, label_collision_ids, _auroc_ap,
)

# --------------------------------------------------------------- config
REPEAT_SEEDS = [42, 101, 202, 303, 404]    # 5 repeated CV partitions (42 == original v2)
MLP_SEEDS = [42, 123, 7, 13, 21]           # 5 init seeds per fold (MLP only)
N_SPLITS = 5
OUT = CP.OUT_ROOT.parent / "master_paper"  # analysis_outputs/master_paper


# --------------------------------------------------------------- splits
def make_repeated_splits(repeat_seeds=REPEAT_SEEDS) -> dict[int, pd.DataFrame]:
    """seed -> cluster-atomic v2 split DataFrame (seq_id, cohort, group_id, fold, boundary).

    Re-runs the validated build_curated_splits.build() per seed. Needs Biopython
    (present in evo2-dl; pip install biopython in the sandbox)."""
    man = pd.read_csv(CP.MANIFEST, low_memory=False)
    ref2c = BS.sequence_clusters(BS.DEFAULT_REFS)
    out = {}
    for s in repeat_seeds:
        _, v2 = BS.build(man.copy(), ref2c, seed=int(s), n_splits=N_SPLITS)
        out[int(s)] = v2
    return out


def assert_atomic(split_df: pd.DataFrame) -> None:
    """Hard check: among kept (non-boundary) rows, no sequence-cluster spans folds."""
    kept = split_df[~split_df["boundary"].astype(bool)].copy()

    def clusters_of(gid):
        if gid.startswith("clustpair_"):
            a, b = gid[len("clustpair_"):].split("__x__")
            return [int(a), int(b)]
        return [int(gid[len("clust_"):])]

    ek = kept.assign(c=kept["group_id"].map(clusters_of)).explode("c")
    span = ek.groupby("c")["fold"].nunique()
    n_bad = int((span > 1).sum())
    if n_bad:
        raise AssertionError(f"LEAK: {n_bad} clusters span >1 fold in a repeat — not atomic.")


# --------------------------------------------------------------- frame
def load_inputs(require_nll=True):
    """Returns (manifest_df, fasta, per_base, collisions, n_collision_groups)."""
    man = pd.read_csv(CP.MANIFEST, usecols=["seq_id", "cohort", "parent_a"], low_memory=False)
    cohort_of = dict(zip(man["seq_id"], man["cohort"]))
    fasta = load_fasta(CP.FASTA)
    collisions, n_groups = label_collision_ids(fasta, cohort_of)
    per_base = load_per_base(CP.PER_BASE) if require_nll else None
    return man, fasta, per_base, set(collisions), n_groups


def build_frame(per_base, split_df, collisions) -> pd.DataFrame:
    """Apply the v2 split + the identical filters as the validated driver:
    keep task cohorts, drop reference/clean-C6, drop boundary, attach NLL, drop collisions."""
    man = pd.read_csv(CP.MANIFEST, usecols=["seq_id", "cohort", "parent_a"], low_memory=False)
    df = man.merge(split_df[["seq_id", "fold", "boundary", "group_id"]], on="seq_id", how="inner")

    pos_cohorts = {CP.primary_cohort(r) for r in CP.ALL_POS_RANKS}
    keep = pos_cohorts | set(CP.NEG_COHORTS)
    df = df[df["cohort"].isin(keep)].copy()
    df = df[~df["cohort"].isin(CP.REFERENCE_EXCLUDED)]
    df = df[~df["cohort"].str.startswith("C6_pyvolve_clean")]

    df["boundary"] = df["boundary"].astype(str).str.lower().isin(["true", "1"])
    df = df[~df["boundary"]].copy()

    df["nll"] = df["seq_id"].map(per_base)
    if int(df["nll"].isna().sum()):
        raise SystemExit("per_base missing for some rows — is the Evo2 run complete?")
    if df["nll"].map(lambda a: a is None or len(a) != CP.NLL_LEN).sum():
        raise SystemExit(f"some NLL profiles are not length {CP.NLL_LEN}.")
    df["mean_nll"] = df["nll"].map(lambda a: float(np.mean(a)))
    df["std_nll"] = df["nll"].map(lambda a: float(np.std(a)))

    df["rank"] = df["cohort"].str.replace("C6_pyvolve_", "", regex=False)
    df["is_pos"] = df["cohort"].isin(pos_cohorts)
    df = df[~df["seq_id"].isin(collisions)].copy()
    return df.reset_index(drop=True)


# --------------------------------------------------------------- models
def eval_deterministic(df: pd.DataFrame, rank: str) -> list[dict]:
    """scalar mean_nll, std_nll, and full-profile logistic for one rank-vs-pool task."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    pos = df[df["cohort"] == CP.primary_cohort(rank)]
    neg = df[~df["is_pos"]]
    task = pd.concat([pos, neg], ignore_index=True)
    task["y"] = task["cohort"].eq(CP.primary_cohort(rank)).astype(int)
    X = np.vstack(task["nll"].to_numpy()).astype(np.float32)
    yv = task["y"].to_numpy()
    fv = task["fold"].to_numpy()
    rows = []
    for fold in sorted(task["fold"].unique()):
        test = fv == fold
        train = ~test
        if yv[test].sum() == 0 or yv[train].sum() == 0:
            continue
        base = {"rank": rank, "fold": int(fold), "test_n": int(test.sum()),
                "test_pos": int(yv[test].sum()), "prevalence": float(yv[test].mean())}
        for name, col in [("mean_nll", "mean_nll"), ("std_nll", "std_nll")]:
            au, ap = _auroc_ap(yv[test], task.loc[test, col].to_numpy())
            rows.append({**base, "model": name, "seed": -1, "auroc": au, "ap": ap})
        lr = Pipeline([("sc", StandardScaler()),
                       ("lr", LogisticRegression(max_iter=4000, class_weight="balanced",
                                                 random_state=42))])
        lr.fit(X[train], yv[train])
        au, ap = _auroc_ap(yv[test], lr.predict_proba(X[test])[:, 1])
        rows.append({**base, "model": "logistic_profile", "seed": -1, "auroc": au, "ap": ap})
    return rows


def eval_mlp(df: pd.DataFrame, rank: str, seeds=MLP_SEEDS, history_sink=None) -> list[dict]:
    """Tiny MLP per (fold, seed) — every row recorded. Needs torch (run on M4).

    history_sink: optional list. When provided, one dict per *epoch* is appended
    with (rank, fold, seed, model, epoch, train_loss, val_loss, val_ap) so the
    training-progress / early-stopping curves can be plotted (appendix figure).
    Leaving it None keeps the original behaviour and cost untouched."""
    import torch
    from torch import nn
    from sklearn.metrics import average_precision_score, roc_auc_score

    pos = df[df["cohort"] == CP.primary_cohort(rank)]
    neg = df[~df["is_pos"]]
    task = pd.concat([pos, neg], ignore_index=True)
    y = task["cohort"].eq(CP.primary_cohort(rank)).astype(int).to_numpy()
    X = np.vstack(task["nll"].to_numpy()).astype(np.float32)
    folds = task["fold"].to_numpy()
    uniq = sorted(np.unique(folds))
    dev = ("cuda" if torch.cuda.is_available()
           else "mps" if torch.backends.mps.is_available() else "cpu")
    rows = []
    for test_fold in uniq:
        val_fold = uniq[(uniq.index(test_fold) + 1) % len(uniq)]
        test = folds == test_fold
        val = folds == val_fold
        train = ~(test | val)
        if y[test].sum() == 0 or y[train].sum() == 0:
            continue
        mu, sd = X[train].mean(0, keepdims=True), X[train].std(0, keepdims=True) + 1e-6
        Xtr, Xva, Xte = (X[train] - mu) / sd, (X[val] - mu) / sd, (X[test] - mu) / sd
        for seed in seeds:
            torch.manual_seed(seed); np.random.seed(seed)
            net = nn.Sequential(nn.Linear(X.shape[1], 64), nn.ReLU(),
                                nn.Dropout(0.10), nn.Linear(64, 1)).to(dev)
            opt = torch.optim.Adam(net.parameters(), lr=1e-3)
            lossf = nn.BCEWithLogitsLoss()
            xtr = torch.tensor(Xtr, device=dev)
            ytr = torch.tensor(y[train], dtype=torch.float32, device=dev)
            xva = torch.tensor(Xva, device=dev)
            yva = torch.tensor(y[val], dtype=torch.float32, device=dev)
            # max_epochs 200 + patience 25: the 80-epoch budget truncated the MLP at the
            # hard ranks (best epoch == 80, val loss still falling, no overfitting — see figS).
            best_ap, best_state, bad = -1.0, None, 0
            for epoch in range(200):
                net.train(); opt.zero_grad()
                loss = lossf(net(xtr).squeeze(-1), ytr); loss.backward(); opt.step()
                net.eval()
                with torch.no_grad():
                    vlogits = net(xva).squeeze(-1)
                    vp = torch.sigmoid(vlogits).cpu().numpy()
                    val_loss = float(lossf(vlogits, yva).cpu())
                ap = average_precision_score(y[val], vp) if y[val].sum() else 0.0
                if history_sink is not None:
                    history_sink.append({"rank": rank, "fold": int(test_fold),
                                         "seed": int(seed), "model": "tiny_mlp",
                                         "epoch": epoch + 1, "train_loss": float(loss.detach().cpu()),
                                         "val_loss": val_loss, "val_ap": float(ap)})
                if ap > best_ap:                       # MLP selection on val-AP is well-behaved
                    best_ap = ap
                    best_state = {k: v.clone() for k, v in net.state_dict().items()}
                    bad = 0
                else:
                    bad += 1
                    if bad >= 25:
                        break
            if best_state:
                net.load_state_dict(best_state)
            net.eval()
            with torch.no_grad():
                tp = torch.sigmoid(net(torch.tensor(Xte, device=dev)).squeeze(-1)).cpu().numpy()
            rows.append({"rank": rank, "fold": int(test_fold), "test_n": int(test.sum()),
                         "test_pos": int(y[test].sum()), "prevalence": float(y[test].mean()),
                         "model": "tiny_mlp", "seed": int(seed),
                         "auroc": float(roc_auc_score(y[test], tp)),
                         "ap": float(average_precision_score(y[test], tp))})
    return rows


def eval_cnn(df: pd.DataFrame, rank: str, pool: str = "avg", seeds=MLP_SEEDS,
             history_sink=None) -> list[dict]:
    """Tiny 1-D CNN per (fold, seed) with avg or max global pooling — the pooling ablation.

    Architecture mirrors the validated p1b CNN: Conv1d(1->8)->ReLU->Conv1d(8->8)->ReLU
    ->global {avg,max} pool->Linear(8->1). Needs torch (run on M4). Every row recorded.

    history_sink: optional list; see eval_mlp. model label is f"cnn_{pool}pool"."""
    import torch
    from torch import nn
    from sklearn.metrics import average_precision_score, roc_auc_score

    pos = df[df["cohort"] == CP.primary_cohort(rank)]
    neg = df[~df["is_pos"]]
    task = pd.concat([pos, neg], ignore_index=True)
    y = task["cohort"].eq(CP.primary_cohort(rank)).astype(int).to_numpy()
    X = np.vstack(task["nll"].to_numpy()).astype(np.float32)
    folds = task["fold"].to_numpy()
    uniq = sorted(np.unique(folds))
    dev = ("cuda" if torch.cuda.is_available()
           else "mps" if torch.backends.mps.is_available() else "cpu")

    class TinyCNN(nn.Module):
        def __init__(self, pool):
            super().__init__()
            self.c1 = nn.Conv1d(1, 8, 5, padding=2); self.c2 = nn.Conv1d(8, 8, 5, padding=2)
            self.relu = nn.ReLU(); self.pool = pool; self.head = nn.Linear(8, 1)
        def forward(self, x):
            h = self.relu(self.c2(self.relu(self.c1(x))))           # (B,8,L)
            h = h.max(dim=-1).values if self.pool == "max" else h.mean(dim=-1)
            return self.head(h).squeeze(-1)

    rows = []
    for test_fold in uniq:
        val_fold = uniq[(uniq.index(test_fold) + 1) % len(uniq)]
        test = folds == test_fold; val = folds == val_fold; train = ~(test | val)
        if y[test].sum() == 0 or y[train].sum() == 0:
            continue
        mu, sd = X[train].mean(0, keepdims=True), X[train].std(0, keepdims=True) + 1e-6
        Xtr, Xva, Xte = (X[train]-mu)/sd, (X[val]-mu)/sd, (X[test]-mu)/sd
        for seed in seeds:
            torch.manual_seed(seed); np.random.seed(seed)
            net = TinyCNN(pool).to(dev)
            opt = torch.optim.Adam(net.parameters(), lr=1e-3)
            lossf = nn.BCEWithLogitsLoss()
            xtr = torch.tensor(Xtr, device=dev).unsqueeze(1)   # (B,1,L)
            ytr = torch.tensor(y[train], dtype=torch.float32, device=dev)
            xva = torch.tensor(Xva, device=dev).unsqueeze(1)
            yva = torch.tensor(y[val], dtype=torch.float32, device=dev)
            # CNN selects on val-LOSS, not val-AP: with so few positives the validation AP is
            # too noisy and locked the model at an ~epoch-1 (untrained) state. Loss is smoother.
            # max_epochs 200 + patience 25 to match the MLP budget.
            best_val, best_state, bad = float("inf"), None, 0
            for epoch in range(200):
                net.train(); opt.zero_grad()
                loss = lossf(net(xtr), ytr); loss.backward(); opt.step()
                net.eval()
                with torch.no_grad():
                    vlogits = net(xva)
                    vp = torch.sigmoid(vlogits).cpu().numpy()
                    val_loss = float(lossf(vlogits, yva).cpu())
                ap = average_precision_score(y[val], vp) if y[val].sum() else 0.0
                if history_sink is not None:
                    history_sink.append({"rank": rank, "fold": int(test_fold),
                                         "seed": int(seed), "model": f"cnn_{pool}pool",
                                         "epoch": epoch + 1, "train_loss": float(loss.detach().cpu()),
                                         "val_loss": val_loss, "val_ap": float(ap)})
                if val_loss < best_val - 1e-4:
                    best_val = val_loss; best_state = {k: v.clone() for k, v in net.state_dict().items()}; bad = 0
                else:
                    bad += 1
                    if bad >= 25:
                        break
            if best_state:
                net.load_state_dict(best_state)
            net.eval()
            with torch.no_grad():
                tp = torch.sigmoid(net(torch.tensor(Xte, device=dev).unsqueeze(1))).cpu().numpy()
            rows.append({"rank": rank, "fold": int(test_fold), "model": f"cnn_{pool}pool",
                         "seed": int(seed), "prevalence": float(y[test].mean()),
                         "auroc": float(roc_auc_score(y[test], tp)),
                         "ap": float(average_precision_score(y[test], tp))})
    return rows


# --------------------------------------------------------------- driver
def run(with_mlp=False, repeat_seeds=REPEAT_SEEDS, mlp_seeds=MLP_SEEDS,
        ranks=None, out_dir: Path = OUT) -> pd.DataFrame:
    """Full repeated-CV sweep. Writes per-replicate rows + mean+/-SD summary."""
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    ranks = ranks or CP.ALL_POS_RANKS
    _, _, per_base, collisions, n_groups = load_inputs(require_nll=True)
    splits = make_repeated_splits(repeat_seeds)

    all_rows = []
    for rep_seed, split_df in splits.items():
        assert_atomic(split_df)
        df = build_frame(per_base, split_df, collisions)
        for r in ranks:
            if CP.primary_cohort(r) not in set(df["cohort"]):
                continue
            det = eval_deterministic(df, r)
            for row in det:
                row["repeat_seed"] = rep_seed
            all_rows += det
            if with_mlp:
                mlp = eval_mlp(df, r, seeds=mlp_seeds)
                for row in mlp:
                    row["repeat_seed"] = rep_seed
                all_rows += mlp

    folds = pd.DataFrame(all_rows)
    folds.to_csv(out_dir / "per_rank_folds_repeated.csv", index=False)
    summ = summarize(folds)
    summ.to_csv(out_dir / "per_rank_summary.csv", index=False)

    meta = {"timestamp": datetime.now().isoformat(timespec="seconds"),
            "repeat_seeds": list(map(int, repeat_seeds)),
            "mlp_seeds": list(map(int, mlp_seeds)) if with_mlp else [],
            "with_mlp": with_mlp, "n_splits": N_SPLITS,
            "label_collision_groups": int(n_groups),
            "replicate_definition": "deterministic: repeat x fold; mlp: repeat x fold x seed",
            "error_bar": "mean +/- SD across replicates (notes.md S1)"}
    (out_dir / "run_meta.json").write_text(json.dumps(meta, indent=2))
    return summ


def summarize(folds: pd.DataFrame) -> pd.DataFrame:
    """mean +/- SD over replicate rows; n = number of replicates per (rank, model)."""
    g = (folds.groupby(["rank", "model"])
         .agg(auroc_mean=("auroc", "mean"), auroc_sd=("auroc", "std"),
              ap_mean=("ap", "mean"), ap_sd=("ap", "std"),
              prevalence=("prevalence", "mean"), n_replicates=("auroc", "size"))
         .reset_index())
    g["rank"] = pd.Categorical(g["rank"], categories=CP.RANK_ORDER, ordered=True)
    return g.sort_values(["model", "rank"]).reset_index(drop=True)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--with-mlp", action="store_true")
    ap.add_argument("--repeats", default=",".join(map(str, REPEAT_SEEDS)))
    ap.add_argument("--mlp-seeds", default=",".join(map(str, MLP_SEEDS)))
    a = ap.parse_args()
    s = run(with_mlp=a.with_mlp,
            repeat_seeds=[int(x) for x in a.repeats.split(",") if x.strip()],
            mlp_seeds=[int(x) for x in a.mlp_seeds.split(",") if x.strip()])
    print(s.to_string(index=False))
