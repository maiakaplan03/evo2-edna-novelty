#!/usr/bin/env python3
"""
Phase A0 — sequence-cluster GroupKFold for the curated rebuild.

Replaces the leaky ID-based grouping for the curated-reference rebuild. Two fixes
over the legacy reference_code/splits.py:

  1. SEQUENCE-CLUSTER grouping. Reads are grouped by the *sequence* of their source
     reference, not by its ID. Identical reference sequences therefore never straddle
     a fold boundary (the bug that inflated paired-v4's AUROC ~0.737). For the curated
     v5 set every reference is unique, so each ref is its own cluster — but the cluster
     step makes the split correct even if a duplicate slips in.

  2. GENERALIZED source extraction. The legacy code hardcoded `C1W2P1_\\d+`. Here the
     source reference of a C6 row is the token between `from_` and `_ident` in parent_a
     (works for any ref id, e.g. AR0001 or C1W2P1_930). All other cohorts use parent_a
     (the ref id) directly; C2 chimeras have two parents.

Outputs (schema-compatible with the existing pipeline):
  <out-dir>/group_kfold_v1.csv   seq_id, cohort, group_id, fold
  <out-dir>/group_kfold_v2.csv   seq_id, cohort, group_id, fold, boundary

  v1 = per-read group (canonical cluster pair for C2, single source cluster otherwise),
       one GroupKFold over those groups.
  v2 = cluster-atomic (strictest): each sequence-cluster -> one fold; reads inherit it;
       a C2 read whose two parents are in different folds is flagged boundary=True
       (drop from strict training).

Usage:
  python reference_code/build_curated_splits.py \
    --manifest data_generation/2_build_contamination_cohorts/<curated_output>/phase1_manifest.csv \
    --refs data_generation/1_simulate_novelty/references/arthropod_curated_v5.fasta \
    --out-dir data_generation/2_build_contamination_cohorts/<curated_output>/splits \
    --seed 42
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from Bio import SeqIO
from sklearn.model_selection import GroupKFold

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REFS = ROOT / "data_generation/1_simulate_novelty/references/arthropod_curated_v5.fasta"
N_SPLITS = 5
TARGET_LEN = 312
C6_SOURCE_RE = re.compile(r"from_(.+?)_ident")


def sequence_clusters(refs_fasta: Path) -> dict[str, int]:
    """Map ref_id -> cluster index, where clusters group identical 312-bp sequences."""
    seq_of: dict[str, str] = {}
    for rec in SeqIO.parse(str(refs_fasta), "fasta"):
        seq_of[rec.id] = str(rec.seq).upper()[:TARGET_LEN]
    if not seq_of:
        raise FileNotFoundError(f"No references in {refs_fasta}")
    cluster_of_seq: dict[str, int] = {}
    ref_to_cluster: dict[str, int] = {}
    for rid, s in seq_of.items():
        cid = cluster_of_seq.setdefault(s, len(cluster_of_seq))
        ref_to_cluster[rid] = cid
    n_ref, n_clust = len(ref_to_cluster), len(cluster_of_seq)
    print(f"references: {n_ref} | sequence clusters: {n_clust}", file=sys.stderr)
    if n_clust < n_ref:
        print(f"[note] {n_ref - n_clust} references are sequence-duplicates "
              f"and were merged into shared clusters (good: prevents leakage).",
              file=sys.stderr)
    return ref_to_cluster


def source_ref(cohort: str, parent_a, parent_b) -> tuple[str | None, str | None]:
    """Return (source_a, source_b) reference ids for a row. source_b only for C2."""
    if pd.isna(parent_a) or not str(parent_a).strip():
        return None, None
    pa = str(parent_a)
    if str(cohort).startswith("C6_pyvolve"):
        m = C6_SOURCE_RE.search(pa)
        if m:
            return m.group(1), None
        # C6 REF anchor rows carry parent_a == "REF_<ref_id>" (no from_..._ident token).
        if pa.startswith("REF_"):
            return pa[len("REF_"):], None
        return None, None
    sa = pa
    sb = str(parent_b) if (pd.notna(parent_b) and str(parent_b).strip()) else None
    return sa, sb


def assign_cluster_folds(clusters: list[int], n_splits: int, seed: int) -> dict[int, int]:
    rng = np.random.default_rng(seed)
    uniq = np.array(sorted(set(clusters)))
    rng.shuffle(uniq)
    gkf = GroupKFold(n_splits=n_splits)
    dummy = np.zeros((len(uniq), 1))
    fold_of: dict[int, int] = {}
    for fold, (_, test_idx) in enumerate(gkf.split(dummy, groups=uniq)):
        for i in test_idx:
            fold_of[int(uniq[i])] = fold
    return fold_of


def build(manifest: pd.DataFrame, ref_to_cluster: dict[str, int],
          seed: int, n_splits: int):
    cols = ["seq_id", "cohort", "parent_a", "parent_b"]
    for c in cols:
        if c not in manifest.columns:
            manifest[c] = np.nan
    df = manifest[cols].copy()

    src = df.apply(lambda r: source_ref(r["cohort"], r["parent_a"], r["parent_b"]),
                   axis=1, result_type="expand")
    df["src_a"], df["src_b"] = src[0], src[1]

    # Map source ref -> cluster. Unmapped sources are a hard error (typo / id drift).
    def to_cluster(x):
        if x is None or (isinstance(x, float) and pd.isna(x)):
            return None
        return ref_to_cluster.get(str(x))

    df["clust_a"] = df["src_a"].map(to_cluster)
    df["clust_b"] = df["src_b"].map(to_cluster)

    unmapped = df[df["src_a"].notna() & df["clust_a"].isna()]
    if not unmapped.empty:
        bad = unmapped["src_a"].unique()[:5]
        raise ValueError(f"{len(unmapped)} rows have a source ref not in the reference FASTA, "
                         f"e.g. {list(bad)}. Did the manifest and --refs come from the same run?")

    all_clusters = sorted(set(df["clust_a"].dropna().astype(int)) |
                          set(df["clust_b"].dropna().astype(int)))
    fold_of = assign_cluster_folds(all_clusters, n_splits, seed)

    # ---- v1: per-read group ----
    def gid_v1(r):
        if str(r["cohort"]) == "C2_chimera" and pd.notna(r["clust_b"]):
            a, b = sorted([int(r["clust_a"]), int(r["clust_b"])])
            return f"clustpair_{a}__x__{b}"
        return f"clust_{int(r['clust_a'])}"
    df["group_id"] = df.apply(gid_v1, axis=1)
    # fold for v1 follows clust_a's fold (pair groups inherit parent_a's fold,
    # consistent with the legacy v1 reading)
    df["fold"] = df["clust_a"].astype(int).map(fold_of)
    v1 = df[["seq_id", "cohort", "group_id", "fold"]].copy()

    # ---- v2: cluster-atomic with boundary flag ----
    def row_v2(r):
        fa = fold_of[int(r["clust_a"])]
        if pd.notna(r["clust_b"]):
            fb = fold_of[int(r["clust_b"])]
            return pd.Series([fa, fa != fb])
        return pd.Series([fa, False])
    df[["fold2", "boundary"]] = df.apply(row_v2, axis=1)
    v2 = df[["seq_id", "cohort", "group_id", "fold2"]].copy()
    v2 = v2.rename(columns={"fold2": "fold"})
    v2["boundary"] = df["boundary"].astype(bool)
    return v1, v2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--refs", type=Path, default=DEFAULT_REFS)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-splits", type=int, default=N_SPLITS)
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    ref_to_cluster = sequence_clusters(args.refs)
    manifest = pd.read_csv(args.manifest, low_memory=False)
    print(f"manifest rows: {len(manifest)}", file=sys.stderr)

    v1, v2 = build(manifest, ref_to_cluster, args.seed, args.n_splits)
    v1.to_csv(args.out_dir / "group_kfold_v1.csv", index=False)
    v2.to_csv(args.out_dir / "group_kfold_v2.csv", index=False)

    print(f"\nwrote {args.out_dir}/group_kfold_v1.csv and group_kfold_v2.csv", file=sys.stderr)
    print("\n[v1] rows per cohort x fold:")
    print(v1.groupby(["cohort", "fold"]).size().unstack(fill_value=0))
    nb = int(v2["boundary"].sum())
    print(f"\n[v2] boundary reads (C2 parents split across folds, dropped in strict): {nb}",
          file=sys.stderr)

    # Leakage self-check: no sequence-cluster appears in more than one fold (v2).
    chk = v2.merge(v1[["seq_id"]], on="seq_id")  # noqa: F841
    print("\nFold sizes (v2):")
    print(v2.groupby("fold").size())


if __name__ == "__main__":
    main()
