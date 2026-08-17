"""
Single source of truth for the curated v5 rebuild dataset (CODEX protocol §POSTRUN).

Import this everywhere instead of re-hardcoding paired-v4 paths, so the four
analysis consumers can never drift apart. Switching datasets = edit this file only.
"""
from __future__ import annotations
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _resolve(path: Path) -> Path:
    """Return `path` if it exists, else its gzipped twin `path.gz` if THAT exists.

    The public repo ships the large inputs gzipped (a 167 MB per_base.jsonl does not
    belong in git); a local working tree may still hold them uncompressed. Both work,
    and callers never have to care which one is on disk.
    """
    if path.exists():
        return path
    gz = path.with_suffix(path.suffix + ".gz")
    return gz if gz.exists() else path

INPUT_DIR = ROOT / "data_generation/2_build_contamination_cohorts/phase1_output_curated_v5"
RESULT_DIR = ROOT / "lt_evo2_evo2_results_curated/curated_v5_p1b"

MANIFEST = _resolve(INPUT_DIR / "phase1_manifest.csv")


def _resolve_fasta() -> Path:
    """The combined FASTA. build_phase1_cohorts writes `phase1_all.fasta`, but it may have
    been renamed (e.g. `phase1_all_curated_v5.fasta` for the Lightning upload)."""
    for name in ("phase1_all.fasta", "phase1_all_curated_v5.fasta"):
        hit = _resolve(INPUT_DIR / name)
        if hit.exists():
            return hit
    hits = sorted(INPUT_DIR.glob("phase1_all*.fasta*"))
    return hits[0] if hits else INPUT_DIR / "phase1_all.fasta"


FASTA = _resolve_fasta()
SPLITS = _resolve(INPUT_DIR / "splits/group_kfold_v2.csv")      # U1: cluster-atomic (has `boundary`)
PER_BASE = _resolve(RESULT_DIR / "per_base.jsonl")

OUT_ROOT = ROOT / "analysis_outputs/curated_v5"        # NEW tree — never overwrite paired_v4

# Taxonomic gradient (U3). Curve ranks vs the shared negative pool; controls are anchors.
RANKS = ["Species", "Genus", "Family", "Order", "Phylum"]
CONTROL_RANKS = ["Novelty", "Noise"]
ALL_POS_RANKS = RANKS + CONTROL_RANKS

# Display/curve order from hardest (most overlap with contamination) to ceiling.
RANK_ORDER = ["Species", "Genus", "Family", "Order", "Phylum", "Novelty", "Noise"]

# Headline "P1b" near-novel task = Species+Genus combined.
PRIMARY_NEAR = ["C6_pyvolve_Species", "C6_pyvolve_Genus"]

# Negatives = the five contamination modes (C1-C5; C4 has three damage tiers).
NEG_COHORTS = [
    "C1_illumina", "C2_chimera", "C3_insert",
    "C4_briggs_low", "C4_briggs_mid", "C4_briggs_high", "C5_stacked",
]

# Never train/eval on these: clean reference anchors and ALL clean C6 rows
# (no error layer -> would teach "error-free = novel", blindspot #4).
REFERENCE_EXCLUDED = {"C0_REF", "C6_pyvolve_REF", "C6_pyvolve_clean_REF"}

NLL_LEN = 311  # per-base NLL profile length for a 312-bp read


def primary_cohort(rank: str) -> str:
    """Illumina (primary) positive cohort for a taxonomic rank."""
    return f"C6_pyvolve_{rank}"
