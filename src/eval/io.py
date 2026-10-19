"""Loading and checking the saved model outputs (contract D).

Every run has one .npz per test condition with arrays:
    image_id (N,)  y_true (N,)  logits (N, 4)  features (N, D)  mc_logits (N, 20, 4)
Row order matches the condition's split CSV. Patient, country and machine are
joined from the split files by image_id.
"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
N_CLASSES = 4
MC_PASSES = 20
REQUIRED = ("image_id", "y_true", "logits", "features", "mc_logits")


def variant_of(run_id: str) -> str:
    """'dinov3s_frozen_s3' -> 'dinov3s_frozen'. The run ID is the unambiguous name."""
    return run_id.rsplit("_s", 1)[0]


def load_manifest(outputs_root: Path) -> pd.DataFrame:
    man = pd.read_csv(outputs_root / "manifest.csv")
    man = man[~man["run_id"].str.startswith("_")]          # ignore smoke tests
    man["variant"] = man["run_id"].map(variant_of)
    return man


def load_conditions() -> pd.DataFrame:
    return pd.read_csv(ROOT / "splits" / "conditions.csv", keep_default_na=False)


def load_split(split_file: str) -> pd.DataFrame:
    return pd.read_csv(ROOT / "splits" / split_file, keep_default_na=False)


def load_output(outputs_root: Path, run_id: str, condition_id: str) -> dict:
    # allow_pickle: image_id may be saved as an object array; these are our own files
    with np.load(outputs_root / run_id / f"{condition_id}.npz", allow_pickle=True) as z:
        return {k: z[k] for k in z.files}


def check_output(out: dict, split: pd.DataFrame) -> list[str]:
    """Return a list of problems (empty if the file matches contract D)."""
    problems = []
    missing = [k for k in REQUIRED if k not in out]
    if missing:
        return [f"missing arrays: {missing}"]
    n = len(split)
    if len(out["image_id"]) != n:
        problems.append(f"N={len(out['image_id'])}, split has {n}")
    elif not np.array_equal(out["image_id"].astype(str), split["image_id"].astype(str).values):
        problems.append("image_id order differs from split CSV")
    elif not np.array_equal(out["y_true"].astype(int), split["label"].astype(int).values):
        problems.append("y_true differs from split labels")
    if out["logits"].shape != (n, N_CLASSES):
        problems.append(f"logits shape {out['logits'].shape}")
    if out["features"].ndim != 2 or out["features"].shape[0] != n:
        problems.append(f"features shape {out['features'].shape}")
    if out["mc_logits"].shape != (n, MC_PASSES, N_CLASSES):
        problems.append(f"mc_logits shape {out['mc_logits'].shape}")
    for k in ("logits", "features", "mc_logits"):
        if not np.isfinite(out[k]).all():
            problems.append(f"non-finite values in {k}")
    return problems
