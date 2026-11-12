"""Loading and checking the saved model outputs (contract D).

Every run has one .npz per test condition with arrays:
    image_id (N,)  y_true (N,)  logits (N, 4)  features (N, D)  mc_logits (N, 20, 4)
Row order matches the condition's split CSV. Patient, country and machine are
joined from the split files by image_id.

All floating-point arrays are converted to double precision (float64) on
load, so every script computes with the same precision.
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
        out = {k: z[k] for k in z.files}
    for k in ("logits", "features", "mc_logits"):
        if k in out:
            out[k] = out[k].astype(np.float64)
    return out


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


# ---- duplicates and evaluation rows ------------------------------------------
# splits/duplicates.csv (from scripts/find_duplicates.py) lists every image that
# copies an image of a DIFFERENT patient: image_id, split, duplicate_of,
# match_type, merge_into_patient_id. The listed image is the copy; the original
# is the copy from the earliest split, so a validation image pointing back to
# training is a leak.
#
# Evaluation rule (same everywhere, so T6 and T7 cannot diverge):
#   - rows listed as copies in their own split are dropped (the original stays)
#   - patients that share images are merged, so the bootstrap treats them as
#     one person
# The frozen split CSVs are never edited.

DUPLICATES_FILE = ROOT / "splits" / "duplicates.csv"


def load_duplicates(path: Path | None = DUPLICATES_FILE) -> pd.DataFrame | None:
    """The duplicates table, or None if deduplication is switched off or the file is missing."""
    if path is None:
        return None
    path = Path(path)
    if not path.exists():
        print(f"WARNING: {path} not found, evaluating all images without deduplication")
        return None
    return pd.read_csv(path, keep_default_na=False)


def evaluation_rows(split_file: str, split: pd.DataFrame,
                    duplicates: pd.DataFrame | None) -> tuple[np.ndarray, np.ndarray]:
    """Row indices to evaluate and the patient ID of each row (after merging).

    split_file: e.g. "africa_test.csv"; every condition built on this split
                (clean, compressed, masked) gets the same rows.
    """
    patients = split["patient_id"].astype(str).to_numpy().copy()
    if duplicates is None or duplicates.empty:
        return np.arange(len(split)), patients
    name = Path(split_file).stem
    copies = duplicates[duplicates["split"] == name]
    keep = ~split["image_id"].astype(str).isin(set(copies["image_id"].astype(str)))

    # merge patients: every patient that owns a copy joins the original's patient
    merge = {}
    by_image = split.set_index(split["image_id"].astype(str))["patient_id"].astype(str)
    for row in duplicates.itertuples():
        target = str(row.merge_into_patient_id)
        if not target:
            continue
        if row.split == name and str(row.image_id) in by_image.index:
            merge[by_image[str(row.image_id)]] = target
    for _ in range(10):                                     # follow chains a -> b -> c
        changed = False
        for src, dst in list(merge.items()):
            if dst in merge and merge[dst] != dst:
                merge[src] = merge[dst]
                changed = True
        if not changed:
            break
    patients = np.array([merge.get(p, p) for p in patients])
    return np.flatnonzero(keep.to_numpy()), patients


def validation_condition(condition_id: str) -> str:
    """Masked test conditions are calibrated on masked validation images; all others on spain_val."""
    return "spain_val_masked" if condition_id.endswith("_masked") else "spain_val"


def reference_condition(condition_id: str) -> str:
    """Familiar images for OOD AUROC: the clean (or masked) Spanish test set."""
    return "spain_test_masked" if condition_id.endswith("_masked") else "spain_test"


def is_validation(condition_id: str) -> bool:
    return condition_id in ("spain_train",) or condition_id.startswith("spain_val")


def subset(out: dict, idx: np.ndarray) -> dict:
    """Restrict every per-image array of a loaded output to rows idx."""
    return {k: (v[idx] if isinstance(v, np.ndarray) and v.ndim >= 1 and len(v) == len(out["y_true"])
                else v) for k, v in out.items()}


# ---- partial runs ------------------------------------------------------------
# Not every run has every condition (e.g. quantized runs such as effb0-int8_s0
# have only a few). Each script evaluates, per model variant, the conditions
# that ALL of that variant's seeds have in the manifest, so the ensemble always
# combines the same images.

def conditions_of(manifest: pd.DataFrame, variant: str) -> set:
    """Conditions present for every run of a variant."""
    v = manifest[manifest["variant"] == variant]
    per_run = v.groupby("run_id")["condition_id"].apply(set)
    return set.intersection(*per_run) if len(per_run) else set()
