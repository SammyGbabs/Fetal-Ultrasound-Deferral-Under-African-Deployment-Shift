"""Masked versus original images, with patient-level 95% confidence intervals.

For the Spanish and African test sets, compares each model on the original
images with the same images after on-screen text was masked (Akachi's masked
conditions). Both versions are scored on the SAME patient resamples, so the
difference "masked minus original" gets its own interval.

Rules follow T6/T7 exactly:
  - original conditions: temperature and deferral thresholds from spain_val
  - masked conditions:   from spain_val_masked
  - duplicate copies dropped, duplicated patients merged, leaked validation
    images removed (splits/duplicates.csv)
Methods: single model (MSP, averaged over seeds) and the 5-seed ensemble,
without temperature scaling. African results are also given without Ghana and
Uganda (calipers inside the fan).

Resumable per model variant. Writes eval/masking_ci.csv.

    !python scripts/run_masking_ci.py --outputs /content/outputs_local
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import softmax

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.eval import bootstrap, calibration, io, scores  # noqa: E402
from src.paths import DRIVE_ROOT  # noqa: E402

parser = argparse.ArgumentParser()
parser.add_argument("--outputs", default=str(DRIVE_ROOT / "outputs"))
parser.add_argument("--eval-dir", default=str(DRIVE_ROOT / "eval"))
parser.add_argument("--duplicates", default=str(io.DUPLICATES_FILE))
parser.add_argument("--B", type=int, default=2000, help="bootstrap resamples")
parser.add_argument("--redo", action="store_true")
args = parser.parse_args()
outputs_root, eval_dir = Path(args.outputs), Path(args.eval_dir)
progress_dir = eval_dir / "masking_per_variant"
progress_dir.mkdir(parents=True, exist_ok=True)

PAIRS = [("spain_test", "spain_test_masked"), ("africa_test", "africa_test_masked")]
METRICS = ["balanced_accuracy", "silent_failure", "deployed_coverage_80", "errors_passed_80"]
CALIPER_COUNTRIES = {"Ghana", "Uganda"}

manifest = io.load_manifest(outputs_root)
conditions = io.load_conditions().set_index("condition_id")
missing = [c for pair in PAIRS for c in pair if c not in conditions.index] + \
          [c for c in ("spain_val", "spain_val_masked") if c not in conditions.index]
if missing:
    sys.exit(f"conditions.csv is missing {missing}; pull Akachi's masking changes first")
duplicates = io.load_duplicates(args.duplicates)
rows_and_patients = {f: io.evaluation_rows(f, io.load_split(f), duplicates)
                     for f in conditions["split_file"].unique()}
val_keep = rows_and_patients[conditions.loc["spain_val", "split_file"]][0]


def logits_only(run_id, condition):
    with np.load(outputs_root / run_id / f"{condition}.npz", allow_pickle=True) as z:
        return z["logits"].astype(np.float64), z["y_true"].astype(int)


def groups_for(base: str, split: pd.DataFrame, keep: np.ndarray) -> dict:
    kept = np.zeros(len(split), bool)
    kept[keep] = True
    g = {f"{base}_all": np.flatnonzero(kept)}
    if base == "africa_test":
        g["africa_test_caliper_free"] = np.flatnonzero(kept & ~split.country.isin(CALIPER_COUNTRIES))
    return g


for variant, vman in manifest.groupby("variant"):
    out_file = progress_dir / f"{variant}.csv"
    if out_file.exists() and not args.redo:
        print(f"skip {variant} (already done)")
        continue
    run_ids = sorted(vman.run_id.unique())
    available = io.conditions_of(manifest, variant)
    v_pairs = [(b, m) for b, m in PAIRS if b in available and m in available]
    if not {"spain_val", "spain_val_masked"} <= available or not v_pairs:
        print(f"skip {variant}: no masked outputs")
        continue

    def predictions(logits_by_run):
        """(pred, confidence, uncertainty) per method and seed, raw (no temperature scaling)."""
        out = {}
        for r in run_ids:
            pred, u = scores.msp(logits_by_run[r])
            out[("msp", r)] = (pred, softmax(logits_by_run[r], axis=1).max(1), u)
        ens = [logits_by_run[r] for r in run_ids]
        pred, u, _ = scores.ensemble(ens)
        mean = np.mean([softmax(l, axis=1) for l in ens], axis=0)
        out[("ensemble", "ens")] = (pred, mean.max(1), u)
        return out

    # thresholds: original from spain_val, masked from spain_val_masked (leaked images removed)
    thresholds = {}
    for vc in ("spain_val", "spain_val_masked"):
        val = predictions({r: logits_only(r, vc)[0][val_keep] for r in run_ids})
        thresholds[vc] = {k: float(np.quantile(v[2], 0.8)) for k, v in val.items()}

    rows = []
    for base, masked in v_pairs:
        split = io.load_split(conditions.loc[base, "split_file"])
        keep_rows, patients = rows_and_patients[conditions.loc[base, "split_file"]]
        y = logits_only(run_ids[0], base)[1]
        preds = {"original": predictions({r: logits_only(r, base)[0] for r in run_ids}),
                 "masked": predictions({r: logits_only(r, masked)[0] for r in run_ids})}
        thr = {"original": thresholds["spain_val"], "masked": thresholds["spain_val_masked"]}

        for group, rows_idx in groups_for(base, split, keep_rows).items():
            sampler = bootstrap.PatientSampler(patients[rows_idx])

            def evaluate(idx):
                res = {}
                for version in ("original", "masked"):
                    for method in ("msp", "ensemble"):
                        keys = [k for k in preds[version] if k[0] == method]
                        per_seed = [bootstrap.headline_metrics(
                            y[idx], *(a[idx] for a in preds[version][k]), thr[version][k])
                            for k in keys]
                        res[(version, method)] = {m: np.nanmean([s[m] for s in per_seed])
                                                  for m in METRICS}
                return res

            point = evaluate(rows_idx)
            boot = {key: {m: [] for m in METRICS} for key in point}
            for _ in range(args.B):
                b = evaluate(rows_idx[sampler.sample()])
                for key in b:
                    for m in METRICS:
                        boot[key][m].append(b[key][m])

            for method in ("msp", "ensemble"):
                for m in METRICS:
                    orig, mask = point[("original", method)][m], point[("masked", method)][m]
                    diff = np.array(boot[("masked", method)][m]) - np.array(boot[("original", method)][m])
                    diff = diff[np.isfinite(diff)]
                    lo, hi = (np.percentile(diff, [2.5, 97.5]) if len(diff) else (np.nan, np.nan))
                    rows.append({"variant": variant, "method": method, "group": group, "metric": m,
                                 "original": orig, "masked": mask, "difference": mask - orig,
                                 "ci_low": lo, "ci_high": hi,
                                 "significant": bool(lo > 0 or hi < 0),
                                 "n_images": len(rows_idx), "n_patients": sampler.n_patients})
        print(f"    {variant} {base} done")

    pd.DataFrame(rows).to_csv(out_file.with_suffix(".tmp"), index=False)
    out_file.with_suffix(".tmp").rename(out_file)
    print(f"ok  {variant}")

variants = [v for v in sorted(manifest.variant.unique()) if (progress_dir / f"{v}.csv").exists()]
result = pd.concat([pd.read_csv(progress_dir / f"{v}.csv") for v in variants])
result.to_csv(eval_dir / "masking_ci.csv", index=False)
print(f"\nWrote {len(result)} rows to {eval_dir / 'masking_ci.csv'}")
