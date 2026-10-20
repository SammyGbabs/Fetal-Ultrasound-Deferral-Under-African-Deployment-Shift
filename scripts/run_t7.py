"""T7: headline results with patient-level 95% confidence intervals.

Covers the locked headline: maximum softmax probability (single model, averaged
over 5 seeds) and the 5-seed deep ensemble, each raw and temperature-scaled,
on Spanish test (all, familiar machines, Voluson S10), African test (all and
per country) and African test at JPEG q10. Every interval comes from 2,000
resamples of patients. The difference "temperature-scaled minus raw" gets its
own interval, from the same resamples.

Resumable like run_eval.py: each variant is saved to eval/t7_per_variant/ when
done, and finished variants are skipped on rerun. Writes (contract E):
    eval/results_main.csv      one row per variant x method x calib x group x metric
    eval/deferral_curves.csv   accuracy vs coverage, with intervals (Figure 3)

    !python scripts/run_t7.py --outputs /content/outputs_local
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import softmax

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.eval import bootstrap, calibration, io  # noqa: E402
from src.eval.scores import entropy  # noqa: E402
from src.paths import DRIVE_ROOT  # noqa: E402

parser = argparse.ArgumentParser()
parser.add_argument("--outputs", default=str(DRIVE_ROOT / "outputs"))
parser.add_argument("--eval-dir", default=str(DRIVE_ROOT / "eval"))
parser.add_argument("--B", type=int, default=2000, help="bootstrap resamples")
parser.add_argument("--redo", action="store_true")
args = parser.parse_args()
outputs_root, eval_dir = Path(args.outputs), Path(args.eval_dir)
progress_dir = eval_dir / "t7_per_variant"
progress_dir.mkdir(parents=True, exist_ok=True)

CONDITIONS = ["spain_test", "africa_test", "africa_test_q10"]
CURVE_GROUPS = {"spain_test_all", "africa_test_all", "africa_test_q10_all"}
GRID = np.round(np.arange(0.10, 1.0001, 0.05), 2)
METRICS = ["accuracy", "balanced_accuracy", "ece", "silent_failure", "acc_at_80",
           "deployed_coverage_80", "deployed_acc_80", "errors_passed_80"]


def logits_only(run_id, condition):
    with np.load(outputs_root / run_id / f"{condition}.npz", allow_pickle=True) as z:
        return z["logits"].astype(np.float64), z["y_true"].astype(int)


def groups_for(condition: str, split: pd.DataFrame) -> dict:
    """Row subsets reported for each condition."""
    g = {f"{condition}_all": np.arange(len(split))}
    if condition == "spain_test":
        g["spain_test_familiar"] = np.flatnonzero(split.machine.isin(["Voluson E6", "Aloka"]))
        g["spain_test_s10"] = np.flatnonzero(split.machine == "Voluson S10")
    if condition == "africa_test":
        for country in sorted(split.country.unique()):
            g[f"africa_test_{country.lower()}"] = np.flatnonzero(split.country == country)
    return g


def summarise(values: np.ndarray) -> tuple:
    v = values[np.isfinite(values)]
    return (np.percentile(v, 2.5), np.percentile(v, 97.5)) if len(v) else (np.nan, np.nan)


manifest = io.load_manifest(outputs_root)
conditions = io.load_conditions().set_index("condition_id")

for variant, vman in manifest.groupby("variant"):
    res_file = progress_dir / f"{variant}_results.csv"
    curve_file = progress_dir / f"{variant}_curves.csv"
    if res_file.exists() and curve_file.exists() and not args.redo:
        print(f"skip {variant} (already done)")
        continue
    run_ids = sorted(vman.run_id.unique())

    # Temperatures and thresholds: Spanish validation only (same fit as T6)
    T, val = {}, {}
    for r in run_ids:
        val[r] = logits_only(r, "spain_val")
        T[r] = calibration.fit_temperature(*val[r])

    def predictions(logits_by_run):
        """(pred, confidence, uncertainty) for each method x calib x seed."""
        out = {}
        for calib in ("raw", "ts"):
            probs = {r: softmax(logits_by_run[r] / (T[r] if calib == "ts" else 1.0), axis=1)
                     for r in run_ids}
            for r in run_ids:
                p = probs[r]
                out[("msp", calib, r)] = (p.argmax(1), p.max(1), 1 - p.max(1))
            mean = np.mean([probs[r] for r in run_ids], axis=0)
            out[("ensemble", calib, "ens")] = (mean.argmax(1), mean.max(1), entropy(mean))
        return out

    val_pred = predictions({r: val[r][0] for r in run_ids})
    thresholds = {k: float(np.quantile(v[2], 0.8)) for k, v in val_pred.items()}

    rows, curves = [], []
    for condition in CONDITIONS:
        split = io.load_split(conditions.loc[condition, "split_file"])
        loaded = {r: logits_only(r, condition) for r in run_ids}
        y = loaded[run_ids[0]][1]
        preds = predictions({r: loaded[r][0] for r in run_ids})
        patients = split.patient_id.to_numpy()

        for group, rows_idx in groups_for(condition, split).items():
            sampler = bootstrap.PatientSampler(patients[rows_idx])

            def evaluate(idx):
                """Metrics on rows idx: seed-averaged MSP and ensemble, raw and ts."""
                result, curve = {}, {}
                for method in ("msp", "ensemble"):
                    for calib in ("raw", "ts"):
                        keys = [k for k in preds if k[0] == method and k[1] == calib]
                        per_seed = [bootstrap.headline_metrics(
                            y[idx], preds[k][0][idx], preds[k][1][idx], preds[k][2][idx],
                            thresholds[k]) for k in keys]
                        result[(method, calib)] = {m: np.nanmean([s[m] for s in per_seed])
                                                   for m in METRICS}
                        if group in CURVE_GROUPS and calib == "raw":
                            curve[method] = np.mean([bootstrap.coverage_curve(
                                y[idx], preds[k][0][idx], preds[k][2][idx], GRID)
                                for k in keys], axis=0)
                return result, curve

            point, point_curve = evaluate(rows_idx)
            seed_sd = {}
            for calib in ("raw", "ts"):
                keys = [k for k in preds if k[0] == "msp" and k[1] == calib]
                per_seed = [bootstrap.headline_metrics(
                    y[rows_idx], preds[k][0][rows_idx], preds[k][1][rows_idx],
                    preds[k][2][rows_idx], thresholds[k]) for k in keys]
                seed_sd[calib] = {m: float(np.nanstd([s[m] for s in per_seed], ddof=1))
                                  for m in METRICS}

            boot = {key: {m: [] for m in METRICS} for key in point}
            boot_curve = {m: [] for m in point_curve}
            for _ in range(args.B):
                b, bc = evaluate(rows_idx[sampler.sample()])
                for key in b:
                    for m in METRICS:
                        boot[key][m].append(b[key][m])
                for m in bc:
                    boot_curve[m].append(bc[m])

            n_img, n_pat = len(rows_idx), sampler.n_patients
            for method in ("msp", "ensemble"):
                for calib in ("raw", "ts", "ts_minus_raw"):
                    for m in METRICS:
                        if calib == "ts_minus_raw":
                            est = point[(method, "ts")][m] - point[(method, "raw")][m]
                            vals = np.array(boot[(method, "ts")][m]) - np.array(boot[(method, "raw")][m])
                            sd = np.nan
                        else:
                            est = point[(method, calib)][m]
                            vals = np.array(boot[(method, calib)][m])
                            sd = seed_sd[calib][m] if method == "msp" else np.nan
                        lo, hi = summarise(vals)
                        rows.append({"variant": variant, "method": method, "calib": calib,
                                     "condition": condition, "group": group, "metric": m,
                                     "estimate": est, "ci_low": lo, "ci_high": hi,
                                     "seed_sd": sd, "n_seeds": len(run_ids),
                                     "n_images": n_img, "n_patients": n_pat})
            for method, curve in point_curve.items():
                arr = np.array(boot_curve[method])
                for j, cov in enumerate(GRID):
                    lo, hi = summarise(arr[:, j])
                    curves.append({"variant": variant, "method": method, "group": group,
                                   "coverage": cov, "accuracy": curve[j],
                                   "ci_low": lo, "ci_high": hi})
        print(f"    {variant} {condition} done")

    for df, f in ((pd.DataFrame(rows), res_file), (pd.DataFrame(curves), curve_file)):
        df.to_csv(f.with_suffix(".tmp"), index=False)
        f.with_suffix(".tmp").rename(f)
    print(f"ok  {variant}: {len(rows)} result rows saved")

variants = sorted(manifest.variant.unique())
results = pd.concat([pd.read_csv(progress_dir / f"{v}_results.csv") for v in variants])
curves = pd.concat([pd.read_csv(progress_dir / f"{v}_curves.csv") for v in variants])
results.to_csv(eval_dir / "results_main.csv", index=False)
curves.to_csv(eval_dir / "deferral_curves.csv", index=False)
print(f"\nWrote {len(results)} rows to results_main.csv and {len(curves)} to deferral_curves.csv")
