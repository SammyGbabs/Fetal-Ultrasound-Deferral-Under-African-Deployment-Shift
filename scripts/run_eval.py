"""T6: compute per-run metrics for every run, condition, uncertainty score and calibration.

Which rows are evaluated (src/eval/io.py, `evaluation_rows`):
  - duplicate copies listed in splits/duplicates.csv are dropped from every
    condition built on that split (395 distinct African images instead of 450;
    14 leaked images removed from Spanish validation before anything is fitted)
  - pass --no-dedup to evaluate all images instead; results then go to
    eval/no_dedup/ so both versions can be compared
Masked test conditions get their temperature and deferral thresholds from
spain_val_masked; every other condition from spain_val.

Saves each model variant's results to <eval-dir>/per_variant/ as soon as it
finishes, so a disconnected session can resume. At the end, combines
everything into per_run_metrics.csv and temperatures.csv.

    !python scripts/run_eval.py --outputs /content/outputs_local --redo
    !python scripts/run_eval.py --outputs /content/outputs_local --redo --no-dedup
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.eval import calibration, io, pipeline, scores  # noqa: E402
from src.paths import DRIVE_ROOT  # noqa: E402

parser = argparse.ArgumentParser()
parser.add_argument("--outputs", default=str(DRIVE_ROOT / "outputs"))
parser.add_argument("--eval-dir", default=str(DRIVE_ROOT / "eval"))
parser.add_argument("--duplicates", default=str(io.DUPLICATES_FILE))
parser.add_argument("--no-dedup", action="store_true", help="evaluate all images")
parser.add_argument("--redo", action="store_true", help="ignore saved progress")
args = parser.parse_args()
outputs_root, eval_dir = Path(args.outputs), Path(args.eval_dir)
if args.no_dedup:
    eval_dir = eval_dir / "no_dedup"
progress_dir = eval_dir / "per_variant"
progress_dir.mkdir(parents=True, exist_ok=True)

duplicates = None if args.no_dedup else io.load_duplicates(args.duplicates)
manifest = io.load_manifest(outputs_root)
conditions = io.load_conditions().set_index("condition_id")
all_conditions = list(conditions.index)
val_conditions = [c for c in ("spain_val", "spain_val_masked") if c in all_conditions]
test_conditions = [c for c in all_conditions if not io.is_validation(c)]

# rows to evaluate and merged patient IDs, per split file
rows_for = {}
for f in conditions["split_file"].unique():
    rows_for[f] = io.evaluation_rows(f, io.load_split(f), duplicates)[0]
print(f"Images evaluated per split: " + ", ".join(
    f"{Path(f).stem} {len(r)}" for f, r in rows_for.items()))


def keep(condition: str):
    return rows_for[conditions.loc[condition, "split_file"]]


for variant, vman in manifest.groupby("variant"):
    metrics_file = progress_dir / f"{variant}_metrics.csv"
    temps_file = progress_dir / f"{variant}_temperatures.csv"
    if metrics_file.exists() and temps_file.exists() and not args.redo:
        print(f"skip {variant} (already done)")
        continue

    run_ids = sorted(vman.run_id.unique())
    fits, cache, rows, temps = {}, {}, [], []
    for run_id in run_ids:                                   # fit on Spanish data only
        train = io.subset(io.load_output(outputs_root, run_id, "spain_train"), keep("spain_train"))
        maha = scores.Mahalanobis().fit(train["features"], train["y_true"])
        for vc in val_conditions:
            val = io.load_output(outputs_root, run_id, vc)
            cache[(run_id, vc)] = val
            v = io.subset(val, keep(vc))
            T = calibration.fit_temperature(v["logits"], v["y_true"])
            fits[(run_id, vc)] = pipeline.RunFit(T=T, maha=maha)
            temps.append({"run_id": run_id, "variant": variant, "val_condition": vc,
                          "temperature": T, "n_val_images": len(v["y_true"])})

    for condition in val_conditions + test_conditions:
        vc = condition if condition in val_conditions else io.validation_condition(condition)
        outs = {r: cache.get((r, condition)) or io.load_output(outputs_root, r, condition)
                for r in run_ids}
        ens_logits = [outs[r]["logits"] for r in run_ids] if len(run_ids) > 1 else None
        ens_temps = [fits[(r, vc)].T for r in run_ids]
        for run_id in run_ids:
            cache[(run_id, condition, "res")] = pipeline.per_image(
                outs[run_id], fits[(run_id, vc)], ens_logits, ens_temps)

    for run_id in run_ids:
        for condition in test_conditions:
            vc, ref = io.validation_condition(condition), io.reference_condition(condition)
            res = cache[(run_id, condition, "res")]
            val_res = pipeline.subset_res(cache[(run_id, vc, "res")], keep(vc))
            ref_res = (None if condition == ref else
                       pipeline.subset_res(cache[(run_id, ref, "res")], keep(ref)))
            for r in pipeline.summarize(res, idx=keep(condition), val_res=val_res, ref_res=ref_res):
                rows.append({"run_id": run_id, "variant": variant,
                             "seed": int(run_id.rsplit("_s", 1)[1]), "condition": condition,
                             "val_condition": vc, **r})

    # write to temporary names first, so a disconnect mid-write never leaves a half file
    pd.DataFrame(rows).to_csv(metrics_file.with_suffix(".tmp"), index=False)
    pd.DataFrame(temps).to_csv(temps_file.with_suffix(".tmp"), index=False)
    metrics_file.with_suffix(".tmp").rename(metrics_file)
    temps_file.with_suffix(".tmp").rename(temps_file)
    print(f"ok  {variant}: {len(run_ids)} runs saved")

variants = sorted(manifest.variant.unique())
metrics = pd.concat([pd.read_csv(progress_dir / f"{v}_metrics.csv") for v in variants])
temperatures = pd.concat([pd.read_csv(progress_dir / f"{v}_temperatures.csv") for v in variants])
metrics.to_csv(eval_dir / "per_run_metrics.csv", index=False)
temperatures.to_csv(eval_dir / "temperatures.csv", index=False)
print(f"\nWrote {len(metrics)} rows to {eval_dir / 'per_run_metrics.csv'}")
