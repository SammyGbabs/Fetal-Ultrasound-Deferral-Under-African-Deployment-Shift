"""T6: compute per-run metrics for every run, condition, uncertainty score and calibration.

Writes eval/per_run_metrics.csv (one row per run x condition x score x calib)
and eval/temperatures.csv. Confidence intervals, subgroups and seed
aggregation are T7.

    !python scripts/run_eval.py                    # real outputs from Drive
    !python scripts/run_eval.py --outputs /path    # e.g. fake outputs for testing
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.eval import io, pipeline  # noqa: E402
from src.paths import DRIVE_ROOT  # noqa: E402

parser = argparse.ArgumentParser()
parser.add_argument("--outputs", default=str(DRIVE_ROOT / "outputs"))
parser.add_argument("--eval-dir", default=str(DRIVE_ROOT / "eval"))
args = parser.parse_args()
outputs_root, eval_dir = Path(args.outputs), Path(args.eval_dir)
eval_dir.mkdir(parents=True, exist_ok=True)

manifest = io.load_manifest(outputs_root)
test_conditions = [c for c in io.load_conditions().condition_id
                   if c not in ("spain_train", "spain_val")]
rows, temps = [], []

for variant, vman in manifest.groupby("variant"):
    run_ids = sorted(vman.run_id.unique())
    fits, cache = {}, {}
    for run_id in run_ids:                                   # fit on Spanish data only
        val = io.load_output(outputs_root, run_id, "spain_val")
        train = io.load_output(outputs_root, run_id, "spain_train")
        fits[run_id] = pipeline.fit_run(val, train)
        cache[(run_id, "spain_val")] = val
        temps.append({"run_id": run_id, "variant": variant, "temperature": fits[run_id].T})
    ens_temps = [fits[r].T for r in run_ids]

    for condition in ["spain_val"] + test_conditions:
        outs = {r: cache.get((r, condition)) or io.load_output(outputs_root, r, condition)
                for r in run_ids}
        ens_logits = [outs[r]["logits"] for r in run_ids] if len(run_ids) > 1 else None
        for run_id in run_ids:
            res = pipeline.per_image(outs[run_id], fits[run_id], ens_logits, ens_temps)
            cache[(run_id, condition, "res")] = res
    for run_id in run_ids:
        val_res = cache[(run_id, "spain_val", "res")]
        ref_res = cache[(run_id, "spain_test", "res")]
        for condition in test_conditions:
            res = cache[(run_id, condition, "res")]
            for r in pipeline.summarize(res, val_res=val_res,
                                        ref_res=None if condition == "spain_test" else ref_res):
                rows.append({"run_id": run_id, "variant": variant,
                             "seed": int(run_id.rsplit("_s", 1)[1]), "condition": condition, **r})
    print(f"ok  {variant}: {len(run_ids)} runs")

pd.DataFrame(rows).to_csv(eval_dir / "per_run_metrics.csv", index=False)
pd.DataFrame(temps).to_csv(eval_dir / "temperatures.csv", index=False)
print(f"\nWrote {len(rows)} rows to {eval_dir / 'per_run_metrics.csv'}")
