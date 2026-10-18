"""Reproducibility check: rerun inference for chosen runs on one condition and compare
the new logits with the saved contract D files. Writes nothing.

Run in Colab after the template's setup cells:
    !python scripts/check_reproducibility.py

By default it reruns seed 0 of each of the 9 model variants on spain_val. It prints the
library versions and GPU in use, then for each run the largest difference against the
saved .npz. GPU type matters: the saved outputs come from the GPU recorded in each run's
meta.json, and a different GPU type can shift logits slightly more than the tolerance.
"""
import argparse
import json
import sys
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.infer import load_run, run_condition  # noqa: E402
from src.paths import DRIVE_ROOT  # noqa: E402

DEFAULT_RUNS = ["effb0_s0", "rn50_s0", "mnv4_s0", "convnext_s0", "vits16_s0",
                "dinov3s_s0", "dinov3s_frozen_s0", "fetalclip_s0", "fetalclip_frozen_s0"]


def check(run_id: str, condition, device, tolerance: float) -> bool:
    saved = np.load(DRIVE_ROOT / "outputs" / run_id / f"{condition.condition_id}.npz")
    meta = json.loads((DRIVE_ROOT / "runs" / run_id / "meta.json").read_text())
    model, cfg = load_run(run_id, device)
    new = run_condition(model, cfg, condition, device, batch_size=128)

    if not np.array_equal(new["image_id"], saved["image_id"]):
        print(f"FAIL  {run_id}: image order differs from the saved file")
        return False
    logit_diff = float(np.abs(new["logits"] - saved["logits"]).max())
    feature_diff = float(np.abs(new["features"] - saved["features"]).max())
    mc_diff = float(np.abs(new["mc_logits"] - saved["mc_logits"]).max())
    same_pred = int((new["logits"].argmax(1) == saved["logits"].argmax(1)).sum())
    ok = logit_diff <= tolerance
    print(f"{'ok  ' if ok else 'FAIL'}  {run_id:20s} max |logit diff| {logit_diff:.1e} | "
          f"features {feature_diff:.1e} | MC logits {mc_diff:.1e} | "
          f"same prediction {same_pred}/{len(new['image_id'])} | trained on {meta['device']}, "
          f"saved commit {saved['git_commit']}")
    return ok


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--runs", nargs="+", default=DEFAULT_RUNS)
    parser.add_argument("--condition", default="spain_val")
    parser.add_argument("--tolerance", type=float, default=1e-4)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("versions:", ", ".join(f"{p} {version(p)}" for p in ["torch", "timm", "open_clip_torch"]))
    print("device:  ", torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu")
    conditions = pd.read_csv(ROOT / "splits" / "conditions.csv", keep_default_na=False)
    condition = next(conditions[conditions["condition_id"] == args.condition].itertuples())

    results = [check(run_id, condition, device, args.tolerance) for run_id in args.runs]
    print(f"\n{sum(results)} of {len(results)} runs within {args.tolerance:g}")


if __name__ == "__main__":
    main()
