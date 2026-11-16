"""MIDL: is plain INT8 losing accuracy to kernel saturation on this CPU?

Without VNNI, ONNX Runtime's x86 kernels for uint8 activations x int8 weights (U8S8)
add pairs of products in int16, which can saturate. Two settings avoid it:
reduce_range (7-bit weights) and uint8 weights (U8U8). This quantizes one or more
runs (default effb0_s0) three ways and scores each on spain_val (leaked copies
removed, as in scripts/quantize.py):

- plain:        the settings used for the INT8 results (QInt8 weights, reduce_range off)
- reduce_range: QInt8 weights on 7 bits
- uint8_weights: QUInt8 weights, asymmetric

each with MinMax and Percentile calibration on the same 500 spain_train images. Each
model is also run with graph optimisation off ("simulated"): the QDQ nodes then run
as DequantizeLinear + float ops, so there are no integer kernels and no saturation.
If plain loses accuracy fused but not simulated, the loss comes from the kernels,
not from quantization. Also records the CPU model, its VNNI flags and the time per
image (1 thread, batch 1).

Spanish data only. Writes outputs/quantization/int8_saturation.csv; the models stay
in local scratch and nothing in runs/ or outputs/<run>/ changes.

    !python scripts/check_int8_saturation.py
    !python scripts/check_int8_saturation.py --runs effb0_s0 effb0_s1
"""
import argparse
import os
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.quantize import (CALIBRATION_FILE, METHODS, QUANT_DIR, accuracy, batches,  # noqa: E402
                              prepared_model, predict)
from scripts.time_models import N_IMAGES, time_model  # noqa: E402
from src import quantize as q  # noqa: E402
from src.eval import io  # noqa: E402
from src.paths import DRIVE_ROOT, LOCAL_ROOT  # noqa: E402

SETTINGS = {
    "plain": {},
    "reduce_range": {"reduce_range": True},
    "uint8_weights": {"weight_type": "QUInt8", "weight_symmetric": False},
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--runs", nargs="+", default=["effb0_s0"])
    args = parser.parse_args()

    flags = q.cpu_int8_flags()
    machine = {"cpu": q.cpu_model(), "cpu_count": os.cpu_count(),
               **{f"cpu_{k}": v for k, v in flags.items()}, "onnxruntime": q.versions()["onnxruntime"]}
    print(f"CPU: {machine['cpu']} ({machine['cpu_count']} logical cores)")
    print("  " + ", ".join(f"{k} {'yes' if v else 'no'}" for k, v in flags.items()))

    split_val = io.load_split("spain_val.csv")
    keep = io.evaluation_rows("spain_val.csv", split_val, io.load_duplicates())[0]
    out = QUANT_DIR / "int8_saturation.csv"
    rows = []
    for run_id in args.runs:
        cfg = torch.load(DRIVE_ROOT / "runs" / run_id / "best.pt", map_location="cpu")["config"]
        check = q.read_json(DRIVE_ROOT / "runs" / run_id / "onnx_check.json")
        val, y_val = [], []
        for images, labels in batches("spain_val", cfg):
            val.append(images)
            y_val.append(labels)
        y_val = np.concatenate(y_val)
        timing_images = np.concatenate(val)[:N_IMAGES]
        calib = [b for b, _ in batches(str(CALIBRATION_FILE), cfg, batch_size=q.CALIB_BATCH)]
        prepared = prepared_model(run_id)
        fp32_acc = check["fp32_val_acc"]
        print(f"\n{run_id}: FP32 spain_val acc {fp32_acc:.4f}")
        fp32_ms = time_model(DRIVE_ROOT / "runs" / run_id / "model_fp32.onnx", timing_images, 1)["median_ms"]
        work = LOCAL_ROOT / "quant_tmp" / f"saturation_{run_id}"
        work.mkdir(parents=True, exist_ok=True)
        for name, change in SETTINGS.items():
            for method in METHODS:
                path = work / f"{name}_{method}.onnx"
                q.quantize_int8(prepared, path, calib, method, settings=change)
                fused = accuracy(predict(q.session(path), val)[0], y_val, keep)
                simulated = accuracy(predict(q.session(path, fuse=False), val)[0], y_val, keep)
                ms = time_model(path, timing_images, 1)["median_ms"]
                rows.append({"run_id": run_id, "setting": name, "calibration_method": method,
                             "weight_type": change.get("weight_type", q.INT8_SETTINGS["weight_type"]),
                             "reduce_range": change.get("reduce_range", q.INT8_SETTINGS["reduce_range"]),
                             "fp32_val_acc": fp32_acc, "val_acc": fused, "val_acc_simulated": simulated,
                             "drop_points": round(100 * (fp32_acc - fused), 2),
                             "median_ms": ms, "fp32_median_ms": fp32_ms, **machine,
                             "date": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")})
                print(f"  {name:14s} {method:10s} spain_val acc {fused:.4f} "
                      f"(simulated {simulated:.4f}), {ms:.1f} ms/image vs FP32 {fp32_ms:.1f}")
        shutil.rmtree(work, ignore_errors=True)
        QUANT_DIR.mkdir(parents=True, exist_ok=True)
        new = pd.DataFrame(rows)
        if out.exists():
            old = pd.read_csv(out)
            new = pd.concat([old[~old["run_id"].isin(new["run_id"])], new], ignore_index=True)
        new.to_csv(out, index=False)
    print(f"\nsaved {out}")


if __name__ == "__main__":
    main()
