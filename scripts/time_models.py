"""MIDL: deployment numbers for the FP32, FP16 and INT8 ONNX models on the CPU.

For each run made by scripts/quantize.py, times runs/<run>/model_fp32.onnx and the
model.onnx of each quantized variant (<model>-fp16_s<seed>, -int8, -fp16mixed,
-int8mixed): batch size 1, a fixed
number of threads (--threads, default 1), 20 warm-up runs, then the median time over
200 images (the first 200 rows of spain_val, preprocessed beforehand, so the time is
the model alone). Also records the file size, the CPU model, and how many saved output
rows have non-finite logits per condition (e.g. FP16 overflow), from outputs/<run>/.

Time every model in one session, so they share a machine. Writes
outputs/quantization/deployment.csv (one row per model file).

    !python scripts/time_models.py
    !python scripts/time_models.py --models effb0 mnv4 --threads 1
"""
import argparse
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.quantize import PRIORITY, QUANT_DIR, VARIANTS, batches  # noqa: E402
from src import quantize as q  # noqa: E402
from src.paths import DRIVE_ROOT  # noqa: E402

WARMUP, N_IMAGES = 20, 200


def time_model(path: Path, images: np.ndarray, threads: int) -> dict:
    sess = q.session(path, threads)
    for i in range(WARMUP):
        q.run_session(sess, images[i % len(images)][None])
    times = []
    for image in images:
        start = time.perf_counter()
        q.run_session(sess, image[None])
        times.append(time.perf_counter() - start)
    times = np.array(times) * 1000
    return {"median_ms": round(float(np.median(times)), 3),
            "p10_ms": round(float(np.percentile(times, 10)), 3),
            "p90_ms": round(float(np.percentile(times, 90)), 3)}


def nonfinite(run_id: str) -> dict:
    """Rows with non-finite logits in each saved condition of a run (reported, never used for a choice)."""
    counts = {}
    for npz in sorted((DRIVE_ROOT / "outputs" / run_id).glob("*.npz")):
        if npz.name.endswith(".tmp.npz"):
            continue
        with np.load(npz) as z:
            n = int((~np.isfinite(z["logits"]).all(1)).sum())
        if n:
            counts[npz.stem] = n
    return {"nonfinite_rows": sum(counts.values()),
            "nonfinite_by_condition": ";".join(f"{c}:{n}" for c, n in counts.items())}


def save(out, rows: list[dict]) -> None:
    """Write deployment.csv, replacing only the runs timed now and keeping the others."""
    new = pd.DataFrame(rows)
    if out.exists():
        old = pd.read_csv(out)
        new = pd.concat([old[~old["base_run"].isin(new["base_run"])], new], ignore_index=True)
    new.to_csv(out, index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--models", nargs="+", default=PRIORITY)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--threads", type=int, default=1)
    args = parser.parse_args()

    out = QUANT_DIR / "deployment.csv"
    rows = []
    machine = {"cpu": q.cpu_model(), "cpu_count": os.cpu_count(), "threads": args.threads,
               "batch_size": 1, "warmup": WARMUP, "n_images": N_IMAGES,
               "onnxruntime": q.versions()["onnxruntime"]}
    print(f"CPU: {machine['cpu']} ({machine['cpu_count']} logical cores), {args.threads} thread(s)")
    for model in args.models:
        for seed in args.seeds:
            run_id = f"{model}_s{seed}"
            fp32 = DRIVE_ROOT / "runs" / run_id / "model_fp32.onnx"
            if not fp32.exists():
                continue
            cfg = torch.load(DRIVE_ROOT / "runs" / run_id / "best.pt", map_location="cpu")["config"]
            images = np.concatenate([b for b, _ in batches("spain_val", cfg, batch_size=50)])[:N_IMAGES]
            files = {"fp32": (run_id, fp32)}
            for precision in VARIANTS:
                qrun = run_id.replace("_s", f"-{precision}_s", 1)
                path = DRIVE_ROOT / "runs" / qrun / "model.onnx"
                if path.exists():
                    files[precision] = (qrun, path)
            for precision, (rid, path) in files.items():
                timing = time_model(path, images, args.threads)
                rows.append({"run_id": rid, "base_run": run_id, "model": model, "seed": seed,
                             "precision": precision, "file_size_mb": round(path.stat().st_size / 2**20, 3),
                             **timing, **nonfinite(rid), **machine,
                             "date": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")})
                bad = f"  {rows[-1]['nonfinite_rows']} non-finite rows" if rows[-1]["nonfinite_rows"] else ""
                print(f"  {rid:24s} {rows[-1]['file_size_mb']:8.1f} MB  {timing['median_ms']:8.2f} ms/image{bad}")
            QUANT_DIR.mkdir(parents=True, exist_ok=True)
            save(out, rows)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
