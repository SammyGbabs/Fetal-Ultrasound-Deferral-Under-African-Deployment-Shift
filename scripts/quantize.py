"""MIDL: FP16 and INT8 versions of trained runs, with ONNX Runtime on the CPU.

For each run (e.g. effb0_s0) this:

1. Exports the FP32 checkpoint to ONNX (runs/<run>/model_fp32.onnx) and checks it
   against PyTorch on the CPU over all of spain_val: the largest difference in logits
   and features goes to runs/<run>/onnx_check.json. A run whose logits differ by more
   than MAX_EXPORT_DIFF stops here, so export bugs never reach quantization.
2. FP16: converts the FP32 model (run <model>-fp16_s<seed>).
3. INT8: static post-training quantization, QDQ, per-channel weights, calibrated on
   the 500 spain_train images in splits/quant_calibration.csv (125 per view, seed 0).
   MinMax and Percentile calibration are both tried and the one with the higher
   spain_val accuracy is kept (MinMax on a tie). Run <model>-int8_s<seed>.
4. Compares each version's spain_val accuracy with the FP32 model and flags a drop of
   more than MAX_DROP.
5. Runs both versions on the test conditions and saves contract D under
   outputs/<run_id>/, then rebuilds the manifest. ONNX models and meta.json go to
   runs/<run_id>/.

Choices use spain_val only, never African data: accuracy is computed on spain_val
without the leaked copies in splits/duplicates.csv (541 images), as in the evaluation.
African accuracies are not printed.

mc_logits holds the plain logits repeated 20 times, since the ONNX models have no
dropout: MC-dropout results are not valid for these runs.

Resumable: finished steps and saved conditions are skipped on rerun. Run in Colab
after the template's setup cells (a CPU is enough):

    !python scripts/quantize.py --models effb0
    !python scripts/quantize.py --models effb0 mnv4 convnext vits16
    !python scripts/quantize.py --models effb0 --seeds 0 --conditions spain_val
"""
import argparse
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src import quantize as q  # noqa: E402
from src.data import make_loader  # noqa: E402
from src.eval import io  # noqa: E402
from src.infer import MC_PASSES, load_run, save_outputs, write_manifest  # noqa: E402
from src.models import normalization  # noqa: E402
from src.paths import DRIVE_ROOT, LOCAL_ROOT  # noqa: E402
from src.train import balanced_accuracy, git_commit  # noqa: E402

PRIORITY = ["effb0", "mnv4", "convnext", "vits16"]
CONDITIONS = ["spain_train", "spain_val", "spain_test", "africa_test", "africa_test_q10"]
METHODS = ["minmax", "percentile"]
CALIBRATION_FILE = ROOT / "splits" / "quant_calibration.csv"
N_CALIBRATION, CALIBRATION_SEED = 500, 0
MAX_EXPORT_DIFF = 1e-3   # largest logit difference allowed between ONNX FP32 and PyTorch
MAX_DROP = 0.01          # spain_val accuracy drop (FP32 minus quantized) that gets flagged
QUANT_DIR = DRIVE_ROOT / "outputs" / "quantization"


def calibration_images() -> pd.DataFrame:
    """500 spain_train images, 125 per view, drawn with a fixed seed; saved once and reused."""
    if CALIBRATION_FILE.exists():
        return pd.read_csv(CALIBRATION_FILE, keep_default_na=False)
    train = pd.read_csv(ROOT / "splits" / "spain_train.csv", keep_default_na=False)
    per_view = N_CALIBRATION // train["label"].nunique()
    chosen = train.groupby("label").sample(n=per_view, random_state=CALIBRATION_SEED).sort_index()
    chosen = chosen[["image_id", "path", "label", "label_name", "patient_id", "machine"]]
    chosen.to_csv(CALIBRATION_FILE, index=False)
    print(f"saved {CALIBRATION_FILE} ({len(chosen)} images)")
    return chosen


def batches(split: str, cfg: dict, image_root: str = "", batch_size: int = 32):
    """Preprocessed images of a split in CSV order, exactly as src/data.py prepares them."""
    mean, std = normalization(cfg["model"])
    loader = make_loader(split, train=False, batch_size=batch_size, num_workers=cfg["num_workers"],
                         image_size=cfg["image_size"], mean=mean, std=std,
                         image_root=image_root or None)
    for batch in loader:
        yield batch["image"].numpy(), batch["label"].numpy()


def predict(sess, images: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    parts = [q.run_session(sess, b) for b in images]
    return np.concatenate([p[0] for p in parts]), np.concatenate([p[1] for p in parts])


def accuracy(logits: np.ndarray, y: np.ndarray, keep: np.ndarray) -> float:
    return float((logits[keep].argmax(1) == y[keep]).mean())


def export_and_check(run_id: str, val: list[np.ndarray], y_val: np.ndarray, keep: np.ndarray,
                     force: bool) -> dict:
    """Step 1: FP32 ONNX export, checked against PyTorch on spain_val."""
    run_dir = DRIVE_ROOT / "runs" / run_id
    fp32, check_file = run_dir / "model_fp32.onnx", run_dir / "onnx_check.json"
    if check_file.exists() and fp32.exists() and not force:
        return q.read_json(check_file)
    model, cfg = load_run(run_id, torch.device("cpu"))
    exporter = q.export_onnx(model, fp32, cfg["image_size"])
    with torch.no_grad():
        ref = [q.ExportWrapper(model).eval()(torch.from_numpy(b)) for b in val]
    ref_logits = np.concatenate([r[0].numpy() for r in ref])
    ref_features = np.concatenate([r[1].numpy() for r in ref])
    logits, features = predict(q.session(fp32), val)
    check = {
        "run_id": run_id, "exporter": exporter, "opset": q.OPSET, "n_images": len(y_val),
        "max_abs_diff_logits": float(np.abs(logits - ref_logits).max()),
        "max_abs_diff_features": float(np.abs(features - ref_features).max()),
        "max_abs_logit": float(np.abs(ref_logits).max()),
        "max_abs_feature": float(np.abs(ref_features).max()),
        "same_predictions": int((logits.argmax(1) == ref_logits.argmax(1)).sum()),
        "fp32_val_acc": accuracy(ref_logits, y_val, keep),
        "onnx_val_acc": accuracy(logits, y_val, keep),
        "file_size_bytes": fp32.stat().st_size,
        **q.versions(), "git_commit": git_commit(), "date": now(),
    }
    check["passed"] = check["max_abs_diff_logits"] <= MAX_EXPORT_DIFF
    q.write_json(check_file, check)
    return check


def now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")


def base_meta(qrun: str, run_id: str, precision: str, cfg: dict, path: Path, check: dict) -> dict:
    model, seed = run_id.rsplit("_s", 1)
    mean, std = normalization(cfg["model"])
    return {
        "run_id": qrun, "base_run": run_id, "model": f"{model}-{precision}", "seed": int(seed),
        "precision": precision, "onnx_file": path.name, "file_size_bytes": path.stat().st_size,
        "input": {"name": q.INPUT, "shape": ["batch", 3, cfg["image_size"], cfg["image_size"]],
                  "preprocessing": "src/data.py build_transform(train=False): resize to a square, "
                                   "grey to 3 channels, normalize", "mean": mean, "std": std},
        "outputs": q.OUTPUTS, "opset": q.OPSET, "exporter": check["exporter"],
        "export_check": {k: check[k] for k in ("max_abs_diff_logits", "max_abs_diff_features")},
        "mc_logits": f"plain logits repeated {MC_PASSES} times; MC-dropout results are not valid",
        "execution_provider": "CPUExecutionProvider", "cpu": q.cpu_model(),
        **q.versions(), "git_commit": git_commit(), "date": now(),
    }


def sanity(meta: dict, fp32_acc: float) -> dict:
    """Step 4: flag a spain_val accuracy drop of more than MAX_DROP."""
    drop = fp32_acc - meta["val_acc"]
    meta.update(fp32_val_acc=fp32_acc, val_drop=round(drop, 5), flagged=bool(drop > MAX_DROP))
    flag = "   <-- drop above 1 point, check this model" if meta["flagged"] else ""
    print(f"  {meta['run_id']:22s} spain_val acc {meta['val_acc']:.4f} vs FP32 {fp32_acc:.4f}{flag}")
    return meta


def make_fp16(run_id, cfg, check, val, y_val, keep, force) -> dict | None:
    qrun = run_id.replace("_s", "-fp16_s", 1)
    run_dir = DRIVE_ROOT / "runs" / qrun
    meta_file, path = run_dir / "meta.json", run_dir / "model.onnx"
    if meta_file.exists() and path.exists() and not force:
        return q.read_json(meta_file)
    try:
        q.convert_fp16(DRIVE_ROOT / "runs" / run_id / "model_fp32.onnx", path)
        logits, _ = predict(q.session(path), val)
    except Exception as error:                                     # noqa: BLE001
        return failed(qrun, run_id, "fp16", error)
    meta = base_meta(qrun, run_id, "fp16", cfg, path, check)
    meta.update(conversion="onnxconverter_common.float16.convert_float_to_float16(keep_io_types=True)",
                val_acc=accuracy(logits, y_val, keep))
    meta = sanity(meta, check["fp32_val_acc"])
    q.write_json(meta_file, meta)
    return meta


def make_int8(run_id, cfg, check, val, y_val, keep, calib, force) -> dict | None:
    qrun = run_id.replace("_s", "-int8_s", 1)
    run_dir = DRIVE_ROOT / "runs" / qrun
    meta_file, path = run_dir / "meta.json", run_dir / "model.onnx"
    if meta_file.exists() and path.exists() and not force:
        return q.read_json(meta_file)
    work = LOCAL_ROOT / "quant_tmp" / qrun
    work.mkdir(parents=True, exist_ok=True)
    prepared = work / "prepared.onnx"
    q.prepare_for_quantization(DRIVE_ROOT / "runs" / run_id / "model_fp32.onnx", prepared)
    tried = {}
    for method in METHODS:
        candidate = work / f"{method}.onnx"
        try:
            q.quantize_int8(prepared, candidate, calib, method)
            logits, _ = predict(q.session(candidate), val)
            tried[method] = {"val_acc": accuracy(logits, y_val, keep),
                             "val_balanced_acc": balanced_accuracy(y_val[keep], logits[keep].argmax(1))}
            print(f"  {qrun:22s} {method:10s} spain_val acc {tried[method]['val_acc']:.4f}")
        except Exception as error:                                 # noqa: BLE001
            tried[method] = {"error": f"{type(error).__name__}: {error}"[:500]}
            print(f"  {qrun:22s} {method:10s} FAILED: {tried[method]['error']}")
    ok = [m for m in METHODS if "val_acc" in tried[m]]
    if not ok:
        return failed(qrun, run_id, "int8", RuntimeError("; ".join(t["error"] for t in tried.values())))
    chosen = max(ok, key=lambda m: (tried[m]["val_acc"], -METHODS.index(m)))   # MinMax wins a tie
    run_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(work / f"{chosen}.onnx", path)
    meta = base_meta(qrun, run_id, "int8", cfg, path, check)
    meta.update(quantization={**q.INT8_SETTINGS, "method": "static post-training",
                              "calibration_file": str(CALIBRATION_FILE.relative_to(ROOT)),
                              "n_calibration": int(sum(len(b) for b in calib)),
                              "calibration_seed": CALIBRATION_SEED, "methods_tried": tried,
                              "calibration_method": chosen,
                              "chosen_on": "spain_val accuracy, leaked copies removed"},
                val_acc=tried[chosen]["val_acc"])
    meta = sanity(meta, check["fp32_val_acc"])
    q.write_json(meta_file, meta)
    shutil.rmtree(work, ignore_errors=True)
    return meta


def failed(qrun: str, run_id: str, precision: str, error: Exception) -> None:
    """Record a model that could not be converted cleanly; no run is created for it."""
    message = f"{type(error).__name__}: {error}"[:2000]
    print(f"  {qrun}: could not make the {precision} model: {message}")
    q.write_json(QUANT_DIR / "failed" / f"{qrun}.json",
                 {"run_id": qrun, "base_run": run_id, "precision": precision, "error": message,
                  **q.versions(), "date": now()})
    return None


def run_conditions(meta: dict, cfg: dict, conditions: pd.DataFrame, batch_size: int, force: bool):
    """Step 5: contract D outputs for one quantized run."""
    qrun = meta["run_id"]
    sess = q.session(DRIVE_ROOT / "runs" / qrun / meta["onnx_file"])
    out_dir = DRIVE_ROOT / "outputs" / qrun
    out_dir.mkdir(parents=True, exist_ok=True)
    for condition in conditions.itertuples():
        if (out_dir / f"{condition.condition_id}.npz").exists() and not force:
            continue
        split = condition.split_file.removesuffix(".csv")
        df = pd.read_csv(ROOT / "splits" / condition.split_file, keep_default_na=False)
        logits, features = [], []
        for images, _ in batches(split, cfg, condition.image_root, batch_size):
            lo, fe = q.run_session(sess, images)
            logits.append(lo)
            features.append(fe)
        logits, features = np.concatenate(logits), np.concatenate(features)
        out = {"image_id": df["image_id"].to_numpy(dtype=str), "y_true": df["label"].to_numpy(dtype=np.int64),
               "logits": logits, "features": features,
               "mc_logits": np.repeat(logits[:, None, :], MC_PASSES, axis=1)}
        save_outputs(out, qrun, condition.condition_id, out_dir)
        note = (f"acc {(out['y_true'] == logits.argmax(1)).mean():.4f} (all rows)"
                if condition.condition_id.startswith("spain") else "saved")
        print(f"  ok  {qrun:22s} {condition.condition_id:18s} {len(df):5d} images | {note}")


def write_checks() -> None:
    """outputs/quantization/checks.csv: one row per quantized run, rebuilt from the meta files."""
    rows = []
    for meta_file in sorted((DRIVE_ROOT / "runs").glob("*-*_s*/meta.json")):
        m = q.read_json(meta_file)
        if m.get("precision") not in ("fp16", "int8"):
            continue
        quant = m.get("quantization") or {}
        tried = quant.get("methods_tried", {})
        rows.append({
            "run_id": m["run_id"], "base_run": m["base_run"], "precision": m["precision"],
            "calibration_method": quant.get("calibration_method", ""),
            "minmax_val_acc": tried.get("minmax", {}).get("val_acc", ""),
            "percentile_val_acc": tried.get("percentile", {}).get("val_acc", ""),
            "fp32_val_acc": m["fp32_val_acc"], "val_acc": m["val_acc"], "val_drop": m["val_drop"],
            "flagged": m["flagged"], "export_max_abs_diff_logits": m["export_check"]["max_abs_diff_logits"],
            "export_max_abs_diff_features": m["export_check"]["max_abs_diff_features"],
            "file_size_bytes": m["file_size_bytes"]})
    for f in sorted((QUANT_DIR / "failed").glob("*.json")):
        m = q.read_json(f)
        rows.append({"run_id": m["run_id"], "base_run": m["base_run"], "precision": m["precision"],
                     "error": m["error"]})
    if rows:
        QUANT_DIR.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(QUANT_DIR / "checks.csv", index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--models", nargs="+", default=PRIORITY)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--conditions", nargs="+", default=CONDITIONS)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--force", action="store_true", help="redo every step")
    args = parser.parse_args()

    conditions = pd.read_csv(ROOT / "splits" / "conditions.csv", keep_default_na=False)
    unknown = set(args.conditions) - set(conditions["condition_id"])
    if unknown:
        raise ValueError(f"Not in conditions.csv: {sorted(unknown)}")
    conditions = conditions.set_index("condition_id").loc[args.conditions].reset_index()

    split_val = io.load_split("spain_val.csv")
    keep = io.evaluation_rows("spain_val.csv", split_val, io.load_duplicates())[0]
    calib_df = calibration_images()
    calib_split = str(CALIBRATION_FILE)

    for model in args.models:
        for seed in args.seeds:
            run_id = f"{model}_s{seed}"
            if not (DRIVE_ROOT / "runs" / run_id / "best.pt").exists():
                print(f"skip {run_id}: no checkpoint")
                continue
            cfg = torch.load(DRIVE_ROOT / "runs" / run_id / "best.pt", map_location="cpu")["config"]
            print(f"\n{run_id}")
            val, y_val = [], []
            for images, labels in batches("spain_val", cfg, batch_size=args.batch_size):
                val.append(images)
                y_val.append(labels)
            y_val = np.concatenate(y_val)

            check = export_and_check(run_id, val, y_val, keep, args.force)
            print(f"  export check: largest difference to PyTorch {check['max_abs_diff_logits']:.2e} "
                  f"(logits), {check['max_abs_diff_features']:.2e} (features); "
                  f"{check['same_predictions']}/{check['n_images']} same predictions")
            if not check["passed"]:
                print(f"  STOP {run_id}: FP32 ONNX differs from PyTorch by more than {MAX_EXPORT_DIFF}; "
                      f"not quantizing")
                continue

            calib = [b for b, _ in batches(calib_split, cfg, batch_size=q.CALIB_BATCH)]
            assert sum(len(b) for b in calib) == len(calib_df)
            for meta in (make_fp16(run_id, cfg, check, val, y_val, keep, args.force),
                         make_int8(run_id, cfg, check, val, y_val, keep, calib, args.force)):
                if meta:
                    run_conditions(meta, cfg, conditions, args.batch_size, args.force)
            write_checks()
            write_manifest(DRIVE_ROOT / "outputs")
    write_checks()
    write_manifest(DRIVE_ROOT / "outputs")
    print(f"\nsaved {QUANT_DIR / 'checks.csv'} and outputs/manifest.csv")


if __name__ == "__main__":
    main()
