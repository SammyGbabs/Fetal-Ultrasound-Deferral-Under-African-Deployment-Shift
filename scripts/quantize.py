"""MIDL: FP16 and INT8 versions of trained runs, with ONNX Runtime on the CPU.

For each run (e.g. effb0_s0) this:

1. Exports the FP32 checkpoint to ONNX (runs/<run>/model_fp32.onnx) and checks it
   against PyTorch on the CPU over all of spain_val (runs/<run>/onnx_check.json). It
   passes if the largest logit difference, divided by the largest |logit|, is at most
   MAX_EXPORT_REL_DIFF and every prediction is the same. A run that fails stops here,
   so export bugs never reach quantization. The difference is relative because some
   checkpoints have logits in the hundreds or thousands (mnv4_s1 reaches 1145).
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

Activation ranges: the largest |activation| of every tensor of the FP32 model on the
500 calibration images goes to runs/<run>/activation_ranges.json, with one summary
row per run in outputs/quantization/activations.csv.

Mixed precision (--variants fp16mixed int8mixed), each with a rule fixed on Spanish
data only and recorded in meta.json:

- <model>-fp16mixed_s<seed>: if plain FP16 gives non-finite outputs on spain_train or
  spain_val, every node that reads or writes a tensor whose FP32 value exceeds the
  FP16 maximum (65504) on spain_train or spain_val stays in FP32. Otherwise the plain
  FP16 model is used unchanged.
- <model>-int8mixed_s<seed>: every node that reads or writes an outlier tensor stays in
  float. An outlier tensor's largest |value| on the calibration images is more than
  OUTLIER_RATIO times the median over all the model's tensors. Calibration as for
  plain INT8 (both methods, the better on spain_val). With no outlier tensors the
  plain INT8 model is used unchanged.

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
    !python scripts/quantize.py --variants fp16mixed int8mixed
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
MAX_EXPORT_REL_DIFF = 1e-4   # largest logit difference / largest |logit|, ONNX FP32 vs PyTorch
MAX_DROP = 0.01          # spain_val accuracy drop (FP32 minus quantized) that gets flagged
QUANT_DIR = DRIVE_ROOT / "outputs" / "quantization"
VARIANTS = ["fp16", "int8", "fp16mixed", "int8mixed"]
FP16_MAX = 65504.0       # largest finite FP16 value
OUTLIER_RATIO = 100      # int8mixed: tensors above this many times the model's median range stay float
FP16_RULE = ("if plain FP16 gives non-finite outputs on spain_train or spain_val, or any tensor of the "
             "FP32 model goes above 65504 (the FP16 maximum) on the 500 calibration images, every node that "
             "reads or writes a tensor whose FP32 |value| on spain_train or spain_val exceeds 65504 "
             "(the FP16 maximum) divided by a headroom factor stays in FP32; the factor starts at 1 and "
             "doubles (1, 2, 4, ... 32) until no spain_train or spain_val output is non-finite")
FP16_HEADROOM = [1, 2, 4, 8, 16, 32]   # fp16mixed: limits tried, as divisors of the FP16 maximum
INT8_RULE = (f"every node that reads or writes a tensor whose largest |value| on the 500 calibration "
             f"images (FP32, after quant_pre_process) is more than {OUTLIER_RATIO} times the median "
             f"of that value over all the model's tensors stays in float")


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
        check = q.read_json(check_file)
        if "rule" not in check:                       # written under the old absolute threshold
            export_passed(check)
            q.write_json(check_file, check)
        return check
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
    export_passed(check)
    q.write_json(check_file, check)
    return check


def export_passed(check: dict) -> bool:
    """Relative logit difference within MAX_EXPORT_REL_DIFF and every prediction the same."""
    check["rel_diff_logits"] = check["max_abs_diff_logits"] / max(check["max_abs_logit"], 1e-12)
    check["rule"] = (f"max |logit diff| / max |logit| <= {MAX_EXPORT_REL_DIFF} "
                     f"and all {check['n_images']} predictions the same")
    check["passed"] = bool(check["rel_diff_logits"] <= MAX_EXPORT_REL_DIFF
                           and check["same_predictions"] == check["n_images"])
    return check["passed"]


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
        "export_check": {k: check.get(k) for k in ("max_abs_diff_logits", "max_abs_diff_features",
                                                    "rel_diff_logits", "same_predictions", "rule")},
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


def qrun_of(run_id: str, variant: str) -> str:
    return run_id.replace("_s", f"-{variant}_s", 1)


def prepared_model(run_id: str) -> Path:
    """FP32 model after ONNX Runtime's quant_pre_process, in local scratch (deterministic)."""
    prepared = LOCAL_ROOT / "quant_tmp" / run_id / "prepared.onnx"
    if not prepared.exists():
        prepared.parent.mkdir(parents=True, exist_ok=True)
        q.prepare_for_quantization(DRIVE_ROOT / "runs" / run_id / "model_fp32.onnx", prepared)
    return prepared


def quantize_and_choose(qrun, run_id, cfg, check, val, y_val, keep, calib, keep_float=None) -> dict | None:
    """INT8 with both calibration methods; keeps the one with the higher spain_val accuracy."""
    work = LOCAL_ROOT / "quant_tmp" / qrun
    work.mkdir(parents=True, exist_ok=True)
    prepared = prepared_model(run_id)
    tried = {}
    for method in METHODS:
        candidate = work / f"{method}.onnx"
        try:
            q.quantize_int8(prepared, candidate, calib, method, keep_float)
            logits, _ = predict(q.session(candidate), val)
            tried[method] = {"val_acc": accuracy(logits, y_val, keep),
                             "val_balanced_acc": balanced_accuracy(y_val[keep], logits[keep].argmax(1))}
            print(f"  {qrun:24s} {method:10s} spain_val acc {tried[method]['val_acc']:.4f}")
        except Exception as error:                                 # noqa: BLE001
            tried[method] = {"error": f"{type(error).__name__}: {error}"[:500]}
            print(f"  {qrun:24s} {method:10s} FAILED: {tried[method]['error']}")
    ok = [m for m in METHODS if "val_acc" in tried[m]]
    precision = qrun.rsplit("_s", 1)[0].split("-")[-1]
    if not ok:
        return failed(qrun, run_id, precision, RuntimeError("; ".join(t["error"] for t in tried.values())))
    chosen = max(ok, key=lambda m: (tried[m]["val_acc"], -METHODS.index(m)))   # MinMax wins a tie
    run_dir = DRIVE_ROOT / "runs" / qrun
    run_dir.mkdir(parents=True, exist_ok=True)
    path = run_dir / "model.onnx"
    shutil.copy2(work / f"{chosen}.onnx", path)
    meta = base_meta(qrun, run_id, precision, cfg, path, check)
    meta.update(quantization={**q.INT8_SETTINGS, "method": "static post-training",
                              "calibration_file": str(CALIBRATION_FILE.relative_to(ROOT)),
                              "n_calibration": int(sum(len(b) for b in calib)),
                              "calibration_seed": CALIBRATION_SEED, "methods_tried": tried,
                              "calibration_method": chosen,
                              "chosen_on": "spain_val accuracy, leaked copies removed"},
                val_acc=tried[chosen]["val_acc"])
    shutil.rmtree(work, ignore_errors=True)
    return meta


def make_int8(run_id, cfg, check, val, y_val, keep, calib, force) -> dict | None:
    qrun = qrun_of(run_id, "int8")
    meta_file = DRIVE_ROOT / "runs" / qrun / "meta.json"
    if meta_file.exists() and (meta_file.parent / "model.onnx").exists() and not force:
        return q.read_json(meta_file)
    meta = quantize_and_choose(qrun, run_id, cfg, check, val, y_val, keep, calib)
    if meta:
        meta = sanity(meta, check["fp32_val_acc"])
        q.write_json(meta_file, meta)
    return meta


def activation_ranges(run_id: str, calib: list[np.ndarray], force: bool) -> dict:
    """Largest |activation| per tensor of the FP32 model (after quant_pre_process) on the
    calibration images: runs/<run>/activation_ranges.json. Used by the int8mixed rule."""
    out = DRIVE_ROOT / "runs" / run_id / "activation_ranges.json"
    if out.exists() and not force:
        data = q.read_json(out)
        if data["outlier_ratio"] != OUTLIER_RATIO:      # the counts depend on the ratio
            values = np.array([r["max_abs"] for r in data["tensors"].values()])
            data.update(outlier_ratio=OUTLIER_RATIO,
                        n_outliers=int((values > OUTLIER_RATIO * data["median_max_abs"]).sum()))
            q.write_json(out, data)
        return data
    ranges = q.tensor_ranges(prepared_model(run_id), calib)
    values = np.array([r["max_abs"] for r in ranges.values()])
    median = float(np.median(values))
    largest = max(ranges, key=lambda t: ranges[t]["max_abs"])
    data = {"run_id": run_id, "graph": "model_fp32.onnx after quant_pre_process",
            "images": str(CALIBRATION_FILE.relative_to(ROOT)), "n_tensors": len(ranges),
            "median_max_abs": median, "largest_max_abs": ranges[largest]["max_abs"],
            "largest_tensor": largest,
            "n_outliers": int((values > OUTLIER_RATIO * median).sum()),
            "n_over_fp16_max": int((values > FP16_MAX).sum()),
            "outlier_ratio": OUTLIER_RATIO, "tensors": ranges, "date": now()}
    q.write_json(out, data)
    return data


def write_activations() -> None:
    """outputs/quantization/activations.csv: one row per FP32 run (Spanish calibration images only)."""
    rows = []
    for f in sorted((DRIVE_ROOT / "runs").glob("*_s*/activation_ranges.json")):
        a = q.read_json(f)
        model, seed = a["run_id"].rsplit("_s", 1)
        rows.append({"run_id": a["run_id"], "model": model, "seed": int(seed), "n_tensors": a["n_tensors"],
                     "median_max_abs": round(a["median_max_abs"], 4),
                     "largest_max_abs": round(a["largest_max_abs"], 2),
                     "largest_over_median": round(a["largest_max_abs"] / max(a["median_max_abs"], 1e-12), 1),
                     "largest_tensor": a["largest_tensor"], "n_outliers": a["n_outliers"],
                     "n_over_fp16_max": a["n_over_fp16_max"]})
    if rows:
        QUANT_DIR.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(QUANT_DIR / "activations.csv", index=False)


def copy_variant(plain: dict, qrun: str, cfg: dict, check: dict, note: dict) -> dict:
    """Mixed variant whose rule kept nothing in float: the plain model and outputs, renamed."""
    run_dir, out_dir = DRIVE_ROOT / "runs" / qrun, DRIVE_ROOT / "outputs" / qrun
    run_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(DRIVE_ROOT / "runs" / plain["run_id"] / plain["onnx_file"], run_dir / "model.onnx")
    for npz in sorted((DRIVE_ROOT / "outputs" / plain["run_id"]).glob("*.npz")):
        if npz.name.endswith(".tmp.npz") or (out_dir / npz.name).exists():
            continue
        with np.load(npz) as z:
            out = {k: z[k] for k in ("image_id", "y_true", "logits", "features", "mc_logits")}
        save_outputs(out, qrun, npz.stem, out_dir)
    meta = {**plain, "run_id": qrun, "model": qrun.rsplit("_s", 1)[0], "precision": qrun.rsplit("_s", 1)[0].split("-")[-1],
            "mixed_precision": {**note, "same_as": plain["run_id"]}, "git_commit": git_commit(), "date": now()}
    q.write_json(run_dir / "meta.json", meta)
    print(f"  {qrun:24s} rule kept nothing in float: same model as {plain['run_id']}")
    return meta


def nonfinite_rows(sess, images) -> int:
    return int(sum((~np.isfinite(q.run_session(sess, b)[0]).all(1)).sum() for b in images))


def make_fp16mixed(run_id, cfg, check, val, y_val, keep, ranges, force) -> dict | None:
    qrun, plain_run = qrun_of(run_id, "fp16mixed"), qrun_of(run_id, "fp16")
    run_dir = DRIVE_ROOT / "runs" / qrun
    if (run_dir / "meta.json").exists() and (run_dir / "model.onnx").exists() and not force:
        meta = q.read_json(run_dir / "meta.json")
        mixed = meta.get("mixed_precision", {})
        left = mixed.get("nonfinite_rows_spain_train_val_after", 0)
        missed = not mixed.get("triggered", True) and ranges["n_over_fp16_max"] > 0
        if not left and not missed:
            return meta
        # made under an earlier version of the rule: rebuild it
        why = (f"still has {left} non-finite Spanish rows" if left else
               f"was not triggered, but {ranges['n_over_fp16_max']} tensors go above {FP16_MAX:.0f} "
               f"on the calibration images")
        print(f"  {qrun}: saved model {why}; rebuilding it")
        shutil.rmtree(DRIVE_ROOT / "outputs" / qrun, ignore_errors=True)
    if not (DRIVE_ROOT / "runs" / plain_run / "meta.json").exists():
        print(f"  {qrun}: make {plain_run} first")
        return None
    plain = q.read_json(DRIVE_ROOT / "runs" / plain_run / "meta.json")
    def spanish(batch_size=32):     # 32 as in the saved outputs, so the check sees what they see
        for split in ("spain_train", "spain_val"):
            for images, _ in batches(split, cfg, batch_size=batch_size):
                yield images

    saved = [DRIVE_ROOT / "outputs" / plain_run / f"{c}.npz" for c in ("spain_train", "spain_val")]
    if all(f.exists() for f in saved):
        bad = 0
        for f in saved:
            with np.load(f) as z:
                bad += int((~np.isfinite(z["logits"]).all(1)).sum())
    else:
        bad = nonfinite_rows(q.session(DRIVE_ROOT / "runs" / plain_run / "model.onnx"), spanish())
    over_calib = ranges["n_over_fp16_max"]
    note = {"rule": FP16_RULE, "plain_nonfinite_rows_spain_train_val": bad,
            "calibration_tensors_over_fp16_max": over_calib}
    if bad == 0 and over_calib == 0:
        return copy_variant(plain, qrun, cfg, check, {**note, "triggered": False, "kept_fp32": []})
    print(f"  {plain_run}: {bad} spain_train/spain_val rows non-finite, {over_calib} tensors above "
          f"{FP16_MAX:.0f} on the calibration images; finding the overflowing tensors")
    fp32 = DRIVE_ROOT / "runs" / run_id / "model_fp32.onnx"
    ranges = q.tensor_ranges(fp32, spanish(q.CALIB_BATCH))   # every tensor kept: small batches
    path, attempts = run_dir / "model.onnx", []
    try:
        for headroom in FP16_HEADROOM:
            limit = FP16_MAX / headroom
            over = {t: r["max_abs"] for t, r in ranges.items() if r["max_abs"] > limit}
            nodes = q.nodes_touching(fp32, set(over))
            q.convert_fp16(fp32, path, keep_fp32=nodes)
            sess = q.session(path)
            still_bad = nonfinite_rows(sess, spanish())
            attempts.append({"headroom": headroom, "limit": limit, "n_tensors": len(over),
                             "n_kept_fp32": len(nodes), "nonfinite_rows_spain_train_val": still_bad})
            print(f"  {qrun:24s} limit {limit:7.0f}: {len(over)} tensors, {len(nodes)} nodes kept in FP32, "
                  f"{still_bad} non-finite Spanish rows")
            if still_bad == 0:
                break
        logits, _ = predict(sess, val)
    except Exception as error:                                     # noqa: BLE001
        return failed(qrun, run_id, "fp16mixed", error)
    meta = base_meta(qrun, run_id, "fp16mixed", cfg, path, check)
    meta.update(conversion="onnxconverter_common.float16.convert_float_to_float16(keep_io_types=True, "
                           "node_block_list=kept_fp32)",
                mixed_precision={**note, "triggered": True, "headroom": headroom, "limit": limit,
                                 "attempts": attempts, "tensors_over_limit": over,
                                 "kept_fp32": nodes, "n_kept_fp32": len(nodes),
                                 "nonfinite_rows_spain_train_val_after": still_bad},
                val_acc=accuracy(logits, y_val, keep))
    meta = sanity(meta, check["fp32_val_acc"])
    q.write_json(run_dir / "meta.json", meta)
    return meta


def make_int8mixed(run_id, cfg, check, val, y_val, keep, calib, ranges, force) -> dict | None:
    qrun, plain_run = qrun_of(run_id, "int8mixed"), qrun_of(run_id, "int8")
    meta_file = DRIVE_ROOT / "runs" / qrun / "meta.json"
    if meta_file.exists() and (meta_file.parent / "model.onnx").exists() and not force:
        return q.read_json(meta_file)
    if not (DRIVE_ROOT / "runs" / plain_run / "meta.json").exists():
        print(f"  {qrun}: make {plain_run} first")
        return None
    threshold = OUTLIER_RATIO * ranges["median_max_abs"]
    outliers = {t: r["max_abs"] for t, r in ranges["tensors"].items() if r["max_abs"] > threshold}
    nodes = q.nodes_touching(prepared_model(run_id), set(outliers))
    note = {"rule": INT8_RULE, "outlier_ratio": OUTLIER_RATIO, "median_max_abs": ranges["median_max_abs"],
            "threshold": threshold, "outlier_tensors": outliers, "kept_float": nodes,
            "n_kept_float": len(nodes), "ranges_file": f"runs/{run_id}/activation_ranges.json"}
    if not nodes:
        return copy_variant(q.read_json(DRIVE_ROOT / "runs" / plain_run / "meta.json"), qrun, cfg, check, note)
    print(f"  {qrun:24s} {len(outliers)} outlier tensors (> {threshold:.1f}), {len(nodes)} nodes kept in float")
    meta = quantize_and_choose(qrun, run_id, cfg, check, val, y_val, keep, calib, keep_float=nodes)
    if meta:
        meta["mixed_precision"] = note
        meta = sanity(meta, check["fp32_val_acc"])
        q.write_json(meta_file, meta)
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
        if m.get("precision") not in VARIANTS:
            continue
        quant = m.get("quantization") or {}
        tried = quant.get("methods_tried", {})
        mixed = m.get("mixed_precision") or {}
        rows.append({
            "run_id": m["run_id"], "base_run": m["base_run"], "precision": m["precision"],
            "calibration_method": quant.get("calibration_method", ""),
            "minmax_val_acc": tried.get("minmax", {}).get("val_acc", ""),
            "percentile_val_acc": tried.get("percentile", {}).get("val_acc", ""),
            "fp32_val_acc": m["fp32_val_acc"], "val_acc": m["val_acc"], "val_drop": m["val_drop"],
            "flagged": m["flagged"],
            "n_kept_float": len(mixed.get("kept_float", mixed.get("kept_fp32", []))) if mixed else "",
            "same_as": mixed.get("same_as", ""),
            "export_max_abs_diff_logits": m["export_check"]["max_abs_diff_logits"],
            "export_rel_diff_logits": m["export_check"].get("rel_diff_logits", ""),
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
    parser.add_argument("--variants", nargs="+", default=["fp16", "int8"], choices=VARIANTS)
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
                  f"(logits, {check['rel_diff_logits']:.1e} of the largest logit), "
                  f"{check['max_abs_diff_features']:.2e} (features); "
                  f"{check['same_predictions']}/{check['n_images']} same predictions")
            if not check["passed"]:
                print(f"  STOP {run_id}: FP32 ONNX fails the export check ({check['rule']}); not quantizing")
                continue

            calib = [b for b, _ in batches(calib_split, cfg, batch_size=q.CALIB_BATCH)]
            assert sum(len(b) for b in calib) == len(calib_df)
            ranges = activation_ranges(run_id, calib, args.force)
            print(f"  activations: largest {ranges['largest_max_abs']:.1f}, median {ranges['median_max_abs']:.2f}, "
                  f"{ranges['n_outliers']} tensors over {OUTLIER_RATIO}x the median, "
                  f"{ranges['n_over_fp16_max']} over the FP16 maximum")
            write_activations()
            makers = {
                "fp16": lambda: make_fp16(run_id, cfg, check, val, y_val, keep, args.force),
                "int8": lambda: make_int8(run_id, cfg, check, val, y_val, keep, calib, args.force),
                "fp16mixed": lambda: make_fp16mixed(run_id, cfg, check, val, y_val, keep, ranges,
                                                    args.force),
                "int8mixed": lambda: make_int8mixed(run_id, cfg, check, val, y_val, keep, calib, ranges,
                                                    args.force),
            }
            for variant in VARIANTS:
                if variant in args.variants:
                    meta = makers[variant]()
                    if meta:
                        run_conditions(meta, cfg, conditions, args.batch_size, args.force)
            shutil.rmtree(LOCAL_ROOT / "quant_tmp" / run_id, ignore_errors=True)
            write_checks()
            write_manifest(DRIVE_ROOT / "outputs")
    write_checks()
    write_activations()
    write_manifest(DRIVE_ROOT / "outputs")
    print(f"\nsaved {QUANT_DIR / 'checks.csv'}, {QUANT_DIR / 'activations.csv'} and outputs/manifest.csv")


if __name__ == "__main__":
    main()
