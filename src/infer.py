"""T5: run one trained model on every test condition and save contract D.

    !python src/infer.py --run effb0_s0
    !python src/infer.py --run effb0_s0 rn50_s0 --conditions spain_val africa_test_q30

For each condition in splits/conditions.csv this writes, under outputs/<run_id>/:
    <condition_id>.npz        image_id, y_true, logits, features, mc_logits
    <condition_id>_preds.csv  image_id, y_true, y_pred, p0-p3
and then rebuilds outputs/manifest.csv. Rows keep the order of the condition's
split CSV. Files that already exist are skipped, so a dropped session is
continued by rerunning the same command.
"""
import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.data import make_loader  # noqa: E402
from src.models import build_model  # noqa: E402
from src.paths import DRIVE_ROOT  # noqa: E402
from src.train import NUM_CLASSES, balanced_accuracy, git_commit, seed_everything  # noqa: E402

MC_PASSES = 20


def load_run(run_id: str, device) -> tuple[torch.nn.Module, dict]:
    checkpoint = torch.load(DRIVE_ROOT / "runs" / run_id / "best.pt", map_location=device)
    cfg = checkpoint["config"]
    model = build_model(cfg["model"], NUM_CLASSES, cfg["dropout"], pretrained=False)
    model.load_state_dict(checkpoint["model"])
    return model.to(device).eval(), cfg


def run_condition(model, cfg: dict, condition, device, batch_size: int) -> dict:
    split = condition.split_file.removesuffix(".csv")
    loader = make_loader(split, train=False, batch_size=batch_size, num_workers=cfg["num_workers"],
                         image_size=cfg["image_size"], image_root=condition.image_root or None)
    df = loader.dataset.df
    # Fixed seed so the 20 dropout masks are reproducible for a given run and condition
    seed_everything(int(cfg["seed"]))
    parts = {"logits": [], "features": [], "mc_logits": []}
    index = []
    for batch in loader:
        out = model.predict(batch["image"].to(device), MC_PASSES)
        for key in parts:
            parts[key].append(out[key].cpu().numpy().astype(np.float32))
        index.append(batch["index"].numpy())
    index = np.concatenate(index)
    if not np.array_equal(index, np.arange(len(df))):
        raise RuntimeError(f"{condition.condition_id}: rows are not in split CSV order")
    return {"image_id": df["image_id"].to_numpy(dtype=str),
            "y_true": df["label"].to_numpy(dtype=np.int64),
            **{key: np.concatenate(value) for key, value in parts.items()}}


def save_outputs(out: dict, run_id: str, condition_id: str, out_dir: Path) -> None:
    # The .npz is written last and atomically, so its presence means the pair is complete
    logits = out["logits"]
    probs = np.exp(logits - logits.max(1, keepdims=True))
    probs /= probs.sum(1, keepdims=True)
    preds = pd.DataFrame({"image_id": out["image_id"], "y_true": out["y_true"],
                          "y_pred": logits.argmax(1)})
    for c in range(probs.shape[1]):
        preds[f"p{c}"] = probs[:, c].round(6)
    preds.to_csv(out_dir / f"{condition_id}_preds.csv", index=False)

    tmp = out_dir / f"{condition_id}.tmp.npz"
    np.savez(tmp, **out, run_id=run_id, condition_id=condition_id, git_commit=git_commit(),
             date=datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"))
    os.replace(tmp, out_dir / f"{condition_id}.npz")


def write_manifest(outputs_dir: Path) -> None:
    """Rebuild outputs/manifest.csv from the saved files (smoke runs excluded)."""
    rows = []
    for path in sorted(outputs_dir.glob("*/*.npz")):
        if path.parent.name.startswith("_") or path.name.endswith(".tmp.npz"):
            continue
        with np.load(path) as z:
            y_true, y_pred = z["y_true"], z["logits"].argmax(1)
            run_id = str(z["run_id"])
            model, seed = run_id.rsplit("_s", 1)
            rows.append({"run_id": run_id, "model": model, "seed": int(seed),
                         "condition_id": str(z["condition_id"]),
                         "path": str(path.relative_to(DRIVE_ROOT)), "n_images": len(y_true),
                         "accuracy": round(float((y_true == y_pred).mean()), 5),
                         "git_commit": str(z["git_commit"]), "date": str(z["date"])})
    if rows:
        pd.DataFrame(rows).to_csv(outputs_dir / "manifest.csv", index=False)


def sanity_check(run_id: str, out_dir: Path) -> None:
    """Clean Spanish test, familiar scanners only, should sit close to the validation score.

    The Voluson S10 images are left out because that scanner is almost absent from
    training and validation, so they would lower the test score for a reason that is
    one of the study's findings, not a bug.
    """
    npz, meta_file = out_dir / "spain_test.npz", DRIVE_ROOT / "runs" / run_id / "meta.json"
    if not (npz.exists() and meta_file.exists()):
        return
    with np.load(npz) as z:
        y_true, y_pred = z["y_true"], z["logits"].argmax(1)
    machine = pd.read_csv(ROOT / "splits" / "spain_test.csv", keep_default_na=False)["machine"]
    familiar = (machine != "Voluson S10").to_numpy()
    test = balanced_accuracy(y_true[familiar], y_pred[familiar])
    val = json.loads(meta_file.read_text())["best_val_balanced_acc"]
    flag = "" if abs(test - val) <= 0.05 else "   <-- more than 5 points apart, check this run"
    print(f"check {run_id}: balanced acc on spain_test without S10 {test:.4f} vs validation {val:.4f}{flag}")


def infer(run_id: str, condition_ids: list[str] | None, batch_size: int, force: bool) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    conditions = pd.read_csv(ROOT / "splits" / "conditions.csv", keep_default_na=False)
    if condition_ids:
        unknown = set(condition_ids) - set(conditions["condition_id"])
        if unknown:
            raise ValueError(f"Not in conditions.csv: {sorted(unknown)}")
        conditions = conditions[conditions["condition_id"].isin(condition_ids)]

    model, cfg = load_run(run_id, device)
    out_dir = DRIVE_ROOT / "outputs" / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    for condition in conditions.itertuples():
        if (out_dir / f"{condition.condition_id}.npz").exists() and not force:
            print(f"skip  {run_id} {condition.condition_id}: already saved")
            continue
        out = run_condition(model, cfg, condition, device, batch_size)
        save_outputs(out, run_id, condition.condition_id, out_dir)
        y_pred = out["logits"].argmax(1)
        print(f"ok    {run_id} {condition.condition_id:24s} {len(y_pred):5d} images | "
              f"acc {(out['y_true'] == y_pred).mean():.4f} | "
              f"balanced acc {balanced_accuracy(out['y_true'], y_pred):.4f}")
    sanity_check(run_id, out_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--run", nargs="+", required=True, help="run IDs, e.g. effb0_s0")
    parser.add_argument("--conditions", nargs="+", help="default: every condition in conditions.csv")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--force", action="store_true", help="overwrite files that already exist")
    args = parser.parse_args()
    for run in args.run:
        infer(run, args.conditions, args.batch_size, args.force)
    write_manifest(DRIVE_ROOT / "outputs")
