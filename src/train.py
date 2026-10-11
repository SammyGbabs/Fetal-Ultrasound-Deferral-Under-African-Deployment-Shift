"""T4: train one run per YAML config and save contract C to the shared Drive.

    !python src/train.py --config configs/effb0_s0.yaml
    !python src/train.py --config configs/effb0_s*.yaml      # several runs in a row

Writes runs/<run_id>/ with best.pt (best validation epoch), last.pt (resume
checkpoint, rewritten every epoch), config.yaml, train_log.csv and meta.json,
then rebuilds runs/summary.csv. A run that already has meta.json is skipped, and
a run with only last.pt resumes from it, so a dropped Colab session costs at
most one epoch: just rerun the same command.

--smoke runs two short epochs into runs/_smoke_<run_id>/ to check the whole
path (data, GPU, Drive writes) before spending compute.
"""
import argparse
import csv
import json
import os
import random
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.data import make_loader  # noqa: E402
from src.models import BACKBONES, build_model  # noqa: E402
from src.paths import DRIVE_ROOT  # noqa: E402

DEFAULTS = {"dropout": 0.2, "lr": 1e-4, "weight_decay": 1e-4, "batch_size": 64,
            "epochs": 30, "patience": 5, "image_size": 224, "num_workers": 2}
LOG_COLUMNS = ["epoch", "train_loss", "val_loss", "val_balanced_acc", "lr"]
NUM_CLASSES = 4


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def git_commit() -> str:
    """Current commit, with '-dirty' appended if tracked files have uncommitted changes."""
    def git(*args):
        return subprocess.run(["git", "-C", str(ROOT), *args], capture_output=True, text=True)
    commit = git("rev-parse", "--short", "HEAD").stdout.strip() or "unknown"
    dirty = git("status", "--porcelain", "--untracked-files=no").stdout.strip()
    return commit + ("-dirty" if dirty else "")


def atomic_save(obj, path: Path) -> None:
    """Write to a temporary file first, so a session dying mid-write never corrupts a checkpoint."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


def balanced_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean recall over the classes present in y_true."""
    recalls = [(y_pred[y_true == c] == c).mean() for c in np.unique(y_true)]
    return float(np.mean(recalls))


@torch.no_grad()
def validate(model, loader, criterion, device) -> tuple[float, float]:
    model.eval()
    loss_sum, y_true, y_pred = 0.0, [], []
    for batch in loader:
        x, y = batch["image"].to(device), batch["label"].to(device)
        logits = model(x).float()
        loss_sum += criterion(logits, y).item() * len(y)
        y_true.append(y.cpu().numpy())
        y_pred.append(logits.argmax(1).cpu().numpy())
    y_true, y_pred = np.concatenate(y_true), np.concatenate(y_pred)
    return loss_sum / len(y_true), balanced_accuracy(y_true, y_pred)


def train_one_epoch(model, loader, criterion, optimizer, scaler, device, max_batches=None) -> float:
    model.train()
    loss_sum, n = 0.0, 0
    for i, batch in enumerate(loader):
        if max_batches is not None and i >= max_batches:
            break
        x, y = batch["image"].to(device, non_blocking=True), batch["label"].to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
            loss = criterion(model(x), y)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        loss_sum += loss.item() * len(y)
        n += len(y)
    return loss_sum / n


def write_summary(runs_dir: Path) -> None:
    """Rebuild runs/summary.csv from every finished run's meta.json (smoke runs excluded)."""
    rows = [json.loads(p.read_text()) for p in sorted(runs_dir.glob("*/meta.json"))
            if not p.parent.name.startswith("_")]
    if rows:
        pd.DataFrame(rows).to_csv(runs_dir / "summary.csv", index=False)


def train(config_file: Path, smoke: bool = False, force: bool = False) -> None:
    cfg = yaml.safe_load(config_file.read_text())
    for key, value in DEFAULTS.items():
        cfg.setdefault(key, value)
    run_id, seed = cfg["run_id"], int(cfg["seed"])
    if run_id != f"{cfg['model']}_s{seed}":
        raise ValueError(f"run_id '{run_id}' does not match model and seed in {config_file}")
    if smoke:
        cfg.update(epochs=2, patience=2)

    runs_dir = DRIVE_ROOT / "runs"
    run_dir = runs_dir / (f"_smoke_{run_id}" if smoke else run_id)
    if (run_dir / "meta.json").exists() and not (force or smoke):
        print(f"skip {run_id}: already finished (use --force to retrain)")
        return
    run_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    seed_everything(seed)
    train_loader = make_loader("spain_train", train=True, batch_size=cfg["batch_size"],
                               num_workers=cfg["num_workers"], image_size=cfg["image_size"], seed=seed)
    val_loader = make_loader("spain_val", train=False, batch_size=cfg["batch_size"],
                             num_workers=cfg["num_workers"], image_size=cfg["image_size"])
    model = build_model(cfg["model"], NUM_CLASSES, cfg["dropout"]).to(device)
    # Plain cross-entropy, no class weights: weighting distorts calibration
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    scaler = torch.amp.GradScaler(device.type, enabled=device.type == "cuda")

    state = {"epoch": 0, "best_epoch": 0, "best_val_balanced_acc": -1.0, "seconds": 0.0}
    last = run_dir / "last.pt"
    if last.exists() and not (force or smoke):
        checkpoint = torch.load(last, map_location=device)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scaler.load_state_dict(checkpoint["scaler"])
        state = checkpoint["state"]
        print(f"resume {run_id} after epoch {state['epoch']}")
    else:
        (run_dir / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
        with open(run_dir / "train_log.csv", "w", newline="") as f:
            csv.writer(f).writerow(LOG_COLUMNS)

    print(f"train {run_id} on {device} | {len(train_loader.dataset)} train, "
          f"{len(val_loader.dataset)} val images | {run_dir}")
    stopped_early = False
    for epoch in range(state["epoch"] + 1, cfg["epochs"] + 1):
        if epoch - 1 - state["best_epoch"] >= cfg["patience"]:
            stopped_early = True
            break
        start = time.time()
        # Reseed per epoch so a resumed run sees the same shuffling and augmentation
        seed_everything(seed * 1000 + epoch)
        train_loader.generator.manual_seed(seed * 1000 + epoch)
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, scaler, device,
                                     max_batches=3 if smoke else None)
        val_loss, val_bacc = validate(model, val_loader, criterion, device)

        state["epoch"] = epoch
        state["seconds"] += time.time() - start
        if val_bacc > state["best_val_balanced_acc"]:
            state["best_epoch"], state["best_val_balanced_acc"] = epoch, val_bacc
            atomic_save({"model": model.state_dict(), "epoch": epoch, "config": cfg}, run_dir / "best.pt")
        with open(run_dir / "train_log.csv", "a", newline="") as f:
            csv.writer(f).writerow([epoch, f"{train_loss:.5f}", f"{val_loss:.5f}",
                                    f"{val_bacc:.5f}", optimizer.param_groups[0]["lr"]])
        atomic_save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                     "scaler": scaler.state_dict(), "state": state}, last)
        print(f"  epoch {epoch:2d} | train loss {train_loss:.4f} | val loss {val_loss:.4f} "
              f"| val balanced acc {val_bacc:.4f}" + ("  *" if state["best_epoch"] == epoch else ""))

    meta = {
        "run_id": run_id, "model": cfg["model"], "weights": BACKBONES[cfg["model"]], "seed": seed,
        "best_epoch": state["best_epoch"],
        "best_val_balanced_acc": round(state["best_val_balanced_acc"], 5),
        "epochs_run": state["epoch"], "stopped_early": stopped_early,
        "training_minutes": round(state["seconds"] / 60, 1),
        "n_train": len(train_loader.dataset), "n_val": len(val_loader.dataset),
        "device": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu",
        "torch": torch.__version__, "git_commit": git_commit(),
        "date": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
    }
    (run_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    last.unlink(missing_ok=True)       # the run is finished; only best.pt is kept
    write_summary(runs_dir)
    print(f"done {run_id}: best epoch {meta['best_epoch']}, "
          f"val balanced acc {meta['best_val_balanced_acc']:.4f}, {meta['training_minutes']} min")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--config", nargs="+", required=True, type=Path)
    parser.add_argument("--smoke", action="store_true", help="two short epochs into runs/_smoke_<run_id>/")
    parser.add_argument("--force", action="store_true", help="retrain a finished run from scratch")
    args = parser.parse_args()
    for config in args.config:
        train(config, smoke=args.smoke, force=args.force)
