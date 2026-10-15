"""T4: train one run per YAML config and save contract C to the shared Drive.

    !python src/train.py --config configs/effb0_s0.yaml
    !python src/train.py --config configs/effb0_s*.yaml      # several runs in a row

Writes runs/<run_id>/ with best.pt (best validation epoch), config.yaml,
train_log.csv and meta.json, then rebuilds runs/summary.csv. A run that already
has meta.json is skipped, so rerunning the same command continues a batch.

While a run trains, its files (including last.pt, the resume checkpoint
rewritten every epoch) stay on the Colab machine's local disk and are copied to
Drive once, when the run finishes. Rewriting checkpoints on Drive every epoch
sends each old copy to the Drive trash, which fills the storage quota. The cost
is that a run interrupted by a lost session restarts from its first epoch; a
run interrupted while the machine is still alive resumes from last.pt.

--smoke runs two short epochs into runs/_smoke_<run_id>/ to check the whole
path (data, GPU, Drive writes) before spending compute.
"""
import argparse
import csv
import json
import os
import random
import shutil
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
from src.models import BACKBONES, build_model, load_state, normalization, saved_state  # noqa: E402
from src.paths import DRIVE_ROOT, LOCAL_ROOT  # noqa: E402

DEFAULTS = {"freeze_backbone": False, "grad_checkpointing": False, "dropout": 0.2, "lr": 1e-4, "weight_decay": 1e-4, "batch_size": 64,
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
    """Write to a temporary file first, so a crash mid-write never corrupts a checkpoint."""
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
    expected = f"{cfg['model']}{'_frozen' if cfg['freeze_backbone'] else ''}_s{seed}"
    if run_id != expected:
        raise ValueError(f"run_id '{run_id}' in {config_file} should be '{expected}'")
    if smoke:
        cfg.update(epochs=2, patience=2)

    runs_dir = DRIVE_ROOT / "runs"
    name = f"_smoke_{run_id}" if smoke else run_id
    run_dir = runs_dir / name                       # on Drive, written once at the end
    work_dir = LOCAL_ROOT / "runs" / name           # on local disk, written every epoch
    if (run_dir / "meta.json").exists() and not (force or smoke):
        print(f"skip {run_id}: already finished (use --force to retrain)")
        return
    work_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    seed_everything(seed)
    mean, std = normalization(cfg["model"])
    loader_args = dict(batch_size=cfg["batch_size"], num_workers=cfg["num_workers"],
                       image_size=cfg["image_size"], mean=mean, std=std)
    train_loader = make_loader("spain_train", train=True, seed=seed, **loader_args)
    val_loader = make_loader("spain_val", train=False, **loader_args)
    model = build_model(cfg["model"], NUM_CLASSES, cfg["dropout"],
                        freeze_backbone=cfg["freeze_backbone"]).to(device)
    if cfg["grad_checkpointing"]:
        # Recomputes activations in the backward pass, so a large ViT fits a T4 at batch 64
        model.backbone.set_grad_checkpointing(True)
    # Plain cross-entropy, no class weights: weighting distorts calibration
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                  lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    scaler = torch.amp.GradScaler(device.type, enabled=device.type == "cuda")

    state = {"epoch": 0, "best_epoch": 0, "best_val_balanced_acc": -1.0, "seconds": 0.0}
    last = work_dir / "last.pt"
    if last.exists() and not (force or smoke):
        checkpoint = torch.load(last, map_location=device)
        load_state(model, checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scaler.load_state_dict(checkpoint["scaler"])
        state = checkpoint["state"]
        print(f"resume {run_id} after epoch {state['epoch']}")
    else:
        (work_dir / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
        with open(work_dir / "train_log.csv", "w", newline="") as f:
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
            atomic_save({"model": saved_state(model), "epoch": epoch, "config": cfg}, work_dir / "best.pt")
        with open(work_dir / "train_log.csv", "a", newline="") as f:
            csv.writer(f).writerow([epoch, f"{train_loss:.5f}", f"{val_loss:.5f}",
                                    f"{val_bacc:.5f}", optimizer.param_groups[0]["lr"]])
        atomic_save({"model": saved_state(model), "optimizer": optimizer.state_dict(),
                     "scaler": scaler.state_dict(), "state": state}, last)
        print(f"  epoch {epoch:2d} | train loss {train_loss:.4f} | val loss {val_loss:.4f} "
              f"| val balanced acc {val_bacc:.4f}" + ("  *" if state["best_epoch"] == epoch else ""))

    meta = {
        "run_id": run_id, "model": cfg["model"], "weights": BACKBONES[cfg["model"]],
        "freeze_backbone": cfg["freeze_backbone"], "seed": seed,
        "best_epoch": state["best_epoch"],
        "best_val_balanced_acc": round(state["best_val_balanced_acc"], 5),
        "epochs_run": state["epoch"], "stopped_early": stopped_early,
        "training_minutes": round(state["seconds"] / 60, 1),
        "n_train": len(train_loader.dataset), "n_val": len(val_loader.dataset),
        "device": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu",
        "torch": torch.__version__, "git_commit": git_commit(),
        "date": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
    }
    # Copy the finished run to Drive; meta.json goes last because it marks the run as complete
    run_dir.mkdir(parents=True, exist_ok=True)
    for file in ["best.pt", "config.yaml", "train_log.csv"]:
        shutil.copyfile(work_dir / file, run_dir / file)
    (run_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    last.unlink(missing_ok=True)
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
