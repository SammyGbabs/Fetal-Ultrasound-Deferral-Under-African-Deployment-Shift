"""Single place that resolves where data lives.

Every CSV stores paths relative to the project root (e.g. data/raw/...).
On Colab, images are unzipped to local disk for speed, while outputs are
written to the shared Drive. Override with environment variables if needed.
"""
import os
from pathlib import Path

# Shared Google Drive folder (same path for Samuel and Akachi via a shortcut)
DRIVE_ROOT = Path(os.environ.get("DRIVE_ROOT", "/content/drive/MyDrive/fetal-shift-drive"))

# Fast local copy of the data on the Colab machine
LOCAL_ROOT = Path(os.environ.get("LOCAL_ROOT", "/content/fetal-shift-local"))


def data_path(relative: str) -> Path:
    """Resolve a path from a split CSV: prefer the local copy, fall back to Drive."""
    local = LOCAL_ROOT / relative
    return local if local.exists() else DRIVE_ROOT / relative
