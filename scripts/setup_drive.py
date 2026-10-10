"""One-time setup of the shared Drive folder (run once, by Samuel, in Colab).

Creates the agreed folder layout, downloads both datasets from Zenodo into
data/zips/, and extracts them into data/raw/ on Drive.

Usage (in Colab, after mounting Drive):
    python scripts/setup_drive.py            # create folders only
    python scripts/setup_drive.py --download # also download and extract data
"""
import argparse
import shutil
import sys
import tarfile
import zipfile
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.paths import DRIVE_ROOT  # noqa: E402

# Zenodo record IDs (both CC BY 4.0)
DATASETS = {
    "spain": 3904280,   # FETAL_PLANES_DB (Burgos-Artizzu et al., 2020)
    "africa": 7540448,  # Sendra-Balcells et al., 2023
}

FOLDERS = ["data/zips", "data/raw", "data/compressed", "runs", "outputs", "eval"]


def make_folders() -> None:
    for folder in FOLDERS:
        (DRIVE_ROOT / folder).mkdir(parents=True, exist_ok=True)
        print(f"ok  {DRIVE_ROOT / folder}")


def download_record(name: str, record_id: int) -> list[Path]:
    """Download every file of a Zenodo record via the public API."""
    meta = requests.get(f"https://zenodo.org/api/records/{record_id}", timeout=60)
    meta.raise_for_status()
    files = meta.json()["files"]
    saved = []
    for f in files:
        key = f["key"]
        url = f["links"]["self"]
        target = DRIVE_ROOT / "data/zips" / name / key
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists() and target.stat().st_size == f.get("size", -1):
            print(f"skip {target.name} (already downloaded)")
        else:
            print(f"get  {key} ({f.get('size', 0) / 1e6:.0f} MB)")
            with requests.get(url, stream=True, timeout=600) as r:
                r.raise_for_status()
                with open(target, "wb") as out:
                    shutil.copyfileobj(r.raw, out)
        saved.append(target)
    return saved


def extract(archive: Path, dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    if archive.suffix == ".zip":
        with zipfile.ZipFile(archive) as z:
            z.extractall(dest)
    elif archive.name.endswith((".tar.xz", ".tar.gz", ".tar")):
        with tarfile.open(archive) as t:
            t.extractall(dest)
    else:  # CSV, XLSX or other loose files: copy as they are
        shutil.copy2(archive, dest / archive.name)
    print(f"ok  extracted {archive.name} -> {dest}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--download", action="store_true")
    args = parser.parse_args()

    make_folders()
    if args.download:
        for name, record_id in DATASETS.items():
            for archive in download_record(name, record_id):
                extract(archive, DRIVE_ROOT / "data/raw" / name)
    print("\nDone. Check data/raw/spain and data/raw/africa on Drive.")


if __name__ == "__main__":
    main()
